#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>

/* MobileNetModel.hpp defines USE_SPLIT_MODEL — must be included before
 * inference_mngt.h so the conditional enum values are compiled in. */
#include "MobileNetModel.hpp"
#include "inference_mngt.h"
#include "../GlobalState.hpp"
#include "LogConfig.hpp"

#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
#include "ZOTrainer.hpp"
#endif

#undef PI
#include "NuMicro.h"
#include "log_macros.h"

#if SINGLE_MODEL_USE_HYPERRAM || SPLIT_MODEL_USE_HYPERRAM
/* Some modes place the tensor arena(s) in external HyperRAM (memory-mapped at
 * 0x82000000 via SPIM DMM): the CPU single model (does not fit on-chip SRAM),
 * plus the NPU/split HyperRAM observation modes. */
#include "hyperflash_code.h"   /* SPIM_HyperRAM_Init() */
#endif

#include "BufAttributes.hpp"
#include "Labels.hpp"

#ifndef CPU_ACTIVATION_BUF_SZ
    #define CPU_ACTIVATION_BUF_SZ  0x00020000  /* 128 KB */
#endif

#if defined (__USE_DISPLAY__)
    #include "Display.h"
#endif
#if defined(__PROFILE__)
    #include "Profiler.hpp"
#endif
#include "pmu_counter.h"

/* ------------------------------------------------------------------ */
/*  External / Shared defines                                         */
/* ------------------------------------------------------------------ */
namespace arm
{
namespace app
{
extern uint8_t tensorArena[ACTIVATION_BUF_SZ];
extern uint8_t cpuTensorArena[];

#if !(defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1))
/* Optional getter function for the model pointer and its size.
 * SINGLE_MODEL_NS resolves to `mobilenet` (NPU/vela) or `baseline_w035`
 * (CPU int8) depending on MODEL_MODE — see MobileNetModel.hpp. */
namespace SINGLE_MODEL_NS
{
extern const uint8_t *GetModelPointer();
extern size_t GetModelLen();
} /* namespace SINGLE_MODEL_NS */
#endif /* !USE_SPLIT_MODEL */

} /* namespace app */
} /* namespace arm */

extern char fb_array[]; /* frame buffer array defined in main.cpp */

/* ------------------------------------------------------------------ */
/*  Split model namespace externs                                     */
/* ------------------------------------------------------------------ */
#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
namespace arm {
namespace app {
namespace feature_extractor_npu {
    extern const uint8_t *GetModelPointer();
    extern size_t GetModelLen();
}
namespace classifier_cpu_int8 {
    extern const uint8_t *GetModelPointer();
    extern size_t GetModelLen();
}
} /* namespace app */
} /* namespace arm */
#endif /* USE_SPLIT_MODEL */

/* ------------------------------------------------------------------ */
/*  State Machine Variables                                           */
/* ------------------------------------------------------------------ */
static ISM_State_t g_currentState = ISM_IDLE;
static ISM_State_t g_requestState = ISM_IDLE;
static bool g_modelLoadFailed = false; /* set after Init() fails; prevents crash-inducing retry */

#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
static bool g_zoLoadedFromFlash = false;
static uint32_t g_zoPersistedTrainSteps = 0;
#endif

/* ------------------------------------------------------------------ */
/*  Helper Functions (Internal)                                       */
/* ------------------------------------------------------------------ */

static std::string BuildTopKResultString(const std::vector<arm::app::ClassificationResult> &vec,
                                         size_t topK)
{
    if (vec.empty())
    {
        return std::string("???");
    }

    const size_t count = (vec.size() < topK) ? vec.size() : topK;
    std::string text;

    for (size_t i = 0; i < count; ++i)
    {
        if (i > 0)
        {
            text += std::string(", ");
        }
        text += vec[i].m_label + std::string(" ") + std::to_string(vec[i].m_normalisedVal);
    }

    return text;
}

/* ------------------------------------------------------------------ */
/*  Center-crop ROI helper                                            */
/* ------------------------------------------------------------------ */
/* A centered SQUARE ROI scaled to the square model input fixes the
 * 4:3 (320x240) -> 1:1 (224x224) aspect distortion that plain full-frame
 * stretching causes. The square crop itself is mandatory; only the zoom
 * factor below is a tuning knob.
 *
 * kCropFraction = fraction of the shorter side kept as the square crop:
 *   224/256 = 0.875  -> exactly matches PC Resize(256)+CenterCrop(224),
 *                       but zooms in and may clip an object that already
 *                       fills the camera frame.
 *   1.0              -> keeps the full shorter side (only the unavoidable
 *                       4:3->1:1 horizontal excess is dropped, 0% vertical),
 *                       never cuts into the object. Best when the camera
 *                       already frames the object near full-frame.
 * Start at 1.0; A/B test against 0.875 if accuracy still lags. */
static constexpr float kCropFraction = 1.0f;

static void SetCenterCropRoi(rectangle_t &r, const image_t &frame)
{
    const int shortSide = (frame.w < frame.h) ? frame.w : frame.h;
    int crop = (int)((float)shortSide * kCropFraction + 0.5f);
    if (crop < 1)         crop = shortSide;   /* degenerate-size guard */
    if (crop > shortSide) crop = shortSide;   /* never exceed the frame  */

    r.w = crop;
    r.h = crop;
    r.x = (frame.w - crop) / 2;               /* centered */
    r.y = (frame.h - crop) / 2;
}

/* ------------------------------------------------------------------ */
/*  HyperRAM tensor arena helper (shared by single + split paths)     */
/* ------------------------------------------------------------------ */
#if SINGLE_MODEL_USE_HYPERRAM || SPLIT_MODEL_USE_HYPERRAM
/* The full int8 MobileNetV2 run entirely on the CPU needs >1.1 MB of
 * activation arena, which does not fit in the 1 MB NPU-side SRAM. The
 * board carries an 8 MB HyperRAM that the SPIM exposes as normal,
 * memory-mapped RAM at 0x82000000 once put in Direct-Map mode, so the
 * arena can live there instead. The CPU single model needs this; the NPU
 * and split HyperRAM observation modes opt in to it deliberately to measure
 * arena-location effects (note the NPU may not be able to reach HyperRAM). */
#define HYPERRAM_ARENA_ADDR   (0x82000000UL)   /* SPIM0 DMM base (SPIM_HYPER_DMM0_SADDR) */
#if (MODEL_MODE) == MODEL_MODE_SINGLE_CPU_FP32
/* fp32 (float32) single CPU model: float activations are ~4x the int8
 * footprint, so the int8 ~1.2 MB peak grows to ~4.5 MB. Reserve 6 MB of the
 * 8 MB HyperRAM to give the greedy planner ample headroom. */
#define HYPERRAM_ARENA_SZ     (0x00800000UL)   /* 6 MB */
#else
#define HYPERRAM_ARENA_SZ     (0x00200000UL)   /* 2 MB — ample headroom over the ~1.2 MB int8 peak */
#endif

static bool g_hyperRamReady = false;

/* Bring up HyperRAM (clock, pins, DLL training, enter DMM) once, then do a
 * lightweight start/mid/end round-trip check on the memory-mapped arena. */
static bool EnsureHyperRamArena(void)
{
    if (g_hyperRamReady)
    {
        return true;
    }

    info_if_token(LOG_MODEL_LOAD, "[ISM] Initialising HyperRAM (SPIM0 Direct-Map)...\r\n");

    /* SPIM clock / pin MFP setup touch protected registers. */
    SYS_UnlockReg();
    SPIM_HyperRAM_Init(SPIM0);
    SYS_LockReg();

    /* Round-trip sanity test at start / middle / end of the arena. Clean+invalidate
     * the D-cache between write and read so the read really fetches from HyperRAM
     * rather than returning a cached copy. */
    volatile uint32_t *p   = (volatile uint32_t *)HYPERRAM_ARENA_ADDR;
    const uint32_t pattern = 0xDEADBEEFUL;
    const uint32_t words   = (uint32_t)(HYPERRAM_ARENA_SZ / sizeof(uint32_t));
    const uint32_t probes[3] = { 0u, words / 2u, words - 1u };

    for (int i = 0; i < 3; i++) { p[probes[i]] = pattern; }
    SCB_CleanInvalidateDCache();
    for (int i = 0; i < 3; i++)
    {
        uint32_t got = p[probes[i]];
        if (got != pattern)
        {
            printf_err("[ISM] HyperRAM sanity FAILED @word %u: got 0x%08X, expected 0x%08X\r\n",
                       (unsigned)probes[i], (unsigned)got, (unsigned)pattern);
            return false;
        }
    }

    g_hyperRamReady = true;
    info_if_token(LOG_MODEL_LOAD, "[ISM] HyperRAM ready @0x%08X (%u bytes)\r\n",
                  (unsigned)HYPERRAM_ARENA_ADDR, (unsigned)HYPERRAM_ARENA_SZ);
    return true;
}
#endif /* SINGLE_MODEL_USE_HYPERRAM || SPLIT_MODEL_USE_HYPERRAM */

#if !(defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1))

static void DoLoadModel(void)
{
    if (modelLoaded)
    {
        info("[ISM] Model already loaded\r\n");
        return;
    }

    if (g_modelLoadFailed)
    {
        printf_err("[ISM] Model previously failed to load. Reset MCU to retry.\r\n");
        return;
    }

    info_if_token(LOG_MODEL_LOAD, "[ISM] Loading model...\r\n");

    extern uint32_t SystemCoreClock;
    const uint64_t t0 = pmu_get_systick_Count();

    const uint8_t *modelPtr = arm::app::SINGLE_MODEL_NS::GetModelPointer();
    size_t         modelLen = arm::app::SINGLE_MODEL_NS::GetModelLen();
    const tflite::Model *fbModel = ::tflite::GetModel(modelPtr);

#if SINGLE_MODEL_USE_HYPERRAM
    /* Arena lives in external HyperRAM (CPU model does not fit on-chip SRAM;
     * the NPU HyperRAM mode places it there deliberately for observation). */
    if (!EnsureHyperRamArena())
    {
        printf_err("[ISM] HyperRAM unavailable; cannot load model\r\n");
        g_modelLoadFailed = true;
        return;
    }
    uint8_t *arenaPtr  = (uint8_t *)HYPERRAM_ARENA_ADDR;
    uint32_t arenaSize = (uint32_t)HYPERRAM_ARENA_SZ;
    const uint8_t arenaMpuAttr = eMPU_ATTR_CACHEABLE_WBWARA; /* write-back: best for slow external RAM */
    info_if_token(LOG_MODEL_LOAD, "[ISM] Using HyperRAM tensor arena @0x%08X (%u bytes)\r\n",
                  (unsigned)arenaPtr, arenaSize);
#else
    uint8_t *arenaPtr  = arm::app::tensorArena;
    uint32_t arenaSize = sizeof(arm::app::tensorArena);
    const uint8_t arenaMpuAttr = eMPU_ATTR_CACHEABLE_WTRA;   /* SRAM */
    info_if_token(LOG_MODEL_LOAD, "[ISM] Using SRAM tensor arena (%u bytes)\r\n", arenaSize);
#endif

    /* Dump all operator types required by this model for diagnostics */
    if (fbModel && fbModel->operator_codes())
    {
        const auto *opcodes = fbModel->operator_codes();
        info_if_token(LOG_MODEL_LOAD, "[ISM] Model requires %u operator type(s):\r\n", (unsigned)opcodes->size());
        for (uint32_t i = 0; i < opcodes->size(); i++)
        {
            const auto *opcode = opcodes->Get(i);
            auto builtinCode = tflite::GetBuiltinCode(opcode);
            if (builtinCode == tflite::BuiltinOperator_CUSTOM)
            {
                const char *name = opcode->custom_code() ? opcode->custom_code()->c_str() : "CUSTOM";
                info_if_token(LOG_MODEL_LOAD, "[ISM]   [%u] CUSTOM: %s\r\n", i, name);
            }
            else
            {
                info_if_token(LOG_MODEL_LOAD, "[ISM]   [%u] %s\r\n", i, tflite::EnumNameBuiltinOperator(builtinCode));
            }
        }
    }

    /* Check quantization of CONV_2D / DEPTHWISE_CONV_2D filter tensors */
    if (fbModel && fbModel->subgraphs() && fbModel->subgraphs()->size() > 0)
    {
        const auto *subgraph = fbModel->subgraphs()->Get(0);
        const auto *operators = subgraph->operators();
        const auto *tensors   = subgraph->tensors();
        const auto *opcodes   = fbModel->operator_codes();

        if (operators && tensors && opcodes)
        {
            info_if_token(LOG_MODEL_LOAD, "[ISM] Checking conv/fc filter quantization...\r\n");
            for (uint32_t i = 0; i < operators->size(); i++)
            {
                const auto *op = operators->Get(i);
                const auto *opcode = opcodes->Get(op->opcode_index());
                auto code = tflite::GetBuiltinCode(opcode);

                if (code != tflite::BuiltinOperator_CONV_2D &&
                    code != tflite::BuiltinOperator_DEPTHWISE_CONV_2D &&
                    code != tflite::BuiltinOperator_FULLY_CONNECTED)
                    continue;

                const char *opName = tflite::EnumNameBuiltinOperator(code);

                /* inputs[1] = filter tensor */
                if (op->inputs() && op->inputs()->size() >= 2)
                {
                    int filterIdx = op->inputs()->Get(1);
                    const auto *filterT = tensors->Get(filterIdx);
                    const char *tType = tflite::EnumNameTensorType(filterT->type());

                    int scaleCount = 0;
                    if (filterT->quantization() && filterT->quantization()->scale())
                        scaleCount = (int)filterT->quantization()->scale()->size();

                    int shapeOC = 0;
                    if (filterT->shape())
                    {
                        const unsigned int ocDim = (code == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) ? 3u : 0u;
                        if (filterT->shape()->size() > ocDim)
                        {
                            shapeOC = filterT->shape()->Get(ocDim);
                        }
                    }

                    info_if_token(LOG_MODEL_LOAD, "[ISM]   OP[%u] %s filter=T[%d] type=%s OC=%d scales=%d %s\r\n",
                         i, opName, filterIdx, tType, shapeOC, scaleCount,
                        (scaleCount == shapeOC) ? "PER-CHANNEL" :
                        (scaleCount == 1)       ? "PER-TENSOR" : "UNKNOWN(!)");

                    if (code == tflite::BuiltinOperator_FULLY_CONNECTED && scaleCount != 1)
                    {
                        printf_err("[ISM]   FULLY_CONNECTED filter is not per-tensor (scales=%d). "
                                "This TFLM build requires scales=1 for FC.\r\n",
                                scaleCount);
                    }
                }

                /* inputs[2] = bias tensor */
                if (op->inputs() && op->inputs()->size() >= 3)
                {
                    int biasIdx = op->inputs()->Get(2);
                    if (biasIdx >= 0)
                    {
                        const auto *biasT = tensors->Get(biasIdx);
                        int biasDims = biasT->shape() ? (int)biasT->shape()->size() : 0;
                        info_if_token(LOG_MODEL_LOAD, "[ISM]         bias=T[%d] type=%s dims=%dD shape=[",
                             biasIdx, tflite::EnumNameTensorType(biasT->type()), biasDims);
                        if (IsLogTokenEnabled(LOG_MODEL_LOAD) && biasT->shape()) {
                            for (int s = 0; s < (int)biasT->shape()->size(); s++) {
                                if (s > 0) info_if_token(LOG_MODEL_LOAD, ",");
                                info_if_token(LOG_MODEL_LOAD, "%d", biasT->shape()->Get(s));
                            }
                        }
                        info_if_token(LOG_MODEL_LOAD, "]\r\n");
                    }
                }
            }
        }
    }

    if (!model.Init(arenaPtr, arenaSize, modelPtr, modelLen))
    {
        printf_err("[ISM] Failed to initialise model\r\n");
        g_modelLoadFailed = true;
        return;
    }

    /* Setup cache policy of tensor arena buffer */
    const std::vector<ARM_MPU_Region_t> mpuConfig =
    {
        {
            // Tensor arena (SRAM for NPU model, HyperRAM for CPU model)
            ARM_MPU_RBAR(((unsigned int)arenaPtr),             // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)arenaPtr) + arenaSize - 1),  // Limit
                         arenaMpuAttr)      // WTRA for SRAM, WBWARA for HyperRAM
        },
#if defined (__USE_CCAP__)
        {
            // Image data from CCAP DMA => Non-cacheable
             ARM_MPU_RBAR(((unsigned int)fb_array),        // Base
                          ARM_MPU_SH_NON,    // Non-shareable
                          0,                 // Read-only
                          1,                 // Non-Privileged
                          1),                // eXecute Never enabled
             ARM_MPU_RLAR((((unsigned int)fb_array) + OMV_FB_SIZE - 1),        // Limit
                          eMPU_ATTR_NON_CACHEABLE) // NonCache
        }
#endif
    };

    // Setup MPU configuration
    InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());

    inputTensor = model.GetInputTensor(0);

    if (!inputTensor->dims || inputTensor->dims->size < 3)
    {
        printf_err("[ISM] Invalid input tensor dims\r\n");
    }
    else
    {
        TfLiteIntArray *inputShape = model.GetInputShape(0);
        inputImgCols   = inputShape->data[arm::app::MobileNetModel::ms_inputColsIdx];
        inputImgRows   = inputShape->data[arm::app::MobileNetModel::ms_inputRowsIdx];
        inputChannels  = inputShape->data[arm::app::MobileNetModel::ms_inputChannelsIdx];

        GetLabelsVector(labels);

        preProcess  = new arm::app::ImgClassPreProcess(&model);
        postProcess = new arm::app::ImgClassPostProcess(classifier, &model,
                                                        labels, results);

        modelLoaded = true;

        size_t arenaUsed = model.GetArenaUsedBytes();
        uint64_t elapsed = pmu_get_systick_Count() - t0;
        float elapsedMs = (float)elapsed * 1000.0f / (float)SystemCoreClock;
        info_critical("[LOAD] time=%.2fms mem=%zu/%uB zo=N/A\r\n",
                  elapsedMs, arenaUsed, arenaSize);
        info_if_token(LOG_MODEL_INIT, "[ISM] Model loaded success\r\n");
    }
}

static void DoInference(void)
{
    if (!modelLoaded)
    {
        info("[ISM] Model not loaded, loading now...\r\n");
        DoLoadModel();
        if (!modelLoaded) return;
    }

    /* Freeze frameBuffer to ensure consistent inference frame */
    extern uint32_t inferenceFrameCount;
    extern bool frameBufferFrozen;
    inferenceFrameCount++;
    frameBufferFrozen = true;

    /* Resize framebuffer image to model input */
    image_t resizeImg;

    /* Center-crop (match PC Resize(256)+CenterCrop(224)) instead of
     * stretching the whole frame into the square model input. */
    SetCenterCropRoi(roi, frameBuffer);

    resizeImg.w = inputImgCols;
    resizeImg.h = inputImgRows;
    resizeImg.data = (uint8_t *)inputTensor->data.data; // direct resize to input tensor buffer
    resizeImg.pixfmt = PIXFORMAT_RGB888;

    imlib_nvt_scale(&frameBuffer, &resizeImg, &roi);

    /* Pre-processing */
    if (!preProcess->DoPreProcess(resizeImg.data, (resizeImg.w * resizeImg.h * inputChannels)))
    {
        printf_err("[ISM] Pre-processing failed\r\n");
        extern bool frameBufferFrozen;
        frameBufferFrozen = false;
        return;
    }

    /* Run inference with timing */
    extern uint32_t SystemCoreClock;
    uint64_t t0 = pmu_get_systick_Count();

    if (!model.RunInference())
    {
        printf_err("[ISM] Inference failed\r\n");
    }
    else
    {
        uint64_t elapsed = pmu_get_systick_Count() - t0;
        uint32_t elapsed_us = (uint32_t)(elapsed * 1000000ULL / SystemCoreClock);

        /* Post-processing */
        std::string predictLabelInfo;
        std::string top5LabelInfo;

        if (postProcess->DoPostProcess())
        {
            top5LabelInfo = BuildTopKResultString(results, 5);
            predictLabelInfo = BuildTopKResultString(results, 1);
        }
        else
        {
            top5LabelInfo = std::string("???");
            predictLabelInfo = std::string("???");
        }

        /* Report result with timing via UART */
        char timingBuf[16];
        snprintf(timingBuf, sizeof(timingBuf), "%.2fms", elapsed_us / 1000.0f);
        info_critical("[FRAME: %u] [RESULT] %s (%s)\r\n", inferenceFrameCount, top5LabelInfo.c_str(), timingBuf);

        /* Update global result string for persistent display in main loop */
        lastInferenceResult = predictLabelInfo + std::string(" (") + timingBuf + std::string(")");
    }

    /* Unfreeze frameBuffer after inference completes */
    frameBufferFrozen = false;
}

#endif /* !USE_SPLIT_MODEL */

/* ------------------------------------------------------------------ */
/*  Split-model: Load & Inference                                     */
/* ------------------------------------------------------------------ */
#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)

namespace {

constexpr uint32_t kZoFlashMagic = 0x5A4F5452UL; /* 'ZOTR' */
constexpr uint32_t kZoFlashVersion = 1UL;
constexpr uint32_t kZoFlashReservedPages = 4UL;
constexpr uint32_t kZoFlashRegionAddr = FMC_APROM_END - (kZoFlashReservedPages * FMC_FLASH_PAGE_SIZE);
constexpr size_t kZoFlashRegionBytes = (size_t)kZoFlashReservedPages * FMC_FLASH_PAGE_SIZE;

struct ZoFlashHeader
{
    uint32_t magic;
    uint32_t version;
    uint32_t payloadBytes;
    uint32_t stepCount;
    uint32_t inputFeatures;
    uint32_t outputClasses;
    uint32_t payloadChecksum;
    uint32_t reserved;
};

static uint32_t CalcFnv1a(const uint8_t *data, size_t len)
{
    uint32_t hash = 2166136261UL;

    for (size_t i = 0; i < len; ++i)
    {
        hash ^= data[i];
        hash *= 16777619UL;
    }

    return hash;
}

static void FlashReadBytes(uint32_t addr, uint8_t *dst, size_t len)
{
    if (!dst || len == 0)
    {
        return;
    }

    /* APROM is memory mapped, so direct read is robust across resets and
     * does not depend on FMC ISP enable state. */
    const uint8_t *src = reinterpret_cast<const uint8_t *>(addr);
    std::memcpy(dst, src, len);
}

static bool FlashWriteBytes(uint32_t addr, const uint8_t *src, size_t len)
{
    if (!src)
    {
        return false;
    }

    int32_t rc = 0;
    bool ok = false;

    SYS_UnlockReg();
    FMC_Open();
    FMC_ENABLE_AP_UPDATE();

    size_t written = 0;
    while (written < len)
    {
        const uint32_t pageAddr = addr + (uint32_t)written;
        size_t pageBytes = len - written;
        if (pageBytes > FMC_FLASH_PAGE_SIZE)
        {
            pageBytes = FMC_FLASH_PAGE_SIZE;
        }

        rc = FMC_Erase(pageAddr);
        if (rc != 0)
        {
            goto cleanup;
        }

        for (size_t off = 0; off < pageBytes; off += 4)
        {
            uint32_t word = 0xFFFFFFFFUL;
            size_t chunk = ((pageBytes - off) >= 4) ? 4 : (pageBytes - off);
            std::memcpy(&word, src + written + off, chunk);

            rc = FMC_Write(pageAddr + (uint32_t)off, word);
            if (rc != 0)
            {
                goto cleanup;
            }
        }

        written += pageBytes;
    }

    ok = true;

cleanup:
    FMC_DISABLE_AP_UPDATE();
    FMC_Close();
    SYS_LockReg();
    return ok;
}

static bool FlashErasePages(uint32_t addr, size_t totalBytes)
{
    int32_t rc = 0;
    bool ok = false;

    if (totalBytes == 0 || (totalBytes % FMC_FLASH_PAGE_SIZE) != 0)
    {
        return false;
    }

    SYS_UnlockReg();
    FMC_Open();
    FMC_ENABLE_AP_UPDATE();

    for (size_t off = 0; off < totalBytes; off += FMC_FLASH_PAGE_SIZE)
    {
        rc = FMC_Erase(addr + (uint32_t)off);
        if (rc != 0)
        {
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    FMC_DISABLE_AP_UPDATE();
    FMC_Close();
    SYS_LockReg();
    return ok;
}

static bool SaveZOTrainerSnapshotToFlash(void)
{
    if (!zoTrainer || !zoTrainer->IsInitialized())
    {
        printf_err("[ZO] Cannot save: trainer not initialised\r\n");
        return false;
    }

    const ZOTrainer::FCInfo &fc = zoTrainer->GetFCInfo();
    const size_t payloadBytes = (size_t)fc.weight_bytes + (size_t)fc.bias_bytes;
    const size_t totalBytes = sizeof(ZoFlashHeader) + payloadBytes;

    if (totalBytes > kZoFlashRegionBytes)
    {
        printf_err("[ZO] Snapshot too large for reserved APROM region (%u > %u)\r\n",
                   (unsigned int)totalBytes, (unsigned int)kZoFlashRegionBytes);
        return false;
    }

    const int8_t *weights = zoTrainer->GetMutableWeights();
    const int32_t *bias = zoTrainer->GetMutableBias();
    if (!weights)
    {
        printf_err("[ZO] Cannot save: mutable weights pointer is null\r\n");
        return false;
    }

    std::vector<uint8_t> snapshot(totalBytes, 0x00);
    ZoFlashHeader hdr{};
    hdr.magic = kZoFlashMagic;
    hdr.version = kZoFlashVersion;
    hdr.payloadBytes = (uint32_t)payloadBytes;
    hdr.stepCount = (uint32_t)zoTrainer->GetStepCount();
    hdr.inputFeatures = (uint32_t)fc.input_features;
    hdr.outputClasses = (uint32_t)fc.output_classes;

    std::memcpy(snapshot.data() + sizeof(ZoFlashHeader), weights, (size_t)fc.weight_bytes);
    if (fc.bias_bytes > 0)
    {
        if (!bias)
        {
            printf_err("[ZO] Cannot save: mutable bias pointer is null\r\n");
            return false;
        }
        std::memcpy(snapshot.data() + sizeof(ZoFlashHeader) + (size_t)fc.weight_bytes,
                    bias, (size_t)fc.bias_bytes);
    }

    hdr.payloadChecksum = CalcFnv1a(snapshot.data() + sizeof(ZoFlashHeader), payloadBytes);
    std::memcpy(snapshot.data(), &hdr, sizeof(ZoFlashHeader));

    if (!FlashWriteBytes(kZoFlashRegionAddr, snapshot.data(), snapshot.size()))
    {
        printf_err("[ZO] Failed to save snapshot to APROM flash\r\n");
        return false;
    }

    g_zoPersistedTrainSteps = hdr.stepCount;
    info_critical("[ZO] Snapshot saved to flash @0x%08X (steps=%u)\r\n",
                  kZoFlashRegionAddr, hdr.stepCount);
    return true;
}

static bool TryLoadZOTrainerSnapshotFromFlash(void)
{
    if (!zoTrainer || !zoTrainer->IsInitialized())
    {
        return false;
    }

    ZoFlashHeader hdr{};
    FlashReadBytes(kZoFlashRegionAddr, reinterpret_cast<uint8_t *>(&hdr), sizeof(hdr));

    const ZOTrainer::FCInfo &fc = zoTrainer->GetFCInfo();
    const uint32_t expectedPayload = (uint32_t)((size_t)fc.weight_bytes + (size_t)fc.bias_bytes);

    if (hdr.magic != kZoFlashMagic || hdr.version != kZoFlashVersion)
    {
        info_if_token(LOG_ZO_TRAINING,
                      "[ZO] Flash header invalid: magic=0x%08X version=%u (expect 0x%08X/%u)\r\n",
                      hdr.magic, hdr.version, kZoFlashMagic, kZoFlashVersion);
        g_zoLoadedFromFlash = false;
        g_zoPersistedTrainSteps = 0;
        return false;
    }

    if (hdr.payloadBytes > (kZoFlashRegionBytes - sizeof(ZoFlashHeader)))
    {
        printf_err("[ZO] Flash snapshot payload too large; using built-in weights\r\n");
        g_zoLoadedFromFlash = false;
        g_zoPersistedTrainSteps = 0;
        return false;
    }

    if (hdr.payloadBytes != expectedPayload ||
        hdr.inputFeatures != (uint32_t)fc.input_features ||
        hdr.outputClasses != (uint32_t)fc.output_classes)
    {
        printf_err("[ZO] Flash snapshot header mismatch; using built-in weights\r\n");
        g_zoLoadedFromFlash = false;
        g_zoPersistedTrainSteps = 0;
        return false;
    }

    std::vector<uint8_t> payload(hdr.payloadBytes, 0);
    FlashReadBytes(kZoFlashRegionAddr + (uint32_t)sizeof(ZoFlashHeader),
                   payload.data(), payload.size());

    const uint32_t checksum = CalcFnv1a(payload.data(), payload.size());
    if (checksum != hdr.payloadChecksum)
    {
        printf_err("[ZO] Flash snapshot checksum mismatch; using built-in weights\r\n");
        g_zoLoadedFromFlash = false;
        g_zoPersistedTrainSteps = 0;
        return false;
    }

    const int8_t *w = reinterpret_cast<const int8_t *>(payload.data());
    const uint8_t *b = payload.data() + (size_t)fc.weight_bytes;

    if (!zoTrainer->LoadFromSnapshot(w, (size_t)fc.weight_bytes, b,
                                     (size_t)fc.bias_bytes, (int)hdr.stepCount))
    {
        printf_err("[ZO] Failed to apply flash snapshot; using built-in weights\r\n");
        g_zoLoadedFromFlash = false;
        g_zoPersistedTrainSteps = 0;
        return false;
    }

    g_zoLoadedFromFlash = true;
    g_zoPersistedTrainSteps = hdr.stepCount;
    info_critical("[ZO] Loaded classifier snapshot from flash (steps=%u)\r\n", hdr.stepCount);
    return true;
}

} // namespace

static void DoZOInit(void);

/* ------------------------------------------------------------------ */
/*  Split-model tensor arena placement                                */
/* ------------------------------------------------------------------ */
/* In SPLIT_HYPERRAM mode both arenas live back-to-back in the 2 MB HyperRAM
 * region (extractor first, classifier right after); otherwise the extractor
 * uses on-chip SRAM tensorArena and the classifier uses cpuTensorArena.
 * Defined as macros so DoLoadSplitModel() and DoZOInit() can never disagree
 * on where the classifier arena is. */
#if SPLIT_MODEL_USE_HYPERRAM
#if (ACTIVATION_BUF_SZ + CPU_ACTIVATION_BUF_SZ) > HYPERRAM_ARENA_SZ
#error "Split HyperRAM arenas (extractor + classifier) exceed the HyperRAM region"
#endif
#define SPLIT_EXT_ARENA_PTR   ((uint8_t *)HYPERRAM_ARENA_ADDR)
#define SPLIT_EXT_ARENA_SZ    ((uint32_t)ACTIVATION_BUF_SZ)
#define SPLIT_CLS_ARENA_PTR   ((uint8_t *)HYPERRAM_ARENA_ADDR + ACTIVATION_BUF_SZ)
#define SPLIT_CLS_ARENA_SZ    ((uint32_t)CPU_ACTIVATION_BUF_SZ)
#define SPLIT_ARENA_MPU_ATTR  eMPU_ATTR_CACHEABLE_WBWARA  /* write-back: best for slow external RAM */
#define SPLIT_ARENA_LOC_STR   "HyperRAM"
#else
#define SPLIT_EXT_ARENA_PTR   arm::app::tensorArena
#define SPLIT_EXT_ARENA_SZ    ((uint32_t)sizeof(arm::app::tensorArena))
#define SPLIT_CLS_ARENA_PTR   arm::app::cpuTensorArena
#define SPLIT_CLS_ARENA_SZ    ((uint32_t)CPU_ACTIVATION_BUF_SZ)
#define SPLIT_ARENA_MPU_ATTR  eMPU_ATTR_CACHEABLE_WTRA    /* SRAM */
#define SPLIT_ARENA_LOC_STR   "SRAM"
#endif

static void DoLoadSplitModel(void)
{
    if (splitModelLoaded)
    {
        info("[ISM] Split models already loaded\r\n");
        return;
    }

    if (g_modelLoadFailed)
    {
        printf_err("[ISM] Model previously failed to load. Reset MCU to retry.\r\n");
        return;
    }

    info_if_token(LOG_MODEL_LOAD, "[ISM] Loading split models (extractor + classifier)...\r\n");

    extern uint32_t SystemCoreClock;
    const uint64_t t0 = pmu_get_systick_Count();

#if SPLIT_MODEL_USE_HYPERRAM
    /* Both split arenas live in external HyperRAM (observation mode). */
    if (!EnsureHyperRamArena())
    {
        printf_err("[ISM] HyperRAM unavailable; cannot load split model\r\n");
        g_modelLoadFailed = true;
        return;
    }
#endif

    /* ---- Feature Extractor (NPU) ---- */
    const uint8_t *extPtr  = arm::app::feature_extractor_npu::GetModelPointer();
    size_t         extLen  = arm::app::feature_extractor_npu::GetModelLen();
    uint8_t       *extArena     = SPLIT_EXT_ARENA_PTR;
    uint32_t       extArenaSize = SPLIT_EXT_ARENA_SZ;

    info_if_token(LOG_MODEL_LOAD, "[ISM] Feature extractor: %zu bytes, %s arena %u bytes at 0x%p\r\n",
         extLen, SPLIT_ARENA_LOC_STR, extArenaSize, extArena);

    if (!extractorModel.Init(extArena, extArenaSize, extPtr, extLen))
    {
        printf_err("[ISM] Failed to init feature extractor model\r\n");
        g_modelLoadFailed = true;
        return;
    }

    /* ---- Classifier (CPU) ---- */
    const uint8_t *clsPtr  = arm::app::classifier_cpu_int8::GetModelPointer();
    size_t         clsLen  = arm::app::classifier_cpu_int8::GetModelLen();
    uint8_t       *clsArena     = SPLIT_CLS_ARENA_PTR;
    uint32_t       clsArenaSize = SPLIT_CLS_ARENA_SZ;

    info_if_token(LOG_MODEL_LOAD, "[ISM] Classifier: %zu bytes, %s arena %u bytes at 0x%p\r\n",
         clsLen, SPLIT_ARENA_LOC_STR, clsArenaSize, clsArena);

    if (!classifierModel.Init(clsArena, clsArenaSize, clsPtr, clsLen))
    {
        printf_err("[ISM] Failed to init classifier model\r\n");
        g_modelLoadFailed = true;
        return;
    }

    /* ---- Verify intermediate tensor compatibility ---- */
    TfLiteTensor *extOutput = extractorModel.GetOutputTensor(0);
    TfLiteTensor *clsInput  = classifierModel.GetInputTensor(0);

    info_if_token(LOG_MODEL_LOAD, "[ISM] Extractor output: %zu bytes, Classifier input: %zu bytes\r\n",
         extOutput->bytes, clsInput->bytes);

    if (extOutput->bytes != clsInput->bytes)
    {
        printf_err("[ISM] Shape mismatch! extractor output %zu != classifier input %zu\r\n",
                   extOutput->bytes, clsInput->bytes);
        g_modelLoadFailed = true;
        return;
    }

    /* ---- MPU: mark tensor arena(s) cacheable ----
     * SRAM mode covers only the extractor SRAM arena (classifier cpuTensorArena
     * keeps default attrs, as before). HyperRAM mode covers both back-to-back
     * arenas in one region with write-back attributes for the slow external RAM. */
#if SPLIT_MODEL_USE_HYPERRAM
    const unsigned int mpuArenaBase = (unsigned int)extArena;
    const unsigned int mpuArenaSize = (unsigned int)extArenaSize + (unsigned int)clsArenaSize;
#else
    const unsigned int mpuArenaBase = (unsigned int)extArena;
    const unsigned int mpuArenaSize = (unsigned int)extArenaSize;
#endif
    const std::vector<ARM_MPU_Region_t> mpuConfig =
    {
        {
            /* Tensor arena (extractor / NPU; +classifier in HyperRAM mode) */
            ARM_MPU_RBAR(mpuArenaBase,
                         ARM_MPU_SH_NON, 0, 1, 1),
            ARM_MPU_RLAR((mpuArenaBase + mpuArenaSize - 1),
                         SPLIT_ARENA_MPU_ATTR)
        },
#if defined (__USE_CCAP__)
        {
            ARM_MPU_RBAR(((unsigned int)fb_array),
                         ARM_MPU_SH_NON, 0, 1, 1),
            ARM_MPU_RLAR((((unsigned int)fb_array) + OMV_FB_SIZE - 1),
                         eMPU_ATTR_NON_CACHEABLE)
        }
#endif
    };
    InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());

    /* ---- Setup image input from extractor ---- */
    inputTensor = extractorModel.GetInputTensor(0);

    if (!inputTensor->dims || inputTensor->dims->size < 3)
    {
        printf_err("[ISM] Invalid extractor input tensor dims\r\n");
        g_modelLoadFailed = true;
        return;
    }

    TfLiteIntArray *inputShape = extractorModel.GetInputShape(0);
    inputImgCols  = inputShape->data[arm::app::MobileNetModel::ms_inputColsIdx];
    inputImgRows  = inputShape->data[arm::app::MobileNetModel::ms_inputRowsIdx];
    inputChannels = inputShape->data[arm::app::MobileNetModel::ms_inputChannelsIdx];

    GetLabelsVector(labels);

    /* Pre-process feeds the extractor; post-process reads the classifier */
    preProcess  = new arm::app::ImgClassPreProcess(&extractorModel);
    postProcess = new arm::app::ImgClassPostProcess(classifier, &classifierModel,
                                                    labels, results);

    splitModelLoaded = true;
    modelLoaded      = true; /* backward compat flag */

    g_zoLoadedFromFlash = false;
    g_zoPersistedTrainSteps = 0;

    /* Auto-restore persisted ZO snapshot on model load so reboot + load_model
     * immediately reuses trained classifier parameters if available. */
    DoZOInit();

    const size_t extUsed = extractorModel.GetArenaUsedBytes();
    const size_t clsUsed = classifierModel.GetArenaUsedBytes();
    const size_t totalUsed = extUsed + clsUsed;
    const size_t totalArena = (size_t)extArenaSize + (size_t)clsArenaSize;
    uint64_t elapsed = pmu_get_systick_Count() - t0;
    float elapsedMs = (float)elapsed * 1000.0f / (float)SystemCoreClock;
    const char *zoState = g_zoLoadedFromFlash ? "YES" : "NO";
    info_critical("[LOAD] time=%.2fms mem=%zu/%zuB zo=%s\r\n",
                  elapsedMs, totalUsed, totalArena, zoState);
    info_if_token(LOG_MODEL_INIT, "[ISM] Split models loaded successfully\r\n");
}

static void DoSplitInference(void)
{
    if (!splitModelLoaded)
    {
        info("[ISM] Split models not loaded, loading now...\r\n");
        DoLoadSplitModel();
        if (!splitModelLoaded) return;
    }

    /* Freeze frameBuffer to ensure consistent inference frame */
    extern uint32_t inferenceFrameCount;
    extern bool frameBufferFrozen;
    inferenceFrameCount++;
    frameBufferFrozen = true;

    /* ---- Resize camera frame → extractor input ---- */
    image_t resizeImg;

    /* Center-crop (match PC Resize(256)+CenterCrop(224)) instead of
     * stretching the whole frame into the square model input. */
    SetCenterCropRoi(roi, frameBuffer);

    resizeImg.w = inputImgCols;
    resizeImg.h = inputImgRows;
    resizeImg.data = (uint8_t *)inputTensor->data.data;
    resizeImg.pixfmt = PIXFORMAT_RGB888;

    imlib_nvt_scale(&frameBuffer, &resizeImg, &roi);

    /* Pre-processing (image → extractor input tensor) */
    if (!preProcess->DoPreProcess(resizeImg.data,
                                  (resizeImg.w * resizeImg.h * inputChannels)))
    {
        printf_err("[ISM] Pre-processing failed\r\n");
        extern bool frameBufferFrozen;
        frameBufferFrozen = false;
        return;
    }

    extern uint32_t SystemCoreClock;

    /* ---- Stage 1: Feature Extractor (NPU) ---- */
    uint64_t t0 = pmu_get_systick_Count();

    if (!extractorModel.RunInference())
    {
        printf_err("[ISM] Feature extractor inference failed\r\n");
        extern bool frameBufferFrozen;
        frameBufferFrozen = false;
        return;
    }

    uint64_t t_ext = pmu_get_systick_Count() - t0;

    /* ---- Copy intermediate features: extractor output → classifier input ---- */
    TfLiteTensor *extOutput = extractorModel.GetOutputTensor(0);
    TfLiteTensor *clsInput  = classifierModel.GetInputTensor(0);
    std::memcpy(clsInput->data.data, extOutput->data.data, extOutput->bytes);

    /* ---- Stage 2: Classifier (CPU) ---- */
    uint64_t t1 = pmu_get_systick_Count();

    if (!classifierModel.RunInference())
    {
        printf_err("[ISM] Classifier inference failed\r\n");
        extern bool frameBufferFrozen;
        frameBufferFrozen = false;
        return;
    }

    uint64_t t_cls   = pmu_get_systick_Count() - t1;
    uint64_t t_total = pmu_get_systick_Count() - t0;

    /* ---- Post-processing (classifier output → label) ---- */
    std::string predictLabelInfo;
    std::string top5LabelInfo;
    if (postProcess->DoPostProcess())
    {
        top5LabelInfo = BuildTopKResultString(results, 5);
        predictLabelInfo = BuildTopKResultString(results, 1);
    }
    else
    {
        top5LabelInfo = std::string("???");
        predictLabelInfo = std::string("???");
    }

    uint32_t ext_us   = (uint32_t)(t_ext   * 1000000ULL / SystemCoreClock);
    uint32_t cls_us   = (uint32_t)(t_cls   * 1000000ULL / SystemCoreClock);
    uint32_t total_us = (uint32_t)(t_total * 1000000ULL / SystemCoreClock);

    char totalTimingBuf[16];
    char splitTimingBuf[48];
    snprintf(totalTimingBuf, sizeof(totalTimingBuf), "%.2fms", total_us / 1000.0f);
    snprintf(splitTimingBuf, sizeof(splitTimingBuf), "ext:%.2f+cls:%.2f",
             ext_us / 1000.0f, cls_us / 1000.0f);

    info_critical("[FRAME: %u] [RESULT] %s (%s) (%s)\r\n", inferenceFrameCount, top5LabelInfo.c_str(), totalTimingBuf,
                  splitTimingBuf);
    lastInferenceResult = predictLabelInfo + std::string(" (") + totalTimingBuf + std::string(") (")
                      + splitTimingBuf + std::string(")");

    /* Unfreeze frameBuffer after inference completes */
    frameBufferFrozen = false;
}

/* ------------------------------------------------------------------ */
/*  ZO Training: Init and Train                                       */
/* ------------------------------------------------------------------ */

static void DoZOInit(void)
{
    /* Ensure split model is loaded first */
    if (!splitModelLoaded)
    {
        info("[ZO] Split model not loaded, loading now...\r\n");
        DoLoadSplitModel();
        if (!splitModelLoaded) return;
    }

    /* If we haven't yet re-initialised the classifier with preserve_all_tensors,
     * we need to re-init it. The first load uses the standard allocator path.
     * On zo_init we reload using preserveAllTensors=true. */
    const uint8_t *clsPtr      = arm::app::classifier_cpu_int8::GetModelPointer();
    size_t         clsLen      = arm::app::classifier_cpu_int8::GetModelLen();
    uint8_t       *clsArena    = SPLIT_CLS_ARENA_PTR;
    uint32_t       clsArenaSize = SPLIT_CLS_ARENA_SZ;

    info_if_token(LOG_ZO_TRAINING, "[ZO] Re-initialising classifier with preserve_all_tensors=true...\r\n");

    /* Reinitialise classifierModel to preserve all tensors */
    if (!classifierModel.Init(clsArena, clsArenaSize, clsPtr, clsLen,
                              nullptr, /* allocator */
                              true     /* preserveAllTensors */))
    {
        printf_err("[ZO] Failed to re-init classifier with preserve_all_tensors\r\n");
        return;
    }

    /* Update postProcess to use the freshly initialised classifier model */
    if (postProcess) {
        delete postProcess;
    }
    postProcess = new arm::app::ImgClassPostProcess(classifier, &classifierModel,
                                                    labels, results);

    /* Allocate ZOTrainer if not yet done */
    if (!zoTrainer) {
        zoTrainer = new ZOTrainer();
    }

    /* Calculate scratch buffer start: arena base + arena used + 1KB alignment guard */
    size_t arenaUsed    = classifierModel.GetArenaUsedBytes();
    size_t guardBytes   = 1024u;
    uint8_t* scratchBuf = clsArena + arenaUsed + guardBytes;
    size_t   scratchSz  = (size_t)clsArenaSize - arenaUsed - guardBytes;

    if (!zoTrainer->Init(classifierModel, scratchBuf, scratchSz))
    {
        printf_err("[ZO] ZOTrainer init failed\r\n");
        return;
    }

    if (!TryLoadZOTrainerSnapshotFromFlash())
    {
        info_critical("[ZO] Weight source: BUILTIN (no valid flash snapshot)\r\n");
    }
    else
    {
        info_critical("[ZO] Weight source: FLASH\r\n");
    }

    const ZOTrainer::FCInfo& fc = zoTrainer->GetFCInfo();
    info_critical("[ZO] Ready. FC[%d in × %d out], mem=%zu bytes\r\n",
         fc.input_features, fc.output_classes, zoTrainer->GetMemoryUsed());
    info_if_token(LOG_ZO_TRAINING, "[ZO] Commands: tra=<label>  zo_reset  zo_lr <val>  zo_q <val>\r\n");
}

static void DoZOTrain(void)
{
    if (!zoTrainer || !zoTrainer->IsInitialized())
    {
        printf_err("[ZO] Trainer not initialised. Send 'zo_init' first.\r\n");
        return;
    }

    if (zoTargetLabel < 0 || zoTargetLabel >= zoTrainer->GetFCInfo().output_classes)
    {
        printf_err("[ZO] Invalid target label index %d\r\n", zoTargetLabel);
        return;
    }

    /* Freeze frameBuffer to ensure consistent training frame */
    extern uint32_t inferenceFrameCount;
    extern bool frameBufferFrozen;
    inferenceFrameCount++;
    frameBufferFrozen = true;

    /* ---- Extract features from current camera frame ---- */
    if (!splitModelLoaded)
    {
        printf_err("[ZO] Split model not loaded\r\n");
        return;
    }

    /* Resize camera frame → extractor input */
    image_t resizeImg;
    /* Center-crop (match PC Resize(256)+CenterCrop(224)) instead of
     * stretching the whole frame into the square model input. */
    SetCenterCropRoi(roi, frameBuffer);

    resizeImg.w      = inputImgCols;
    resizeImg.h      = inputImgRows;
    resizeImg.data   = (uint8_t *)inputTensor->data.data;
    resizeImg.pixfmt = PIXFORMAT_RGB888;

    imlib_nvt_scale(&frameBuffer, &resizeImg, &roi);

    if (!preProcess->DoPreProcess(resizeImg.data,
                                  (resizeImg.w * resizeImg.h * inputChannels)))
    {
        printf_err("[ZO] Pre-processing failed\r\n");
        extern bool frameBufferFrozen;
        frameBufferFrozen = false;
        return;
    }

    /* Run feature extractor (NPU) */
    if (!extractorModel.RunInference())
    {
        printf_err("[ZO] Feature extractor failed\r\n");
        extern bool frameBufferFrozen;
        frameBufferFrozen = false;
        return;
    }

    /* Copy features → classifier input */
    TfLiteTensor *extOutput = extractorModel.GetOutputTensor(0);
    TfLiteTensor *clsInput  = classifierModel.GetInputTensor(0);
    std::memcpy(clsInput->data.data, extOutput->data.data, extOutput->bytes);

    /* Cache FC input vector — runs classifier once, no more NPU calls in TrainStep */
    if (!zoTrainer->CacheFeature(classifierModel))
    {
        printf_err("[ZO] CacheFeature failed\r\n");
        frameBufferFrozen = false;
        return;
    }

    /* ---- Run one ZO training step (CPU-only node perturbation) ---- */
    extern uint32_t SystemCoreClock;
    uint64_t t0 = pmu_get_systick_Count();

    float loss = (zoMethod == ZO_METHOD_WP)
                 ? zoTrainer->TrainStepWP(classifierModel, zoTargetLabel,
                                          zoLearningRate, zoNumPerturbations)
                 : zoTrainer->TrainStep(classifierModel, zoTargetLabel,
                                        zoLearningRate, zoNumPerturbations);

    uint64_t elapsed_cyc = pmu_get_systick_Count() - t0;
    uint32_t elapsed_us  = (uint32_t)(elapsed_cyc * 1000000ULL / SystemCoreClock);

    /* Device reports raw per-step measurements only; the external host decides
     * convergence/stopping from this log series. No on-device judgement here. */
    const ZOTrainer::StepMetrics& m = zoTrainer->GetLastMetrics();
    (void)loss; /* full A→B series is in the metrics struct */
    const std::string& labelName = labels[(size_t)zoTargetLabel];
    const char* methodName = (zoMethod == ZO_METHOD_WP) ? "WP" : "NP";
    info_critical("[FRAME: %u] [ZO] Step %d | method=%s | target=%s(%d) | loss=%.4f->%.4f | loss_ema=%.4f | "
                  "delta_params=%.2f%% | grad_norm=%.4f | lr=%.6f | Q=%d | Time: %.2fms | Mem: %zu bytes\r\n",
         inferenceFrameCount,
         zoTrainer->GetStepCount(),
         methodName,
         labelName.c_str(), zoTargetLabel,
         m.loss_before, m.loss_after,
         m.loss_ema,
         m.delta_params * 100.0f,
         m.grad_norm,
         zoLearningRate, zoNumPerturbations,
         elapsed_us / 1000.0f,
         zoTrainer->GetMemoryUsed());

    /* Sub-LSB diagnostics: if dw_max < 0.5 the whole step is below the INT8 grid
     * so no weight can move — that is "lr too small", not a broken write-back.
     * a_rms/lr_eff let you back out the lr needed to cross 0.5. */
    info_critical("[FRAME: %u] [ZO]   dw_max=%.4f (need>=0.5) | dw_mean=%.4f | db_max=%.4f | "
                  "lr_eff=%.6e | a_rms=%.4f | %s\r\n",
         inferenceFrameCount,
         m.dw_max, m.dw_mean, m.db_max,
         m.lr_eff, m.a_rms,
         (m.dw_max < 0.5f) ? "[SUB-LSB: weights frozen, raise lr]" : "[weights moving]");

    info_if_token(LOG_ZO_TRAINING, "[ZO] Snapshot not auto-saved. Send 'zo_save' to persist to APROM flash.\r\n");

    /* Unfreeze frameBuffer after training completes */
    frameBufferFrozen = false;
}

int ISM_ZO_SaveToFlash(void)
{
    return SaveZOTrainerSnapshotToFlash() ? 1 : 0;
}

int ISM_ZO_ClearFlash(void)
{
    if (!FlashErasePages(kZoFlashRegionAddr, kZoFlashRegionBytes))
    {
        printf_err("[ZO] Failed to erase persisted snapshot from APROM flash\r\n");
        return 0;
    }

    g_zoLoadedFromFlash = false;
    g_zoPersistedTrainSteps = 0;
    info_critical("[ZO] Persisted snapshot erased from flash @0x%08X\r\n", kZoFlashRegionAddr);
    return 1;
}

void ISM_ZO_PrintStatus(void)
{
    if (!zoTrainer || !zoTrainer->IsInitialized())
    {
        info("[ZO] Status: trainer=NOT_INITIALISED\r\n");
        return;
    }

    const char *source = g_zoLoadedFromFlash ? "FLASH" : "BUILTIN";
    info_critical("[ZO] Status: trainer=READY, source=%s, runtime_steps=%d, flash_steps=%u\r\n",
                  source, zoTrainer->GetStepCount(), g_zoPersistedTrainSteps);
}

#endif /* USE_SPLIT_MODEL */

/* ------------------------------------------------------------------ */
/*  DoGraphDump - Traverse TFLite model graph structure               */
/* ------------------------------------------------------------------ */
#if !(defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1))
static void DoGraphDump(void)
{
    /* Graph dump reads FlatBuffer directly — does NOT require model.Init() */

    /* Print basic model info manually */
    const uint8_t *modelPtr = arm::app::SINGLE_MODEL_NS::GetModelPointer();
    printf("Model Address: %p\r\n", modelPtr);
    printf("Model Size:    %zu bytes\r\n", arm::app::SINGLE_MODEL_NS::GetModelLen());

    /* Now do detailed FlatBuffer traversal */
    const tflite::Model *fbModel = ::tflite::GetModel(modelPtr);
    if (!fbModel || !fbModel->subgraphs() || fbModel->subgraphs()->size() == 0)
    {
        printf("[ISM] Invalid model FlatBuffer\r\n");
        return;
    }

    const tflite::SubGraph *subgraph = fbModel->subgraphs()->Get(0);
    const auto *tensors   = subgraph->tensors();
    const auto *operators  = subgraph->operators();
    const auto *opcodes    = fbModel->operator_codes();

    /* ---- Tensor Table ---- */
    printf("\r\n========== TENSOR TABLE (%d tensors) ==========\r\n",
         tensors ? (int)tensors->size() : 0);

    if (tensors)
    {
        for (uint32_t t = 0; t < tensors->size(); t++)
        {
            const tflite::Tensor *tensor = tensors->Get(t);
            const char *name = tensor->name() ? tensor->name()->c_str() : "(unnamed)";
            const char *typeName = tflite::EnumNameTensorType(tensor->type());

            printf("  T[%3" PRIu32 "] %-10s ", t, typeName);

            /* Print shape */
            if (tensor->shape() && tensor->shape()->size() > 0)
            {
                printf("shape=[");
                for (uint32_t s = 0; s < tensor->shape()->size(); s++)
                {
                    if (s > 0) printf(",");
                    printf("%d", tensor->shape()->Get(s));
                }
                printf("] ");
            }
            printf("%s\r\n", name);
        }
    }

    /* ---- Operator Graph ---- */
    printf("\r\n========== OPERATOR GRAPH (%d ops) ==========\r\n",
         operators ? (int)operators->size() : 0);

    if (operators && opcodes)
    {
        for (uint32_t i = 0; i < operators->size(); i++)
        {
            const tflite::Operator *op = operators->Get(i);
            const tflite::OperatorCode *opcode = opcodes->Get(op->opcode_index());

            /* Resolve operator name */
            auto builtinCode = tflite::GetBuiltinCode(opcode);
            const char *opName;
            if (builtinCode == tflite::BuiltinOperator_CUSTOM)
            {
                opName = opcode->custom_code() ? opcode->custom_code()->c_str() : "CUSTOM";
            }
            else
            {
                opName = tflite::EnumNameBuiltinOperator(builtinCode);
            }

            printf("  OP[%3" PRIu32 "] %s\r\n", i, opName);

            /* Print input tensor indices */
            if (op->inputs() && op->inputs()->size() > 0)
            {
                printf("         inputs : [");
                for (uint32_t j = 0; j < op->inputs()->size(); j++)
                {
                    if (j > 0) printf(", ");
                    printf("%d", op->inputs()->Get(j));
                }
                printf("]\r\n");
            }

            /* Print output tensor indices */
            if (op->outputs() && op->outputs()->size() > 0)
            {
                printf("         outputs: [");
                for (uint32_t j = 0; j < op->outputs()->size(); j++)
                {
                    if (j > 0) printf(", ");
                    printf("%d", op->outputs()->Get(j));
                }
                printf("]\r\n");
            }
        }
    }

    /* ---- Subgraph I/O ---- */
    printf("\r\n========== SUBGRAPH I/O ==========\r\n");
    if (subgraph->inputs() && subgraph->inputs()->size() > 0)
    {
        printf("  Graph Inputs : [");
        for (uint32_t i = 0; i < subgraph->inputs()->size(); i++)
        {
            if (i > 0) printf(", ");
            printf("%d", subgraph->inputs()->Get(i));
        }
        printf("]\r\n");
    }
    if (subgraph->outputs() && subgraph->outputs()->size() > 0)
    {
        printf("  Graph Outputs: [");
        for (uint32_t i = 0; i < subgraph->outputs()->size(); i++)
        {
            if (i > 0) printf(", ");
            printf("%d", subgraph->outputs()->Get(i));
        }
        printf("]\r\n");
    }
    printf("========== END GRAPH DUMP ==========\r\n");
}

#endif /* !USE_SPLIT_MODEL */

/* ------------------------------------------------------------------ */
/*  API Implementation                                                */
/* ------------------------------------------------------------------ */

void ISM_Init(void)
{
    g_currentState = ISM_IDLE;
    g_requestState = ISM_IDLE;
#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
    g_zoLoadedFromFlash = false;
    g_zoPersistedTrainSteps = 0;
#endif
}

void ISM_SetState(ISM_State_t newState)
{
    g_requestState = newState;
}

ISM_State_t ISM_GetState(void)
{
    return g_currentState;
}

bool ISM_RequestCapturesFrame(void)
{
#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
    /* Both classify (this_is?) and one-step ZO training (tra=<label>) grab a
     * fresh camera frame, so both want the LCD blanked first. */
    return (g_requestState == ISM_SPLIT_INFERENCE) ||
           (g_requestState == ISM_ZO_TRAIN);
#else
    return (g_requestState == ISM_INFERENCE);
#endif
}

void ISM_Process(void)
{
    /* Check if there is a new request */
    if (g_requestState != ISM_IDLE)
    {
        g_currentState = g_requestState;
        g_requestState = ISM_IDLE; /* Clear request immediately */

        switch (g_currentState)
        {
#if !(defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1))
            case ISM_LOAD_MODEL:
                DoLoadModel();
                g_currentState = ISM_IDLE;
                break;

            case ISM_INFERENCE:
                DoInference();
                g_currentState = ISM_IDLE;
                break;

            case ISM_DUMP_GRAPH:
                DoGraphDump();
                g_currentState = ISM_IDLE;
                break;
#endif /* !USE_SPLIT_MODEL */

#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
            case ISM_LOAD_SPLIT_MODEL:
                DoLoadSplitModel();
                g_currentState = ISM_IDLE;
                break;

            case ISM_SPLIT_INFERENCE:
                DoSplitInference();
                g_currentState = ISM_IDLE;
                break;

            case ISM_ZO_INIT:
                DoZOInit();
                g_currentState = ISM_IDLE;
                break;

            case ISM_ZO_TRAIN:
                DoZOTrain();
                g_currentState = ISM_IDLE;
                break;
#endif

            case ISM_IDLE:
            default:
                break;
        }
    }
}