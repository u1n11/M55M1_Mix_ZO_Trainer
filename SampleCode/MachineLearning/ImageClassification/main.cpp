/**************************************************************************//**
 * @file     main.cpp
 * @version  V2.10
 * @brief    MobileNetV2 network sample. UART command-driven image classification.
 *           - Uses GlobalState for shared variables.
 *           - Uses uart/uart_cmd.cpp for command handling.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "BoardInit.hpp"      /* Board initialisation */
#include "log_macros.h"      /* Logging macros (optional) */

#include "BufAttributes.hpp" /* Buffer attributes to be applied */
#include "Classifier.hpp"    /* Classifier for the result */
#include "ClassificationResult.hpp"
#include "MobileNetModel.hpp"       /* Model API */
#include "Labels.hpp"
#include "ImgClassProcessing.hpp"

#include "imlib.h"          /* Image processing */
#include "framebuffer.h"

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

#define __USE_CCAP__        /* Camera (CCAP) is mandatory */
#include "ImageSensor.h"

//#define __PROFILE__
#define __USE_DISPLAY__
//#define __USE_UVC__

#include "Profiler.hpp"

#if defined (__USE_DISPLAY__)
    #include "Display.h"
#endif

#if defined (__USE_UVC__)
    #include "UVC.h"
#endif

/* Include Global State and UART Command Header */
#include "GlobalState.hpp"
#include "uart_cmd.h"
#include "inference_mngt.h"

#define IMAGE_DISP_UPSCALE_FACTOR 1
#if defined(LT7381_LCD_PANEL)
    #define FONT_DISP_UPSCALE_FACTOR 2
#else
    #define FONT_DISP_UPSCALE_FACTOR 1
#endif

using ImageClassifier = arm::app::Classifier;

namespace arm
{
namespace app
{
/* Tensor arena buffer for NPU model (in SRAM) */
uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;

/* Tensor arena buffer for CPU model (in SRAM, split-model classifier path) */
#ifndef CPU_ACTIVATION_BUF_SZ
    #define CPU_ACTIVATION_BUF_SZ  0x00020000  /* 128 KB */
#endif
__attribute__((aligned(16), section(".bss.NoInit.activation_buf_sram")))
uint8_t cpuTensorArena[CPU_ACTIVATION_BUF_SZ];

/* Optional getter function for the model pointer and its size. */
namespace mobilenet
{
extern uint8_t *GetModelPointer();
extern size_t GetModelLen();
} /* namespace mobilenet */
} /* namespace app */
} /* namespace arm */

/* Image processing initiate function */
//Used by omv library
#if defined(__USE_UVC__)
    //UVC only support QVGA, QQVGA
    #define GLCD_WIDTH  320
    #define GLCD_HEIGHT 240
#else
    #define GLCD_WIDTH 320
    #define GLCD_HEIGHT 240
#endif

//RGB565
#define IMAGE_FB_SIZE   (GLCD_WIDTH * GLCD_HEIGHT * 2)

#undef OMV_FB_SIZE
#define OMV_FB_SIZE (IMAGE_FB_SIZE + 1024)

/* fb_array - NO LONGER STATIC (needed by uart_cmd.cpp MPU config) */
__attribute__((section(".bss.vram.data"), aligned(32))) char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".bss.vram.data"), aligned(32))) static char jpeg_array[OMV_JPEG_BUF_SIZE];

char *_fb_base = NULL;
char *_fb_end = NULL;
char *_jpeg_buf = NULL;
char *_fballoc = NULL;

static void omv_init()
{
    // frameBuffer is now global
    frameBuffer.w = GLCD_WIDTH;
    frameBuffer.h = GLCD_HEIGHT;
    frameBuffer.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    frameBuffer.pixfmt = PIXFORMAT_RGB565;

    _fb_base = fb_array;
    _fb_end =  fb_array + OMV_FB_SIZE - 1;
    _fballoc = _fb_base + OMV_FB_SIZE + OMV_FB_ALLOC_SIZE;
    _jpeg_buf = jpeg_array;

    fb_alloc_init0();

    framebuffer_init0();
    framebuffer_init_from_image(&frameBuffer);
}

/* ------------------------------------------------------------------ */
/*  Global Variable Definitions (Declared extern in GlobalState.hpp)  */
/* ------------------------------------------------------------------ */
arm::app::MobileNetModel model;
arm::app::Classifier classifier;
bool modelLoaded = false;

#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
arm::app::MobileNetModel extractorModel;
arm::app::MobileNetModel classifierModel;
bool splitModelLoaded = false;
#endif

TfLiteTensor *inputTensor = nullptr;
int inputImgCols = 0;
int inputImgRows = 0;
uint32_t inputChannels = 0;

std::vector<std::string> labels;
std::vector<arm::app::ClassificationResult> results;

arm::app::ImgClassPreProcess  *preProcess  = nullptr;
arm::app::ImgClassPostProcess *postProcess = nullptr;

image_t frameBuffer;
rectangle_t roi;

/* Frame synchronisation */
uint32_t inferenceFrameCount = 0;  /* Increments on each inference start */
bool frameBufferFrozen = false;    /* When true, skip camera capture updates */

/* Result string for display persistence */
std::string lastInferenceResult = "";

#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
ZOTrainer* zoTrainer          = nullptr;
int        zoTargetLabel      = -1;
float      zoLearningRate     = 0.01f;
int        zoNumPerturbations = 20;
#endif

/* ------------------------------------------------------------------ */
/*  UART command receiver (non-blocking, line-based)                  */
/* ------------------------------------------------------------------ */
#define UART_CMD_BUF_SIZE 256

static bool UART_GetCommand(char *buf, uint32_t bufSize)
{
    static uint32_t idx = 0;

    while (UART_IS_RX_READY(DEBUG_PORT))
    {
        char c = (char)UART_READ(DEBUG_PORT);

        if (c == '\r' || c == '\n')
        {
            if (idx > 0)
            {
                buf[idx] = '\0';
                idx = 0;
                return true;
            }
            /* ignore empty lines */
        }
        else if (idx < bufSize - 1)
        {
            buf[idx++] = c;
        }
    }

    return false;  /* no complete command yet */
}

volatile bool g_bBtn0Pressed = false;

extern "C" void GPI_IRQHandler(void)
{
    /* To check if PI.11 external interrupt occurred */
    if (GPIO_GET_INT_FLAG(PI, BIT11))
    {
        GPIO_CLR_INT_FLAG(PI, BIT11);
        g_bBtn0Pressed = true;
    }
}

/* ================================================================== */
/*  main                                                              */
/* ================================================================== */
int main()
{

    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();

    /* omv library init */
    omv_init();
    framebuffer_init_image(&frameBuffer);

#if defined(__PROFILE__)
    // Profiler is now local or handled in cmd
    pmu_reset_counters(); // Reset needed if generic profiler not used globally here
#else
    pmu_reset_counters();
#endif

#if defined (__USE_CCAP__)
    /* Setup image sensor */
    int iCameraRet = ImageSensor_Init();
    if (iCameraRet != 0)
    {
        printf_err("Failed to initialize camera (error code: %d)\n", iCameraRet);
        return 10;
    }
    iCameraRet = ImageSensor_Config(eIMAGE_FMT_RGB565, frameBuffer.w, frameBuffer.h, true);
    if (iCameraRet != 0)
    {
        printf_err("Failed to configure camera (error code: %d)\n", iCameraRet);
        return 11;
    }
#endif

#if defined (__USE_DISPLAY__)
    /* Display init */
    Display_Init();
    Display_ClearLCD(C_WHITE);
#endif

#if defined (__USE_UVC__)
    UVC_Init();
    HSUSBD_Start();
#endif

    /* UART command buffer */
    char cmdBuf[UART_CMD_BUF_SIZE];

    /* Enable LIRC for GPIO debounce */
    SYS_UnlockReg();
    CLK_EnableXtalRC(CLK_SRCCTL_LIRCEN_Msk);
    CLK_WaitClockReady(CLK_STATUS_LIRCSTB_Msk);
    SYS_LockReg();

    /* Configure PI.11 as input pin and enable interrupt by falling edge trigger */
    GPIO_SetMode(PI, BIT11, GPIO_MODE_INPUT);
    GPIO_EnableInt(PI, 11, GPIO_INT_FALLING);
    NVIC_EnableIRQ(GPI_IRQn);
    /* Enable interrupt de-bounce function and select de-bounce sampling cycle time is 1024 clocks of LIRC clock */
    GPIO_SET_DEBOUNCE_TIME(PI, GPIO_DBCTL_DBCLKSRC_LIRC, GPIO_DBCTL_DBCLKSEL_1024);
    GPIO_ENABLE_DEBOUNCE(PI, BIT11);

    info("System ready. \n");
#if !(defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1))
    info("  load_model  - Load the monolithic model\n");
    info("  this_is?    - Classify (monolithic model)\n");
    info("  show_graph  - Dump model graph structure\n");
#endif /* !USE_SPLIT_MODEL */
#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
    info("  load_model  - Load split model (extractor+classifier)\n");
    info("  this_is?    - Classify (split model)\n");
    info("  zo_init     - Init ZO trainer (preserves all tensors)\n");
    info("  tra=<label> - Train one step (e.g. tra=cat)\n");
    info("  zo_reset    - Restore original weights\n");
    info("  zo_status   - Show trainer source/step status\n");
    info("  zo_save     - Force save current trainer snapshot to flash\n");
    info("  zo_set_lr <val>  - Set learning rate (default 0.01)\n");
    info("  zo_set_q <val>   - Set perturbation count Q (default 20)\n");
#endif /* USE_SPLIT_MODEL */

    /* ============================================================== */
    /*  Main loop: Camera capture + Display + UART command handling    */
    /* ============================================================== */
    while (1)
    {
        /* ---------------------------------------------------------- */
        /*  1. Display current camera frame on LCD                    */
        /* ---------------------------------------------------------- */
#if defined (__USE_DISPLAY__)
        {
            S_DISP_RECT sDispRect;

            sDispRect.u32TopLeftX = 0;
            sDispRect.u32TopLeftY = 0;
            sDispRect.u32BottonRightX = ((frameBuffer.w * IMAGE_DISP_UPSCALE_FACTOR) - 1);
            sDispRect.u32BottonRightY = ((frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) - 1);

            Display_FillRect((uint16_t *)frameBuffer.data, &sDispRect, IMAGE_DISP_UPSCALE_FACTOR);

            /* Display Persistent Result String */
            if (!lastInferenceResult.empty())
            {
                char szDisplayText[160];
                sprintf(szDisplayText, "%s", lastInferenceResult.c_str());

                Display_PutText(
                    szDisplayText,
                    strlen(szDisplayText),
                    0,
                    frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR,
                    C_BLUE,
                    C_WHITE,
                    false,
                    FONT_DISP_UPSCALE_FACTOR
                );
            }
        }
#endif

        /* ---------------------------------------------------------- */
        /*  2. UVC streaming (if enabled)                             */
        /* ---------------------------------------------------------- */
#if defined (__USE_UVC__)
        if (UVC_IsConnect())
        {
            /* implementation omitted for brevity, logic remains same if UVC used */
             UVC_SendImage((uint32_t)frameBuffer.data, IMAGE_FB_SIZE, uvcStatus.StillImage);
        }
#endif

        /* ---------------------------------------------------------- */
        /*  3. Trigger next camera capture (skip if inference frozen) */
        /* ---------------------------------------------------------- */
#if defined (__USE_CCAP__)
        if (!frameBufferFrozen)
        {
            ImageSensor_TriggerCapture((uint32_t)frameBuffer.data);
        }
#endif

        /* ---------------------------------------------------------- */
        /*  4. Check for UART commands (non-blocking)                 */
        /* ---------------------------------------------------------- */
        if (UART_GetCommand(cmdBuf, sizeof(cmdBuf)))
        {
            /* Use the new command processor from uart_cmd */
            UART_ProcessCommand(cmdBuf);
        }

        /* Check for button press */
        if (g_bBtn0Pressed)
        {
            g_bBtn0Pressed = false;
            UART_ProcessCommand((char*)"this_is?");
        }

        /* ---------------------------------------------------------- */
        /*  5. Process Inference State Machine (ISM)                  */
        /* ---------------------------------------------------------- */
        ISM_Process();

        /* ---------------------------------------------------------- */
        /*  6. Wait for camera capture to complete (skip if frozen)   */
        /* ---------------------------------------------------------- */
#if defined (__USE_CCAP__)
        if (!frameBufferFrozen)
        {
            ImageSensor_WaitCaptureDone();
        }
#endif

    } /* end while(1) */
}
