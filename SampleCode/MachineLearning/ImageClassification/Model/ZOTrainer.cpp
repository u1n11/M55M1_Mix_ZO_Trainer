#include "ZOTrainer.hpp"
#include "log_macros.h"

#include <cstring>
#include <cmath>
#include <cstdlib>

/* TFLite Micro headers needed for FlatBuffer traversal and tensor access */
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/c/common.h"

/* ------------------------------------------------------------------ */
/*  Constructor                                                        */
/* ------------------------------------------------------------------ */
ZOTrainer::ZOTrainer()
    : m_xorState(12345u),
      m_mutableWeights(nullptr),
      m_mutableBias(nullptr),
      m_originalWeights(nullptr),
      m_originalBias(nullptr),
      m_stepSnapshot(nullptr),
      m_featureCache(nullptr),
      m_nodeGradBuf(nullptr),
      m_weightGradBuf(nullptr),
      m_fcInfo{},
      m_stepCount(0),
      m_lastLoss(0.0f),
      m_metrics{},
      m_lossEma(0.0f),
      m_emaInited(false),
      m_inited(false),
      m_memUsed(0)
{
}

/* ------------------------------------------------------------------ */
/*  XORShift PRNG (paper Appendix C)                                  */
/* ------------------------------------------------------------------ */
void ZOTrainer::SetSeed(uint32_t seed)
{
    m_xorState = (seed == 0) ? 1u : seed; /* zero seed would lock the PRNG */
}

uint32_t ZOTrainer::XorRand()
{
    m_xorState ^= m_xorState << 13;
    m_xorState ^= m_xorState >> 17;
    m_xorState ^= m_xorState << 5;
    return m_xorState;
}

int ZOTrainer::Rademacher()
{
    return (XorRand() & 1u) ? -1 : 1;
}

/* ------------------------------------------------------------------ */
/*  Init — locate last FC layer, copy weights to scratch buffer       */
/* ------------------------------------------------------------------ */
bool ZOTrainer::Init(arm::app::Model& classifierModel, uint8_t* scratchBuf, size_t scratchSize)
{
    if (m_inited) {
        info("[ZO] Already initialized\r\n");
        return true;
    }

    tflite::MicroInterpreter* interp  = classifierModel.GetInterpreter();
    const tflite::Model*      fbModel = classifierModel.GetModelFlatBuffer();

    if (!interp || !fbModel) {
        printf_err("[ZO] Interpreter or FlatBuffer model not available\r\n");
        return false;
    }

    if (!interp->preserve_all_tensors()) {
        printf_err("[ZO] Interpreter was not created with preserve_all_tensors=true. "
                   "Re-init the classifier model with preserveAllTensors=true.\r\n");
        return false;
    }

    /* ---- Locate the LAST FULLY_CONNECTED op in the FlatBuffer ---- */
    if (!fbModel->subgraphs() || fbModel->subgraphs()->size() == 0) {
        printf_err("[ZO] No subgraphs in classifier FlatBuffer\r\n");
        return false;
    }

    const tflite::SubGraph* sg        = fbModel->subgraphs()->Get(0);
    const auto*             opcodes   = fbModel->operator_codes();
    const auto*             operators = sg->operators();
    const auto*             tensors   = sg->tensors();

    if (!sg || !opcodes || !operators || !tensors) {
        printf_err("[ZO] FlatBuffer subgraph fields are null\r\n");
        return false;
    }

    /* Bug 4 fix: iterate all ops without break so fcOpIdx = last FC */
    int fcOpIdx = -1;
    for (uint32_t i = 0; i < operators->size(); i++) {
        const tflite::Operator*     op     = operators->Get(i);
        const tflite::OperatorCode* opcode = opcodes->Get(op->opcode_index());
        if (tflite::GetBuiltinCode(opcode) == tflite::BuiltinOperator_FULLY_CONNECTED) {
            fcOpIdx = (int)i; /* keep scanning — classification head is the last FC */
        }
    }

    if (fcOpIdx < 0) {
        printf_err("[ZO] No FULLY_CONNECTED op found in classifier model\r\n");
        return false;
    }

    const tflite::Operator* fcOp = operators->Get((uint32_t)fcOpIdx);

    /* inputs[0]=feature a, inputs[1]=weight, inputs[2]=bias */
    if (!fcOp->inputs() || fcOp->inputs()->size() < 3) {
        printf_err("[ZO] FC op does not have expected 3 inputs\r\n");
        return false;
    }

    m_fcInfo.feature_tensor_idx = fcOp->inputs()->Get(0); /* NEW: FC input a */
    m_fcInfo.weight_tensor_idx  = fcOp->inputs()->Get(1);
    m_fcInfo.bias_tensor_idx    = fcOp->inputs()->Get(2);

    /* ---- Read weight tensor metadata ---- */
    const tflite::Tensor* wTensor = tensors->Get((uint32_t)m_fcInfo.weight_tensor_idx);
    if (!wTensor || wTensor->type() != tflite::TensorType_INT8) {
        printf_err("[ZO] FC weight tensor is not INT8 (type=%d)\r\n",
                   wTensor ? (int)wTensor->type() : -1);
        return false;
    }

    if (!wTensor->shape() || wTensor->shape()->size() != 2) {
        printf_err("[ZO] Unexpected FC weight shape dims=%d\r\n",
                   wTensor->shape() ? (int)wTensor->shape()->size() : -1);
        return false;
    }

    m_fcInfo.output_classes = wTensor->shape()->Get(0);
    m_fcInfo.input_features = wTensor->shape()->Get(1);
    m_fcInfo.weight_bytes   = m_fcInfo.output_classes * m_fcInfo.input_features;
    m_fcInfo.bias_bytes     = m_fcInfo.output_classes * (int)sizeof(int32_t);

    /* Bug 3 fix: count available per-channel scales */
    m_fcInfo.weight_scale         = 1.0f;
    m_fcInfo.weight_zero_point    = 0;
    m_fcInfo.weight_scales_per_ch = nullptr; /* populated after scratch partitioning */
    int nScales = 0;
    if (wTensor->quantization() && wTensor->quantization()->scale()) {
        nScales = (int)wTensor->quantization()->scale()->size();
    }
    if (nScales >= 1) {
        m_fcInfo.weight_scale = wTensor->quantization()->scale()->Get(0);
    }
    if (wTensor->quantization() && wTensor->quantization()->zero_point() &&
        wTensor->quantization()->zero_point()->size() > 0)
    {
        m_fcInfo.weight_zero_point = (int)wTensor->quantization()->zero_point()->Get(0);
    }

    /* ---- Read feature tensor quantization ---- */
    m_fcInfo.feature_scale      = 1.0f;
    m_fcInfo.feature_zero_point = 0;
    {
        const tflite::Tensor* fTensor = tensors->Get((uint32_t)m_fcInfo.feature_tensor_idx);
        if (fTensor && fTensor->quantization() && fTensor->quantization()->scale() &&
            fTensor->quantization()->scale()->size() > 0)
        {
            m_fcInfo.feature_scale = fTensor->quantization()->scale()->Get(0);
            if (fTensor->quantization()->zero_point() &&
                fTensor->quantization()->zero_point()->size() > 0)
            {
                m_fcInfo.feature_zero_point =
                    (int)fTensor->quantization()->zero_point()->Get(0);
            }
        }
    }

    /* ---- Read output tensor quantization ---- */
    m_fcInfo.output_scale      = 1.0f;
    m_fcInfo.output_zero_point = 0;
    if (sg->outputs() && sg->outputs()->size() > 0) {
        int outIdx = sg->outputs()->Get(0);
        const tflite::Tensor* outT = tensors->Get((uint32_t)outIdx);
        if (outT && outT->quantization() && outT->quantization()->scale() &&
            outT->quantization()->scale()->size() > 0)
        {
            m_fcInfo.output_scale = outT->quantization()->scale()->Get(0);
            if (outT->quantization()->zero_point() &&
                outT->quantization()->zero_point()->size() > 0)
            {
                m_fcInfo.output_zero_point =
                    (int)outT->quantization()->zero_point()->Get(0);
            }
        }
    }

    /* ---- Calculate required scratch memory ---- */
    size_t needed = (size_t)m_fcInfo.weight_bytes                           /* mutableWeights  */
                  + (size_t)m_fcInfo.bias_bytes                             /* mutableBias     */
                  + (size_t)m_fcInfo.weight_bytes                           /* originalWeights */
                  + (size_t)m_fcInfo.bias_bytes                             /* originalBias    */
                  + (size_t)m_fcInfo.weight_bytes                           /* stepSnapshot    */
                  + (size_t)m_fcInfo.input_features                         /* featureCache    */
                  + (size_t)m_fcInfo.output_classes * sizeof(float)         /* nodeGradBuf     */
                  + (size_t)m_fcInfo.weight_bytes   * sizeof(float);        /* weightGradBuf   */
    if (nScales > 1) {
        needed += (size_t)nScales * sizeof(float);                          /* per-ch scales   */
    }

    if (scratchSize < needed) {
        printf_err("[ZO] Scratch buffer too small: need %zu, got %zu bytes\r\n",
                   needed, scratchSize);
        return false;
    }

    /* ---- Partition scratch buffer ---- */
    uint8_t* ptr = scratchBuf;

    m_mutableWeights  = reinterpret_cast<int8_t*>(ptr);
    ptr += (size_t)m_fcInfo.weight_bytes;

    m_mutableBias     = reinterpret_cast<int32_t*>(ptr);
    ptr += (size_t)m_fcInfo.bias_bytes;

    m_originalWeights = reinterpret_cast<int8_t*>(ptr);
    ptr += (size_t)m_fcInfo.weight_bytes;

    m_originalBias    = reinterpret_cast<int32_t*>(ptr);
    ptr += (size_t)m_fcInfo.bias_bytes;

    m_stepSnapshot    = reinterpret_cast<int8_t*>(ptr);
    ptr += (size_t)m_fcInfo.weight_bytes;

    m_featureCache    = reinterpret_cast<int8_t*>(ptr);
    ptr += (size_t)m_fcInfo.input_features;

    m_nodeGradBuf     = reinterpret_cast<float*>(ptr);
    ptr += (size_t)m_fcInfo.output_classes * sizeof(float);

    m_weightGradBuf   = reinterpret_cast<float*>(ptr);
    ptr += (size_t)m_fcInfo.weight_bytes * sizeof(float);

    /* Bug 3 fix: store per-channel weight scales from scratch */
    if (nScales > 1) {
        m_fcInfo.weight_scales_per_ch = reinterpret_cast<float*>(ptr);
        ptr += (size_t)nScales * sizeof(float);
        for (int c = 0; c < nScales; c++) {
            m_fcInfo.weight_scales_per_ch[c] = wTensor->quantization()->scale()->Get(c);
        }
    }

    m_memUsed = needed;

    /* ---- Copy Flash weights to mutable buffers ---- */
    TfLiteEvalTensor* evalW = interp->GetTensor(m_fcInfo.weight_tensor_idx);
    TfLiteEvalTensor* evalB = interp->GetTensor(m_fcInfo.bias_tensor_idx);

    if (!evalW || !evalW->data.data) {
        printf_err("[ZO] Could not get FC weight eval tensor\r\n");
        return false;
    }

    std::memcpy(m_mutableWeights,  evalW->data.data, (size_t)m_fcInfo.weight_bytes);
    std::memcpy(m_originalWeights, evalW->data.data, (size_t)m_fcInfo.weight_bytes);

    if (evalB && evalB->data.data && m_fcInfo.bias_bytes > 0) {
        std::memcpy(m_mutableBias,  evalB->data.data, (size_t)m_fcInfo.bias_bytes);
        std::memcpy(m_originalBias, evalB->data.data, (size_t)m_fcInfo.bias_bytes);
    }

    /* ---- Redirect interpreter tensor data pointers to mutable buffers ---- */
    evalW->data.data = m_mutableWeights;
    if (evalB && evalB->data.data && m_fcInfo.bias_bytes > 0) {
        evalB->data.data = m_mutableBias;
    }

    m_inited    = true;
    m_stepCount = 0;
    m_lastLoss  = 0.0f;

    info("[ZO] Init OK: FC[%d×%d] w_scale=%.6f f_scale=%.6f per_ch=%s mem=%zu bytes\r\n",
         m_fcInfo.input_features, m_fcInfo.output_classes,
         m_fcInfo.weight_scale, m_fcInfo.feature_scale,
         (m_fcInfo.weight_scales_per_ch ? "yes" : "no"),
         m_memUsed);

    return true;
}

/* ------------------------------------------------------------------ */
/*  CacheFeature — run one classifier inference, store FC input a     */
/* ------------------------------------------------------------------ */
bool ZOTrainer::CacheFeature(arm::app::Model& model)
{
    if (!m_inited) {
        printf_err("[ZO] CacheFeature called before Init\r\n");
        return false;
    }

    model.RunInference(); /* one NPU+CPU pass per sample */

    TfLiteEvalTensor* evalA =
        model.GetInterpreter()->GetTensor(m_fcInfo.feature_tensor_idx);
    if (!evalA || !evalA->data.data) {
        printf_err("[ZO] Cannot get feature tensor (idx=%d)\r\n",
                   m_fcInfo.feature_tensor_idx);
        return false;
    }

    std::memcpy(m_featureCache, evalA->data.data, (size_t)m_fcInfo.input_features);
    return true;
}

/* ------------------------------------------------------------------ */
/*  ComputeLossFromCachedFeature — CPU-only dequant FC + softmax-CE   */
/* ------------------------------------------------------------------ */
float ZOTrainer::ComputeLossFromCachedFeature(const float* nodeOffset, int targetLabel)
{
    const int   C   = m_fcInfo.output_classes;
    const int   F   = m_fcInfo.input_features;
    const float fS  = m_fcInfo.feature_scale;
    const int   fZP = m_fcInfo.feature_zero_point;

    float z[10]; /* C=10, safe on stack */
    for (int c = 0; c < C; c++) {
        float ws_c = (m_fcInfo.weight_scales_per_ch)
                     ? m_fcInfo.weight_scales_per_ch[c]
                     : m_fcInfo.weight_scale;
        float acc = 0.0f;
        for (int f = 0; f < F; f++) {
            float w_f = ws_c * (float)(m_mutableWeights[c * F + f]
                                       - m_fcInfo.weight_zero_point);
            float a_f = fS   * (float)(m_featureCache[f] - fZP);
            acc += w_f * a_f;
        }
        /* bias_scale[c] = weight_scale[c] * feature_scale (TFLite INT8 FC conv.) */
        acc += (ws_c * fS) * (float)m_mutableBias[c];
        z[c] = acc;
    }

    if (nodeOffset) {
        for (int c = 0; c < C; c++) z[c] += nodeOffset[c];
    }

    /* Numerically stable softmax-CE */
    float maxV = z[0];
    for (int c = 1; c < C; c++) if (z[c] > maxV) maxV = z[c];
    float sumExp = 0.0f, targetExp = 0.0f;
    for (int c = 0; c < C; c++) {
        float e = expf(z[c] - maxV);
        sumExp += e;
        if (c == targetLabel) targetExp = e;
    }
    return -logf(targetExp / (sumExp + 1e-7f) + 1e-7f);
}

/* ------------------------------------------------------------------ */
/*  TrainStep — Node Perturbation ZO-SGD (Zhao et al. 2024 §3.2)     */
/* ------------------------------------------------------------------ */
float ZOTrainer::TrainStep(arm::app::Model& /* classifierModel */, int targetLabel,
                           float learningRate, int numPerturbations)
{
    if (!m_inited) {
        printf_err("[ZO] Not initialized\r\n");
        return -1.0f;
    }

    const int C = m_fcInfo.output_classes;
    const int F = m_fcInfo.input_features;
    const int Q = numPerturbations;

    /* Snapshot weights (Bug 2 guard; weights are not modified during NP loop) */
    std::memcpy(m_stepSnapshot, m_mutableWeights, (size_t)m_fcInfo.weight_bytes);

    /* Clean loss ℓ₀ */
    float lossClean = ComputeLossFromCachedFeature(nullptr, targetLabel);

    /* Clear node gradient accumulator */
    std::memset(m_nodeGradBuf, 0, (size_t)C * sizeof(float));

    /* ---- Q node perturbation passes ---- */
    for (int q = 0; q < Q; q++) {
        uint32_t seed = (uint32_t)(m_stepCount * 1000 + q + 1);

        /* Generate ξ_q ∈ {-1,+1}^C and evaluate perturbed loss ℓ_q */
        float nodeOffset[10]; /* C=10, safe on stack */
        SetSeed(seed);
        for (int c = 0; c < C; c++) {
            nodeOffset[c] = (float)Rademacher();
        }

        float lossPert = ComputeLossFromCachedFeature(nodeOffset, targetLabel);
        float delta    = lossPert - lossClean;

        /* Accumulate ∇̂z += (ℓ_q − ℓ₀) · ξ_q  (replay same seed) */
        SetSeed(seed);
        for (int c = 0; c < C; c++) {
            m_nodeGradBuf[c] += delta * (float)Rademacher();
        }
    }

    /* ZO gradient-estimate norm (measurement only): ‖∇̂z‖₂ over the C nodes. */
    float gradNorm = 0.0f;
    for (int c = 0; c < C; c++) {
        gradNorm += m_nodeGradBuf[c] * m_nodeGradBuf[c];
    }
    gradNorm = sqrtf(gradNorm);

    /* ---- GNS factor: NQ / (NQ + C − 1),  N=1, µ=1 (Eq. 11) ---- */
    float normScale = (float)Q / ((float)Q + (float)C - 1.0f);

    const float fS  = m_fcInfo.feature_scale;
    const int   fZP = m_fcInfo.feature_zero_point;

    /* ---- Apply weight and bias updates per output channel ---- */
    for (int c = 0; c < C; c++) {
        float ws_c = (m_fcInfo.weight_scales_per_ch)
                     ? m_fcInfo.weight_scales_per_ch[c]
                     : m_fcInfo.weight_scale;
        /* SGD is taken in REAL weight space and then re-quantized to the stored
         * INT8 grid: w_real -= lr·∂ℓ/∂w_real, so the INT8 step is the real step
         * divided by the LSB size ws_c:
         *   Δw_int = (lr·∂ℓ/∂w_real)/ws_c = (lr·gz·a_real)/ws_c    →  ×(1/ws_c).
         * This is NOT ×ws_c (= int-space SGD Δq=lr·∂ℓ/∂q): with INT8 ws_c ≪ 1
         * that made every step ~ws_c² too small, so delta_w rounded to 0 for all
         * weights at every lr — frozen weights, delta_params=0, loss_after==before.
         * The earlier 1/ws_c² railed the weights to ±127; 1/ws_c is the fix. */
        float invWs_c = (ws_c > 1e-12f) ? (1.0f / ws_c) : 0.0f;
        float lr_c    = learningRate * normScale * invWs_c / (float)Q;

        float gz = m_nodeGradBuf[c];

        /* Bias update (INT32, unclamped per TFLite bias range).
         * bias_scale = ws_c·fS, and ∂ℓ/∂bias_real = gz, so the INT32 step is
         * gz/(ws_c·fS) = (lr_c·gz)/fS — the weight branch keeps its fS inside
         * a_real, the bias branch divides it back out. */
        float db    = (fS > 1e-12f) ? (lr_c * gz / fS) : 0.0f;
        int delta_b = (int)(db >= 0.0f ? (db + 0.5f) : (db - 0.5f));
        m_mutableBias[c] -= delta_b;

        /* Weight update: ∇W_row_c = ∇z_c · aᵀ */
        for (int f = 0; f < F; f++) {
            float a_f   = fS * (float)(m_featureCache[f] - fZP);
            float gw    = lr_c * gz * a_f;
            int delta_w = (int)(gw >= 0.0f ? (gw + 0.5f) : (gw - 0.5f));
            int w_new   = (int)m_mutableWeights[c * F + f] - delta_w;
            if (w_new >  127) w_new =  127;
            if (w_new < -128) w_new = -128;
            m_mutableWeights[c * F + f] = (int8_t)w_new;
        }
    }

    /* ---- Post-update measurements (device reports raw data only) ---- */

    /* delta_params: fraction of INT8 weight elements that actually changed.
     * INT8 updates round to integers, so many sub-LSB steps leave the stored
     * weight unchanged; this is the raw quantization-saturation measurement.
     * Weights are untouched during the NP loop, so m_stepSnapshot still holds
     * the pre-update integer weights. */
    int changed = 0;
    for (int i = 0; i < m_fcInfo.weight_bytes; i++) {
        if (m_mutableWeights[i] != m_stepSnapshot[i]) changed++;
    }
    float deltaParams = (m_fcInfo.weight_bytes > 0)
                        ? (float)changed / (float)m_fcInfo.weight_bytes
                        : 0.0f;

    /* loss_after (B): cross-entropy at the updated weights, same cached feature. */
    float lossAfter = ComputeLossFromCachedFeature(nullptr, targetLabel);

    /* loss_ema: EMA of the pre-update loss series, α=0.1. Seeded on first step.
     * Reported only — no on-device judgement is derived from it. */
    if (!m_emaInited) {
        m_lossEma   = lossClean;
        m_emaInited = true;
    } else {
        m_lossEma = kLossEmaAlpha * lossClean + (1.0f - kLossEmaAlpha) * m_lossEma;
    }

    m_metrics.loss_before  = lossClean;
    m_metrics.loss_after   = lossAfter;
    m_metrics.loss_ema     = m_lossEma;
    m_metrics.delta_params = deltaParams;
    m_metrics.grad_norm    = gradNorm;

    m_stepCount++;
    m_lastLoss = lossClean;
    return lossClean;
}

/* ------------------------------------------------------------------ */
/*  TrainStepWP — Weight Perturbation ZO-SGD (classic SPSA)           */
/* ------------------------------------------------------------------ */
float ZOTrainer::TrainStepWP(arm::app::Model& /* classifierModel */, int targetLabel,
                             float learningRate, int numPerturbations)
{
    if (!m_inited) {
        printf_err("[ZO] Not initialized\r\n");
        return -1.0f;
    }

    const int C = m_fcInfo.output_classes;
    const int F = m_fcInfo.input_features;
    const int D = C * F;          /* weight perturbation dimension */
    const int Q = numPerturbations;

    /* Snapshot base parameters: pivot for every perturbation AND restore source.
     * Weights → m_stepSnapshot; bias → stack (C=10 is small). */
    std::memcpy(m_stepSnapshot, m_mutableWeights, (size_t)m_fcInfo.weight_bytes);
    int32_t biasSnap[10]; /* C=10, safe on stack */
    for (int c = 0; c < C; c++) biasSnap[c] = m_mutableBias[c];

    /* Clean loss ℓ₀ at the pivot */
    float lossClean = ComputeLossFromCachedFeature(nullptr, targetLabel);

    /* Clear gradient accumulators: weights in m_weightGradBuf, bias in m_nodeGradBuf */
    std::memset(m_weightGradBuf, 0, (size_t)D * sizeof(float));
    std::memset(m_nodeGradBuf,   0, (size_t)C * sizeof(float));

    /* ---- Q weight-perturbation passes (every parameter ±1 LSB) ---- */
    for (int q = 0; q < Q; q++) {
        uint32_t seed = (uint32_t)(m_stepCount * 1000 + q + 1);

        /* Apply ξ_q over all weights and bias, always pivoting on the snapshot.
         * Every coordinate is rewritten each pass, so no in-loop restore needed. */
        SetSeed(seed);
        for (int i = 0; i < D; i++) {
            int w = (int)m_stepSnapshot[i] + Rademacher(); /* ±1 INT8 LSB */
            if (w >  127) w =  127;
            if (w < -128) w = -128;
            m_mutableWeights[i] = (int8_t)w;
        }
        for (int c = 0; c < C; c++) {
            m_mutableBias[c] = biasSnap[c] + Rademacher();  /* ±1 INT32 LSB */
        }

        float lossPert = ComputeLossFromCachedFeature(nullptr, targetLabel);
        float delta    = lossPert - lossClean;

        /* Accumulate ĝ += (ℓ_q − ℓ₀)·ξ_q  (replay same seed → same ξ_q) */
        SetSeed(seed);
        for (int i = 0; i < D; i++) {
            m_weightGradBuf[i] += delta * (float)Rademacher();
        }
        for (int c = 0; c < C; c++) {
            m_nodeGradBuf[c]   += delta * (float)Rademacher();
        }
    }

    /* Restore the pivot parameters before applying the averaged update */
    std::memcpy(m_mutableWeights, m_stepSnapshot, (size_t)m_fcInfo.weight_bytes);
    for (int c = 0; c < C; c++) m_mutableBias[c] = biasSnap[c];

    /* ZO gradient-estimate norm (measurement only): ‖ĝ‖₂ over all perturbed params */
    float gradNorm = 0.0f;
    for (int i = 0; i < D; i++) gradNorm += m_weightGradBuf[i] * m_weightGradBuf[i];
    for (int c = 0; c < C; c++) gradNorm += m_nodeGradBuf[c]   * m_nodeGradBuf[c];
    gradNorm = sqrtf(gradNorm);

    /* ---- GNS factor with the FULL WP perturbation dimension d = C·F + C ----
     * NQ / (NQ + d − 1), N=1, µ=1 (Eq. 11). Far smaller than NP's NQ/(NQ+C−1)
     * because d ≫ C — this term *is* the dimensionality/variance penalty WP pays
     * for perturbing every weight instead of the C logits. */
    const int   dWP       = D + C;
    const float normScale = (float)Q / ((float)Q + (float)dWP - 1.0f);
    const float step      = learningRate * normScale / (float)Q;

    /* ---- Apply averaged update directly in INT8 / INT32 LSB space ----
     * No qaScale here: m_weightGradBuf already holds ∂ℓ/∂w in per-LSB units. */
    for (int i = 0; i < D; i++) {
        float gw    = step * m_weightGradBuf[i];
        int delta_w = (int)(gw >= 0.0f ? (gw + 0.5f) : (gw - 0.5f));
        int w_new   = (int)m_mutableWeights[i] - delta_w;
        if (w_new >  127) w_new =  127;
        if (w_new < -128) w_new = -128;
        m_mutableWeights[i] = (int8_t)w_new;
    }
    for (int c = 0; c < C; c++) {
        float gb    = step * m_nodeGradBuf[c];
        int delta_b = (int)(gb >= 0.0f ? (gb + 0.5f) : (gb - 0.5f));
        m_mutableBias[c] -= delta_b;
    }

    /* ---- Post-update measurements (identical reporting to the NP path) ---- */
    int changed = 0;
    for (int i = 0; i < m_fcInfo.weight_bytes; i++) {
        if (m_mutableWeights[i] != m_stepSnapshot[i]) changed++;
    }
    float deltaParams = (m_fcInfo.weight_bytes > 0)
                        ? (float)changed / (float)m_fcInfo.weight_bytes
                        : 0.0f;

    float lossAfter = ComputeLossFromCachedFeature(nullptr, targetLabel);

    if (!m_emaInited) {
        m_lossEma   = lossClean;
        m_emaInited = true;
    } else {
        m_lossEma = kLossEmaAlpha * lossClean + (1.0f - kLossEmaAlpha) * m_lossEma;
    }

    m_metrics.loss_before  = lossClean;
    m_metrics.loss_after   = lossAfter;
    m_metrics.loss_ema     = m_lossEma;
    m_metrics.delta_params = deltaParams;
    m_metrics.grad_norm    = gradNorm;

    m_stepCount++;
    m_lastLoss = lossClean;
    return lossClean;
}

/* ------------------------------------------------------------------ */
/*  Reset — restore original weights                                  */
/* ------------------------------------------------------------------ */
void ZOTrainer::Reset(arm::app::Model& /* classifierModel */)
{
    if (!m_inited) return;

    std::memcpy(m_mutableWeights, m_originalWeights, (size_t)m_fcInfo.weight_bytes);
    if (m_fcInfo.bias_bytes > 0) {
        std::memcpy(m_mutableBias, m_originalBias, (size_t)m_fcInfo.bias_bytes);
    }

    m_stepCount = 0;
    m_lastLoss  = 0.0f;
    m_metrics   = StepMetrics{};
    m_lossEma   = 0.0f;
    m_emaInited = false;
    info("[ZO] Weights reset to original values\r\n");
}

/* ------------------------------------------------------------------ */
/*  LoadFromSnapshot                                                   */
/* ------------------------------------------------------------------ */
bool ZOTrainer::LoadFromSnapshot(const int8_t* weights,
                                 size_t weightBytes,
                                 const uint8_t* biasBytesPtr,
                                 size_t biasBytes,
                                 int stepCount)
{
    if (!m_inited || !weights || !m_mutableWeights || !m_originalWeights) {
        return false;
    }

    if ((int)weightBytes != m_fcInfo.weight_bytes) {
        return false;
    }

    if ((int)biasBytes != m_fcInfo.bias_bytes) {
        return false;
    }

    std::memcpy(m_mutableWeights,  weights, weightBytes);
    std::memcpy(m_originalWeights, weights, weightBytes);

    if (m_fcInfo.bias_bytes > 0) {
        if (!biasBytesPtr || !m_mutableBias || !m_originalBias) {
            return false;
        }
        std::memcpy(m_mutableBias,  biasBytesPtr, biasBytes);
        std::memcpy(m_originalBias, biasBytesPtr, biasBytes);
    }

    m_stepCount = (stepCount < 0) ? 0 : stepCount;

    /* Loss EMA cannot be reconstructed from a weights-only snapshot; re-seed
     * it on the next step so the reported series restarts cleanly. */
    m_metrics   = StepMetrics{};
    m_lossEma   = 0.0f;
    m_emaInited = false;
    return true;
}
