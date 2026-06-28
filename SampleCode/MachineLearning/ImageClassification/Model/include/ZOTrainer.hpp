#ifndef ZO_TRAINER_HPP
#define ZO_TRAINER_HPP

#include "Model.hpp"
#include <cstdint>
#include <cstddef>

/**
 * @brief ZOTrainer — Zeroth-Order Optimizer for the INT8 FC layer.
 *
 * Implements BP-free weight update using quantized ZO-SGD with node
 * perturbations, following "Poor Man's Training on MCUs" (Zhao et al. 2024).
 *
 * Usage:
 *   1. Call Init() after loading the classifier model with preserveAllTensors=true.
 *   2. Fill classifier input tensor with extractor features.
 *   3. Call CacheFeature() to run one inference and cache the FC input vector.
 *   4. Call TrainStep() with the ground-truth label index.
 *   5. Repeat for each sample/epoch.
 */
class ZOTrainer
{
public:
    /** @brief  FC layer metadata extracted from the FlatBuffer. */
    struct FCInfo {
        int     weight_tensor_idx;
        int     bias_tensor_idx;
        int     feature_tensor_idx;      /**< FC input (a) tensor index         */
        int     input_features;          /**< F ≈ 1280                          */
        int     output_classes;          /**< C = 10                            */
        int     weight_bytes;            /**< C * F (INT8)                      */
        int     bias_bytes;              /**< C * sizeof(int32_t)               */
        float   weight_scale;            /**< Scalar fallback when per_ch=null  */
        float*  weight_scales_per_ch;    /**< Per-channel array [C], or nullptr */
        int     weight_zero_point;
        float   feature_scale;           /**< Input activation scale            */
        int     feature_zero_point;
        float   output_scale;
        int     output_zero_point;
    };

    /**
     * @brief  Per-step measurement data reported to the external host.
     *
     * The device only *measures* and reports these values. It performs NO
     * convergence judgement and changes NO training behaviour based on them —
     * the host decides convergence/stopping from the reported log series.
     */
    struct StepMetrics {
        float loss_before;   /**< Cross-entropy loss before the update (A)        */
        float loss_after;    /**< Cross-entropy loss after the update  (B)        */
        float loss_ema;      /**< EMA of loss_before, smoothing α=0.1             */
        float delta_params;  /**< Fraction [0,1] of INT8 weights that changed     */
        float grad_norm;     /**< L2 norm of the ZO node-gradient estimate        */

        /* ---- Sub-LSB diagnostics: why the INT8 weights do/don't move ---- *
         * The INT8 update is round(dw). If dw_max < 0.5 the step is entirely
         * sub-LSB and NO weight can change regardless of grad_norm — the lr is
         * simply below the quantization grid for this layer. Compare dw_max
         * against 0.5 to tell "lr too small" from "write-back broken".         */
        float dw_max;        /**< Max |Δw| (float, pre-rounding) over all weights */
        float dw_mean;       /**< Mean |Δw| (float, pre-rounding) over weights    */
        float db_max;        /**< Max |Δbias| (float, pre-rounding) over channels */
        float lr_eff;        /**< Effective per-ch lr_c at the strongest-grad node*/
        float a_rms;         /**< Real-space RMS of the cached FC input feature   */
    };

    /** @brief  EMA smoothing factor for loss reporting (host-agnostic). */
    static constexpr float kLossEmaAlpha = 0.1f;

    ZOTrainer();

    /**
     * @brief  Initialise the trainer.
     *
     * Locates the last FULLY_CONNECTED operator in the classifier FlatBuffer,
     * reads quantization parameters (per-channel if present), copies the const
     * Flash weights to a mutable scratch buffer, and redirects the interpreter's
     * tensor data pointers.
     *
     * @param classifierModel  Model initialised with preserveAllTensors=true.
     * @param scratchBuf       Pointer to a mutable buffer in HyperRAM (~91 KB needed).
     * @param scratchSize      Size of the scratch buffer in bytes.
     * @return true on success.
     */
    bool Init(arm::app::Model& classifierModel, uint8_t* scratchBuf, size_t scratchSize);

    /**
     * @brief  Run one classifier inference and cache the FC input vector.
     *
     * Must be called once per training sample, after filling the classifier
     * input tensor with the extractor features, and before TrainStep.
     *
     * @param model  The classifier model (must be Init()'d).
     * @return true on success.
     */
    bool CacheFeature(arm::app::Model& model);

    /**
     * @brief  Execute one ZO-SGD training step using node perturbation.
     *
     * Perturbs the logit vector z (C=10 floats) rather than the weights, so
     * no NPU invocations occur during the Q perturbation evaluations. The
     * feature vector used is the one cached by the last CacheFeature() call.
     *
     * @param classifierModel   The classifier model (kept for API compat; unused).
     * @param targetLabel       Ground-truth CIFAR-10 class index (0–9).
     * @param learningRate      Learning rate (e.g. 0.01).
     * @param numPerturbations  Number of node perturbations Q per step (e.g. 50).
     * @return Cross-entropy loss before the weight update.
     *
     * @note  Per-step measurement data (loss A→B, loss EMA, delta_params,
     *        grad_norm) is recorded in the StepMetrics accessible via
     *        GetLastMetrics(). No convergence judgement is made on-device.
     */
    float TrainStep(arm::app::Model& classifierModel, int targetLabel,
                    float learningRate, int numPerturbations);

    /**
     * @brief  Execute one ZO-SGD training step using WEIGHT perturbation (WP).
     *
     * Classic ZO-SGD: perturbs every FC parameter (C×F INT8 weights + C INT32
     * bias) by ±1 LSB with a Rademacher sign, evaluates the perturbed loss, and
     * accumulates the SPSA gradient estimate over Q passes — discarding the
     * analytic ∇W = ∇z·aᵀ structure that NP exploits. Provided for a quantitative
     * NP-vs-WP comparison only; for this single-FC head NP has ~1000× lower
     * gradient variance (perturbation dim C vs C×F).
     *
     * Shares all buffers, quantization handling, and StepMetrics reporting with
     * TrainStep(); the weight gradient is accumulated in m_weightGradBuf and the
     * bias gradient in m_nodeGradBuf. Same signature/semantics as TrainStep so
     * the two are drop-in interchangeable at the call site.
     *
     * @note  Because WP applies the GNS factor with the full perturbation
     *        dimension d = C·F + C (≫ C), the same learningRate behaves very
     *        differently than under NP — WP typically needs a larger LR and/or
     *        larger Q to move the INT8 weights off their quantization grid.
     */
    float TrainStepWP(arm::app::Model& classifierModel, int targetLabel,
                      float learningRate, int numPerturbations);

    bool               IsInitialized()  const { return m_inited; }
    int                GetStepCount()   const { return m_stepCount; }
    float              GetLastLoss()    const { return m_lastLoss; }
    const StepMetrics& GetLastMetrics() const { return m_metrics; }
    const FCInfo&  GetFCInfo()      const { return m_fcInfo; }
    size_t         GetMemoryUsed()  const { return m_memUsed; }
    const int8_t*  GetMutableWeights() const { return m_mutableWeights; }
    const int32_t* GetMutableBias()    const { return m_mutableBias; }

    /**
     * @brief  Print s_theta diagnostics for lr calibration.
     *
     * Prints per-channel weight quantization scale ws_c, the real-space RMS of
     * each output channel's weights (s_theta_c), the global s_theta across all
     * weights, and — if featureCached is true — the feature RMS and the derived
     * minimum LR per channel needed to produce at least one non-zero INT8 update
     * (for Q=nomQ perturbations).
     *
     * Call after Init(); set featureCached=true only after CacheFeature().
     */
    void PrintSTheta(bool featureCached = false, int nomQ = 50) const;

    /**
     * @brief  Overwrite mutable/original FC parameters from an external snapshot.
     */
    bool LoadFromSnapshot(const int8_t* weights,
                          size_t weightBytes,
                          const uint8_t* biasBytesPtr,
                          size_t biasBytes,
                          int stepCount);

    /** @brief  Restore weights to their original values (as loaded from Flash). */
    void Reset(arm::app::Model& classifierModel);

private:
    /* XORShift PRNG (paper Appendix C) */
    uint32_t m_xorState;
    void     SetSeed(uint32_t seed);
    uint32_t XorRand();
    int      Rademacher();

    /* CPU-only forward pass using cached feature vector */
    float ComputeLossFromCachedFeature(const float* nodeOffset, int targetLabel);

    /* Scratch-allocated buffers */
    int8_t*  m_mutableWeights;   /**< Working copy of FC weights (INT8)          */
    int32_t* m_mutableBias;      /**< Working copy of FC bias   (INT32)          */
    int8_t*  m_originalWeights;  /**< Backup of original weights                 */
    int32_t* m_originalBias;     /**< Backup of original bias                    */
    int8_t*  m_stepSnapshot;     /**< Per-step weight snapshot (Bug 2 guard)     */
    int8_t*  m_featureCache;     /**< Cached FC input vector a (INT8, size F)    */
    float*   m_nodeGradBuf;      /**< Node gradient ∇̂z accumulator (C floats)   */
    float*   m_weightGradBuf;    /**< Weight gradient ∇̂W accumulator (C×F float)*/

    FCInfo      m_fcInfo;
    int         m_stepCount;
    float       m_lastLoss;
    StepMetrics m_metrics;     /**< Last step's reported measurement data       */
    float       m_lossEma;     /**< Running EMA of loss_before                  */
    bool        m_emaInited;   /**< False until the first step seeds the EMA    */
    bool        m_inited;
    size_t      m_memUsed;
};

#endif /* ZO_TRAINER_HPP */
