/*
 * Copyright (c) 2021 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef IMG_CLASS_MOBILENETMODEL_HPP
#define IMG_CLASS_MOBILENETMODEL_HPP

#include "Model.hpp"

/* ------------------------------------------------------------------
 *  Model deployment mode selector
 * ------------------------------------------------------------------
 *  Change MODEL_MODE below to pick which model(s) get loaded and where
 *  the tensor arena(s) live:
 *
 *    MODEL_MODE_SINGLE_NPU          : mbn-v2_w035_int8_vela.tflite
 *                                     full model on Ethos-U NPU, SRAM arena
 *    MODEL_MODE_SINGLE_CPU          : mbn-v2_w035_int8.tflite
 *                                     full model on Cortex-M CPU, HyperRAM arena
 *    MODEL_MODE_SPLIT              : mbn-v2_w035_feature_extractor_int8_vela.tflite (NPU)
 *                                  + mbn-v2_w035_classifier_int8.tflite (CPU, ZO-trainable)
 *                                     both arenas in SRAM
 *    MODEL_MODE_SINGLE_NPU_HYPERRAM: same NPU model as SINGLE_NPU, but the
 *                                     tensor arena is placed in external
 *                                     HyperRAM instead of SRAM.
 *    MODEL_MODE_SPLIT_HYPERRAM     : same split NPU+CPU model as SPLIT, but
 *                                     both tensor arenas live in HyperRAM.
 *
 *  NOTE on the *_HYPERRAM observation modes: the Ethos-U NPU is a separate
 *  bus master and may NOT be able to reach the SPIM-mapped HyperRAM region.
 *  These two modes exist specifically to build and observe/measure that
 *  behaviour; they are not guaranteed to run NPU inference successfully.
 * ------------------------------------------------------------------ */
#define MODEL_MODE_SINGLE_NPU            0
#define MODEL_MODE_SINGLE_CPU            1
#define MODEL_MODE_SPLIT                 2
#define MODEL_MODE_SINGLE_NPU_HYPERRAM   3
#define MODEL_MODE_SPLIT_HYPERRAM        4

#ifndef MODEL_MODE
#define MODEL_MODE              MODEL_MODE_SINGLE_NPU
#endif

/* Derived split-model switch, kept for backward compatibility with the
 * `defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)` checks used across
 * the codebase. Evaluates to 1 in either split mode (SRAM or HyperRAM). */
#define USE_SPLIT_MODEL         (((MODEL_MODE) == MODEL_MODE_SPLIT) || \
                                 ((MODEL_MODE) == MODEL_MODE_SPLIT_HYPERRAM))

/* Source namespace of the active single (monolithic) model. Only meaningful
 * when !USE_SPLIT_MODEL. The NPU (vela) build lives in namespace `mobilenet`,
 * the plain CPU int8 build in namespace `baseline_w035`. Both NPU single
 * modes (SRAM / HyperRAM arena) use the vela `mobilenet` build. */
#if (MODEL_MODE) == MODEL_MODE_SINGLE_CPU
#define SINGLE_MODEL_NS         baseline_w035
#else
#define SINGLE_MODEL_NS         mobilenet
#endif

/* Derived "tensor arena lives in external HyperRAM" switches, used by
 * inference_mngt.cpp to decide arena placement and MPU cache attributes.
 *   SINGLE_MODEL_USE_HYPERRAM : single (monolithic) model arena in HyperRAM
 *                               (CPU model, or the NPU HyperRAM observation mode)
 *   SPLIT_MODEL_USE_HYPERRAM  : both split arenas in HyperRAM */
#define SINGLE_MODEL_USE_HYPERRAM  (((MODEL_MODE) == MODEL_MODE_SINGLE_CPU) || \
                                    ((MODEL_MODE) == MODEL_MODE_SINGLE_NPU_HYPERRAM))
#define SPLIT_MODEL_USE_HYPERRAM   ((MODEL_MODE) == MODEL_MODE_SPLIT_HYPERRAM)

namespace arm
{
namespace app
{

class MobileNetModel : public Model
{

public:
    /* Indices for the expected model - based on input tensor shape */
    static constexpr uint32_t ms_inputRowsIdx     = 1;
    static constexpr uint32_t ms_inputColsIdx     = 2;
    static constexpr uint32_t ms_inputChannelsIdx = 3;

protected:
    /** @brief   Gets the reference to op resolver interface class. */
    const tflite::MicroOpResolver &GetOpResolver() override;

    /** @brief   Adds operations to the op resolver instance. */
    bool EnlistOperations() override;

    const uint8_t *ModelPointer();

    size_t ModelSize();

private:
    /* Maximum number of individual operations that can be enlisted.
     * 13 base ops + DEQUANTIZE + QUANTIZE + FullyConnected + EthosU = 17 max */
    static constexpr int ms_maxOpCnt = 17;

    /* A mutable op resolver instance. */
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;

    /* Track whether operations have already been enlisted (to avoid duplicate registration). */
    bool m_opsEnlisted = false;
};

} /* namespace app */
} /* namespace arm */

#endif /* IMG_CLASS_MOBILENETMODEL_HPP */