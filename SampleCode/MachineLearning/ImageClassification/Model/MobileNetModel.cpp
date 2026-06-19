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
#include "MobileNetModel.hpp"
#include "log_macros.h"

/* Compile-time switch for verbose model-load internals.
 * Keep disabled by default to avoid noisy boot/load output. */
#ifndef IMGCLS_ENABLE_MODEL_LOAD_VERBOSE_LOGS
#define IMGCLS_ENABLE_MODEL_LOAD_VERBOSE_LOGS 0
#endif


const tflite::MicroOpResolver &arm::app::MobileNetModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::MobileNetModel::EnlistOperations()
{
    /* Prevent duplicate registration if Init() is called more than once
     * (e.g. after a previous failure due to insufficient tensor arena). */
    if (this->m_opsEnlisted)
    {
        return true;
    }

    if (kTfLiteOk != this->m_opResolver.AddDepthwiseConv2D())
    {
        return false;
    }
    if (kTfLiteOk != this->m_opResolver.AddConv2D())
    {
        return false;
    }
    if (kTfLiteOk != this->m_opResolver.AddAveragePool2D())
    {
        return false;
    }
    if (kTfLiteOk != this->m_opResolver.AddAdd())
    {
        return false;
    }
    if (kTfLiteOk != this->m_opResolver.AddMul())
    {
        return false;
    }
    if (kTfLiteOk != this->m_opResolver.AddPad())
    {
        return false;
    }
    if (kTfLiteOk != this->m_opResolver.AddMean())
    {
        return false;
    }
    if (kTfLiteOk != this->m_opResolver.AddRelu())
    {
        return false;
    }
    if (kTfLiteOk != this->m_opResolver.AddRelu6())
    {
        return false;
    }
    if (kTfLiteOk != this->m_opResolver.AddReshape())
    {
        return false;
    }
    if (kTfLiteOk != this->m_opResolver.AddSoftmax())
    {
        return false;
    }
    if (kTfLiteOk != this->m_opResolver.AddFullyConnected())
    {
        return false;
    }
    if (kTfLiteOk != this->m_opResolver.AddQuantize())
    {
        return false;
    }

    /* Float16 models store weights as float16 and need DEQUANTIZE to
     * convert them to float32 at runtime.  Adding it unconditionally is
     * harmless for int8/NPU models (unused ops are simply ignored). */
    if (kTfLiteOk != this->m_opResolver.AddDequantize())
    {
        return false;
    }

#if defined(ARM_NPU)
    /* Always register EthosU — unused ops are harmless, and split-model
     * extractor (vela) needs it even when the monolithic model is CPU-only. */
    if (kTfLiteOk == this->m_opResolver.AddEthosU())
    {
#if IMGCLS_ENABLE_MODEL_LOAD_VERBOSE_LOGS
        info("Added %s support to op resolver\n",
             tflite::GetString_ETHOSU());
#endif
    }
    else
    {
        printf_err("Failed to add Arm NPU support to op resolver.");
        return false;
    }
#endif /* ARM_NPU */

    this->m_opsEnlisted = true;
    return true;
}

#if !(defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1))
namespace arm
{
namespace app
{
namespace SINGLE_MODEL_NS
{
extern const uint8_t *GetModelPointer();
extern size_t GetModelLen();
}
}
} /* namespace SINGLE_MODEL_NS */

const uint8_t *arm::app::MobileNetModel::ModelPointer()
{
    return arm::app::SINGLE_MODEL_NS::GetModelPointer();
}

size_t arm::app::MobileNetModel::ModelSize()
{
    return arm::app::SINGLE_MODEL_NS::GetModelLen();
}
#else
namespace arm
{
namespace app
{
namespace feature_extractor_npu {
    extern const uint8_t *GetModelPointer();
    extern size_t GetModelLen();
}
namespace classifier_cpu_int8 {
    extern const uint8_t *GetModelPointer();
    extern size_t GetModelLen();
}
}
}
#endif /* USE_SPLIT_MODEL */