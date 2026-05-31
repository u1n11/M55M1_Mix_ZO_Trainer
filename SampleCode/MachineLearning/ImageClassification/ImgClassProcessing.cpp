/*
 * Copyright (c) 2022 Arm Limited. All rights reserved.
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
#include "ImgClassProcessing.hpp"
#include "ImageUtils.hpp"
#include "log_macros.h"

namespace arm
{
namespace app
{

ImgClassPreProcess::ImgClassPreProcess(Model *model)
{
    this->m_model = model;
}

bool ImgClassPreProcess::DoPreProcess(const void *data, size_t inputSize)
{
    if (data == nullptr)
    {
        printf_err("Data pointer is null");
    }

    auto input = static_cast<const uint8_t *>(data);
    TfLiteTensor *inputTensor = this->m_model->GetInputTensor(0);

    memcpy(inputTensor->data.data, input, inputSize);
    debug("Input tensor populated \n");

    if (this->m_model->IsDataSigned())
    {
        image::ConvertImgToInt8(inputTensor->data.data, inputTensor->bytes);
    }

    return true;
}

ImgClassPostProcess::ImgClassPostProcess(Classifier &classifier, Model *model,
                                         const std::vector<std::string> &labels,
                                         std::vector<ClassificationResult> &results)
    : m_imgClassifier{classifier},
      m_labels{labels},
      m_results{results}
{
    this->m_model = model;
}

bool ImgClassPostProcess::DoPostProcess()
{
    /* Float32 output = raw logits (no built-in Softmax) → apply Softmax here.
     * Int8/UInt8 output = quantized probability (Softmax already in model) → skip. */
    TfLiteTensor* output = this->m_model->GetOutputTensor(0);
    bool needSoftmax = (output && output->type == kTfLiteFloat32);
    uint32_t topNCount = (m_labels.size() < 5u) ? (uint32_t)m_labels.size() : 5u;

    return this->m_imgClassifier.GetClassificationResults(
               output, this->m_results, this->m_labels, topNCount, needSoftmax);
}

} /* namespace app */
} /* namespace arm */