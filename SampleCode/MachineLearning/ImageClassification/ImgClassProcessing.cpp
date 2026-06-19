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
        return false;
    }

    TfLiteTensor *inputTensor = this->m_model->GetInputTensor(0);
    auto *src = static_cast<const uint8_t *>(data);

    /* resize 通常已直接寫進 input tensor；只有來源與 tensor buffer 不同才需 copy */
    if (src != static_cast<const uint8_t *>(inputTensor->data.data))
    {
        memcpy(inputTensor->data.data, src, inputSize);
    }
    debug("Input tensor populated \n");

    /* 對齊 PC 訓練 pipeline：x/255 → ImageNet normalize → 用模型自己的
     * input scale/zero_point 量化。mean/std 順序為 R,G,B（HWC 交錯）。
     * 若 imlib RGB888 實際為 B,G,R，將 mean/stdv 反序即可。 */
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};

    if (inputTensor->type == kTfLiteInt8)
    {
        const float scale = inputTensor->params.scale;
        const int   zp    = inputTensor->params.zero_point;

        /* 每通道仿射：out = round(u·A[c] + B[c]) */
        float A[3], B[3];
        for (int c = 0; c < 3; ++c)
        {
            A[c] = 1.0f / (255.0f * stdv[c] * scale);
            B[c] = (-mean[c] / stdv[c]) / scale + (float)zp;
        }

        auto *u = static_cast<uint8_t *>(inputTensor->data.data);
        auto *q = static_cast<int8_t  *>(inputTensor->data.data);
        const size_t n = inputTensor->bytes; /* int8：元素數 == 位元組數 */
        for (size_t i = 0; i < n; ++i)
        {
            const int c = (int)(i % 3);
            float v = (float)u[i] * A[c] + B[c]; /* 先讀 u[i] 再寫 q[i]，同位元組 in-place 安全 */
            int   r = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
            if (r > 127)  r = 127;
            if (r < -128) r = -128;
            q[i] = (int8_t)r;
        }
    }
    else if (inputTensor->type == kTfLiteFloat32)
    {
        /* float 模型退路：就地覆寫成正規化 float。float(4B) > uint8(1B)，
         * 必須由尾端往前做，避免覆蓋尚未讀取的來源位元組。 */
        auto *u = static_cast<uint8_t *>(inputTensor->data.data);
        auto *f = reinterpret_cast<float *>(inputTensor->data.data);
        const size_t n = inputSize;
        for (size_t i = n; i-- > 0; )
        {
            const int c = (int)(i % 3);
            f[i] = ((float)u[i] / 255.0f - mean[c]) / stdv[c];
        }
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