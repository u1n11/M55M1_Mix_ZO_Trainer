# ZOTrainer 架構說明

> 對象檔案:[`ZOTrainer.cpp`](./ZOTrainer.cpp) / [`include/ZOTrainer.hpp`](./include/ZOTrainer.hpp)
> 呼叫端:[`inference_mngt.cpp`](./inference_mngt.cpp)
> 論文依據:Zhao et al. 2024《Poor Man's Training on MCUs》

---

## 1. 整體架構

`ZOTrainer` 是一個**零階(Zeroth-Order, ZO)優化器**,只訓練分類模型**最後一層全連接層(FC)**的 INT8 權重與 INT32 偏置,完全不需要反向傳播(BP-free)。

### 邏輯分區

| 區段 | 行數 | 職責 |
|------|------|------|
| Constructor | `ZOTrainer.cpp:16-35` | 初始化成員指標為 null/0 |
| XORShift PRNG | `ZOTrainer.cpp:40-56` | 產生 Rademacher ±1 隨機擾動 |
| **`Init`** | `ZOTrainer.cpp:61-289` | **解析 TFLM 模型、權重重定向** |
| `CacheFeature` | `ZOTrainer.cpp:294-313` | 跑一次推論,快取 FC 輸入特徵 a |
| `ComputeLossFromCachedFeature` | `ZOTrainer.cpp:318-356` | CPU 反量化 FC + softmax-CE loss |
| **`TrainStep`** (NP) | `ZOTrainer.cpp:361-479` | **節點擾動 ZO-SGD** |
| `TrainStepWP` (WP) | `ZOTrainer.cpp:484-601` | 權重擾動 ZO-SGD(對照組) |
| `Reset` / `LoadFromSnapshot` | `ZOTrainer.cpp:606-663` | 還原 / 載入權重快照 |

---

## 2. 一次訓練步驟的流程

```
填入 extractor 特徵到分類器輸入
  → CacheFeature()      跑 1 次 NPU+CPU 推論,memcpy 出 FC 輸入向量 a
  → TrainStep()         以快取的 a 在 CPU 上做 Q 次擾動評估(不再碰 NPU)
       ├ ComputeLoss(nullptr)   乾淨 loss ℓ₀
       ├ Q 次: 擾動 logit → ComputeLoss(ξ) → 累積梯度估計
       ├ 套用權重/偏置更新(含量化感知縮放)
       └ 記錄 StepMetrics(loss A→B、EMA、delta_params、grad_norm)
```

**設計重點:擾動只加在 10 維 logit 上,所以 Q 次評估全在 CPU 完成,不需重跑 NPU**,這是它能在 MCU 上即時訓練的關鍵。

---

## 3. 四大關鍵部分

### 3.1 節點擾動(Node Perturbation)

核心:`ZOTrainer.cpp:383-401`

```cpp
for (int q = 0; q < Q; q++) {
    uint32_t seed = (uint32_t)(m_stepCount * 1000 + q + 1);
    float nodeOffset[10];
    SetSeed(seed);
    for (int c = 0; c < C; c++) nodeOffset[c] = (float)Rademacher();  // ξ_q ∈ {-1,+1}^C

    float lossPert = ComputeLossFromCachedFeature(nodeOffset, targetLabel);
    float delta    = lossPert - lossClean;                            // ℓ_q − ℓ₀

    SetSeed(seed);                                                    // 重播同一 seed
    for (int c = 0; c < C; c++)
        m_nodeGradBuf[c] += delta * (float)Rademacher();              // ∇̂z += δ·ξ_q
}
```

- 擾動向量 ξ 只作用於 **C=10 個輸出節點(logit)**,而非 C×F≈12800 個權重 → 變異數比 WP 低約 1000 倍。
- `nodeOffset` 加進前向的位置:`ComputeLossFromCachedFeature` `ZOTrainer.cpp:342-344`(`z[c] += nodeOffset[c]`)。
- PRNG 與 Rademacher:`ZOTrainer.cpp:40-56`。利用「相同 seed 重播」省去儲存 ξ 的記憶體。

---

### 3.2 解析全連接 TFLM 模型與權重重定向

全部在 `Init`:

| 步驟 | 行數 |
|------|------|
| 定位最後一個 FC op(分類頭,刻意不 break) | `ZOTrainer.cpp:99-113` |
| 取出三個輸入張量索引(特徵 a / 權重 / 偏置) | `ZOTrainer.cpp:121-123` |
| 讀權重形狀 metadata | `ZOTrainer.cpp:139-142` |
| 讀 per-channel 權重 scale | `ZOTrainer.cpp:149-159` |
| 讀特徵 / 輸出量化參數 | `ZOTrainer.cpp:161-196` |
| 複製 Flash const 權重到可變 scratch | `ZOTrainer.cpp:264-270` |
| **重定向 interpreter 張量指標** | `ZOTrainer.cpp:272-276` |

關鍵重定向(`ZOTrainer.cpp:272-276`):

```cpp
evalW->data.data = m_mutableWeights;   // interpreter 之後即用訓練後的權重
if (evalB && evalB->data.data && m_fcInfo.bias_bytes > 0)
    evalB->data.data = m_mutableBias;
```

#### 是否還使用 HyperRAM?

**是,間接使用 —— 取決於編譯期巨集。** 此檔案本身不直接碰 HyperRAM,而是接收外部傳入的 `scratchBuf`。實際分配在 `inference_mngt.cpp:1128-1134`:

```cpp
uint8_t* scratchBuf = clsArena + arenaUsed + guardBytes;   // 接在分類器 arena 之後
zoTrainer->Init(classifierModel, scratchBuf, scratchSz);
```

而 `clsArena` 在 `SPLIT_MODEL_USE_HYPERRAM` 模式下指向 HyperRAM(`inference_mngt.cpp:807-816`,
`SPLIT_CLS_ARENA_PTR = HYPERRAM_ARENA_ADDR + ACTIVATION_BUF_SZ`)。

> **結論**:當以 split-HyperRAM 模式編譯時,ZOTrainer 的可變權重 / 梯度 scratch(約 91 KB,含 `weightGradBuf` C×F×float)就落在 HyperRAM;若以純 SRAM 模式編譯則在 SRAM。目前架構仍保留 HyperRAM 路徑。
> scratch 記憶體計算與分區布局:`ZOTrainer.cpp:198-253`。

---

### 3.3 ZO-SGD 演算法

梯度估計累積後的更新階段:`ZOTrainer.cpp:410-441`

```cpp
float normScale = (float)Q / ((float)Q + (float)C - 1.0f);   // GNS 因子 (Eq.11)
...
float lr_c = learningRate * normScale * qaScale_c / (float)Q; // 每通道有效學習率
float gz   = m_nodeGradBuf[c];

// 偏置更新(INT32)
m_mutableBias[c] -= delta_b;                                  // db = lr_c·gz 四捨五入

// 權重更新:∇W_row_c = ∇z_c · aᵀ
for (int f = 0; f < F; f++) {
    float a_f = fS * (float)(m_featureCache[f] - fZP);
    float gw  = lr_c * gz * a_f;
    int w_new = m_mutableWeights[c*F+f] - round(gw);
    clamp(w_new, -128, 127);                                  // INT8 飽和
}
```

要點:
- **GNS(Gradient Normalization Scaling)因子** `NQ/(NQ+C−1)`:`ZOTrainer.cpp:411`(論文 Eq.11,N=1, µ=1)。
- 利用 NP 的解析結構 `∇W = ∇z·aᵀ`,把 logit 梯度展開成權重梯度(`ZOTrainer.cpp:432-434`),不必逐權重擾動。
- `grad_norm` 僅量測用:`ZOTrainer.cpp:404-408`。
- 對照組 WP(經典 SPSA,擾動所有權重):`TrainStepWP` `ZOTrainer.cpp:511-572`,GNS 用完整維度 d=C·F+C(`ZOTrainer.cpp:554-556`),可與 NP 做變異數比較。

---

### 3.4 量化感知縮放(Quantization-Aware Scaling)

#### (a) 前向反量化 — `ZOTrainer.cpp:326-339`

```cpp
float ws_c = per_ch ? weight_scales_per_ch[c] : weight_scale;  // 每通道 scale
float w_f  = ws_c * (m_mutableWeights[c*F+f] - weight_zero_point);
float a_f  = fS   * (m_featureCache[f] - feature_zero_point);
acc += w_f * a_f;
acc += (ws_c * fS) * m_mutableBias[c];     // bias_scale = w_scale·feat_scale
```

這是 TFLite INT8 FC 的反量化慣例:bias 的 scale = 權重 scale × 特徵 scale。

#### (b) 更新時的量化感知縮放因子 — `ZOTrainer.cpp:418-422`

```cpp
float qaScale_c = (ws_c > 1e-12f) ? (1.0f / (ws_c * ws_c)) : 1.0f;
float lr_c      = learningRate * normScale * qaScale_c / (float)Q;
```

`qaScale = 1/ws²` 把「以實數梯度表示的更新」轉回「INT8 LSB 整數格點」上的步長,使 scale 很小(權重解析度高)的通道得到較大的整數步長,避免更新被四捨五入吃光。更新後再 clamp 回 [−128,127](`ZOTrainer.cpp:437-439`)。

`delta_params`(`ZOTrainer.cpp:450-456`)量測實際有多少 INT8 權重真的改變,即量化飽和度。

> 對比:WP 路徑直接在 LSB 空間運作,**不需** qaScale(`ZOTrainer.cpp:558-559` 註解說明)。

---

## 4. NP vs WP 摘要

| 項目 | NP(`TrainStep`) | WP(`TrainStepWP`) |
|------|------------------|---------------------|
| 擾動對象 | C 個 logit 節點 | 全部 C×F 權重 + C 偏置 |
| 擾動維度 | C = 10 | d = C·F + C ≈ 12810 |
| GNS 分母 | NQ + C − 1 | NQ + d − 1 |
| 梯度變異數 | 低(約 1/1000) | 高 |
| 量化感知縮放 | 需要(`1/ws²`) | 不需要(直接 LSB 空間) |
| NPU 重跑 | 否(僅 CPU) | 否(僅 CPU) |
