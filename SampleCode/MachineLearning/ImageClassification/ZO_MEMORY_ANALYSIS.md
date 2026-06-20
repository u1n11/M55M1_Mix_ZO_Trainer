# ZO 訓練記憶體用量分析

## 一、模型參數

本分析以 `mbn-v2_w035_int8`（MobileNetV2 α=0.35，CIFAR-10）為基準：

| 符號 | 定義 | 值 | 來源 |
|------|------|----|------|
| F | FC 層輸入特徵維度 | **448** | `wTensor->shape()->Get(1)`（ZOTrainer.cpp:140） |
| C | FC 層輸出類別數 | **10** | `wTensor->shape()->Get(0)`（ZOTrainer.cpp:139） |
| `weight_bytes` | C × F，INT8 | **4,480 B** | ZOTrainer.cpp:141 |
| `bias_bytes` | C × sizeof(int32_t) | **40 B** | ZOTrainer.cpp:142 |

F=448 源自 1280 × 0.35 ≈ 448（MobileNetV2 全域平均池化後的特徵寬度）。  
`ZOTrainer.hpp` 第 71 行的注釋 `~91 KB needed` 是針對 F≈1280 的版本，  
本模型（α=0.35，F=448）實際需求約 **31.2 KB**，詳見下表。

---

## 二、ZO Scratch 各分段明細

`ZOTrainer::Init()` 在 [ZOTrainer.cpp:199–251](Model/ZOTrainer.cpp#L199) 以下列順序線性分配 scratch buffer：

```
scratchBuf [ mutableWeights | mutableBias | originalWeights | originalBias |
             stepSnapshot | featureCache | nodeGradBuf | weightGradBuf | (per-ch scales) ]
```

| 分段 | 成員指標 | 元素型別 | 大小公式 | 計算（C=10, F=448） | 程式碼行號 |
|------|----------|----------|----------|---------------------|------------|
| 可修改權重（工作副本） | `m_mutableWeights` | `int8_t` | C × F | 10 × 448 = **4,480 B** | ZOTrainer.cpp:220–221 |
| 可修改偏差 | `m_mutableBias` | `int32_t` | C × 4 | 10 × 4 = **40 B** | ZOTrainer.cpp:223–224 |
| 原始權重備份 | `m_originalWeights` | `int8_t` | C × F | **4,480 B** | ZOTrainer.cpp:226–227 |
| 原始偏差備份 | `m_originalBias` | `int32_t` | C × 4 | **40 B** | ZOTrainer.cpp:229–230 |
| 步驟快照（Bug-2 guard） | `m_stepSnapshot` | `int8_t` | C × F | **4,480 B** | ZOTrainer.cpp:232–233 |
| FC 輸入向量快取 | `m_featureCache` | `int8_t` | F | **448 B** | ZOTrainer.cpp:235–236 |
| 節點梯度累加器 | `m_nodeGradBuf` | `float` | C × 4 | 10 × 4 = **40 B** | ZOTrainer.cpp:238–239 |
| 權重梯度累加器（WP 路徑） | `m_weightGradBuf` | `float` | C × F × 4 | 10 × 448 × 4 = **17,920 B** | ZOTrainer.cpp:241–242 |
| 逐通道量化尺度（條件式） | `weight_scales_per_ch` | `float` | C × 4（nScales > 1 時） | **40 B** | ZOTrainer.cpp:246–251 |
| **ZO Scratch 合計** | | | | **31,968 B ≈ 31.2 KB** | ZOTrainer.cpp:253 |

> `weight_scales_per_ch` 僅在模型使用逐通道（per-channel）量化且 nScales > 1 時配置。
> 上表含此欄位；不含時合計為 **31,928 B**，差異僅 40 B。

---

## 三、Scratch Buffer 在記憶體中的佈局

Scratch buffer 並非獨立靜態陣列，而是從 CPU 分類器 tensor arena 的「閒置尾端」切出。  
見 [inference_mngt.cpp:1071–1075](Model/inference_mngt.cpp#L1071)：

```cpp
size_t   arenaUsed  = classifierModel.GetArenaUsedBytes();  // FC 推論後 arena 實際佔用
size_t   guardBytes = 1024u;                                 // 1 KB 對齊保護
uint8_t* scratchBuf = clsArena + arenaUsed + guardBytes;     // ZO scratch 起始位址
size_t   scratchSz  = (size_t)clsArenaSize - arenaUsed - guardBytes;  // 可用空間
```

| 區段 | 配置方式 | 大小 | 說明 |
|------|----------|------|------|
| `cpuTensorArena`（全體） | 靜態 `.bss` | `CPU_ACTIVATION_BUF_SZ` = **128 KB** | [main.cpp:67,70](main.cpp#L67) |
| FC head 推論佔用 | TFLite Micro arena | 遠小於 128 KB（單一 FC 層） | `GetArenaUsedBytes()` 取得 |
| 1 KB guard | 靜態保留 | **1,024 B** | 對齊保護 |
| **ZO scratch** | 尾端剩餘空間 | **≈ 31.2 KB** | 實際需求遠低於剩餘量 |

FC head 僅有一層全連接算子，推論佔用 arena 的量極少，因此 ZO scratch 有充裕的剩餘空間。

---

## 四、BP 對照：為何反向傳播無法在 SRAM 中執行

### 4.1 推論峰值記憶體參考

| 模式 | 峰值記憶體 | 來源 |
|------|-----------|------|
| NPU（特徵提取器 vela 編譯） | `ACTIVATION_BUF_SZ` ≈ **832 KB**（SRAM） | EXPERIMENT_METHOD.md |
| CPU 全模型（無 NPU，單一 tflite） | **> 1.1 MB** | inference_mngt.cpp:164 |

TFLite Micro 的推論 arena 採「貪婪型記憶體規劃」（greedy memory planner）：  
每個算子執行完畢後，其輸入張量的空間即可重用於後續張量。  
因此推論峰值 = **同時存活張量的最大疊加**，遠小於所有中間激活值的總和。

### 4.2 BP 激活值估算

反向傳播（Backpropagation）需要在前向傳遞中**保留每一層的中間激活值**，  
因為反向計算 ∂L/∂W 和 ∂L/∂a 都需要對應層的前向輸出：

```
BP 激活值儲存 ≈ 所有中間張量總和
             ≈ 推論峰值 × 5–7 倍   ← 峰值只是同時存活的最大截面，BP 需全部保留
```

| 比較項目 | ZO Scratch（本實作，F=448） | BP 全模型（MobileNetV2 0.35） |
|----------|----------------------------|-------------------------------|
| 記憶體型態 | 僅 FC 頭層的 8 個固定緩衝區 | 骨幹 52 層激活值 + 梯度 + 優化器狀態 |
| 最大單一緩衝 | `m_weightGradBuf` = 17,920 B | 最大特徵圖 ~110 KB（擴張層輸出） |
| **合計估算** | **≈ 31.2 KB** | **≈ 1.1 MB × 5–7 = 5.5–7.7 MB** |
| M55M1 內部 SRAM（總量） | 2 MB | **明顯超出** |
| 外部 HyperRAM | 8 MB | Adam 狀態再 × 2–3，仍逼近或超出 8 MB |

### 4.3 ZO 的記憶體優勢

ZO Node Perturbation 只擾動最後一層 FC 的 **C=10 個 logit 節點**：

- 前向推論照常由 NPU 執行（不需儲存任何骨幹激活值）
- 梯度估計在 `float z[10]`、`float nodeOffset[10]`（棧上暫存，40 B）中完成
- 唯一的 SRAM 額外開銷就是 ZO scratch 的 **31.2 KB**
- 相對全模型 BP 的 5.5–7.7 MB，記憶體壓縮比約 **175–250 倍**

---

## 五、ZO Scratch 配置流程總覽

```
Init() 呼叫流程（ZOTrainer.cpp:61–288）
│
├── 從 FlatBuffer 讀取最後一個 FULLY_CONNECTED op
│   └── 取得 weight/bias/feature tensor 的 shape 與量化參數（:125–196）
│
├── 計算 needed = weight_bytes×3 + bias_bytes×2 + F + C×4 + C×F×4 [+ C×4]（:199–209）
│
├── 線性分配 scratch buffer 8（或 9）個分段（:218–251）
│   ├── m_mutableWeights  ← ptr；ptr += weight_bytes
│   ├── m_mutableBias     ← ptr；ptr += bias_bytes
│   ├── m_originalWeights ← ptr；ptr += weight_bytes
│   ├── m_originalBias    ← ptr；ptr += bias_bytes
│   ├── m_stepSnapshot    ← ptr；ptr += weight_bytes
│   ├── m_featureCache    ← ptr；ptr += input_features
│   ├── m_nodeGradBuf     ← ptr；ptr += C × sizeof(float)
│   ├── m_weightGradBuf   ← ptr；ptr += weight_bytes × sizeof(float)
│   └── [weight_scales_per_ch ← ptr；ptr += nScales × sizeof(float)]（per-ch 才有）
│
├── 從 Flash 複製原始權重至 mutableWeights / originalWeights（:264–270）
└── 重導 interpreter tensor 資料指標至可修改緩衝（:273–276）
```
