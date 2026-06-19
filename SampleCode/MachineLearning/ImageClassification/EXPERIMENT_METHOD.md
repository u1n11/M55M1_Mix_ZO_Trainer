# 實驗方法總結：M55M1 MCU 上的影像分類系統與裝置端訓練

> 涵蓋 commit 範圍：`8dbcdd5b` → `4a5c06c4`（共 14 個 commit）

---

## 一、設計背景

本研究在 **Nuvoton M55M1 微控制器**（搭載 Arm Cortex-M55 CPU + Ethos-U55 NPU, H256 配置）上實現 CIFAR-10 影像分類，並進一步探索**裝置端 (on-device) 微調**的可行性。

MCU 資源極為有限：

| 資源 | 容量 |
|---|---|
| 內部 SRAM（NPU tensor arena） | ~832 KB |
| 外部 HyperRAM（CPU tensor arena） | 3–8 MB |
| Flash（模型儲存 + 權重快照） | APROM |

所有設計皆圍繞「如何在嚴苛記憶體限制下完成推論與訓練」展開。

---

## 二、階段一：基礎推論系統建立

### 2.1 模型部署

- 將 PC 端訓練好的 **MobileNetV2 α=0.35**（CIFAR-10, INT8 全量化）透過 Vela 編譯器編譯為 Ethos-U NPU 可執行格式
- 模型以 C array (`.tflite.cc`) 嵌入韌體，存放於 Flash
- 使用 TensorFlow Lite for Microcontrollers (TFLM) 作為推論引擎

### 2.2 UART 命令介面

設計文字命令介面透過 UART 與 PC 端互動，主要命令包括：

| 命令 | 功能 |
|---|---|
| `load_model` | 載入模型至記憶體 |
| `this_is?` | 對當前相機畫面進行分類推論 |
| `show_graph` | 傾印模型結構（算子類型、tensor shape） |

接收方式為**非阻塞式輪詢**：主迴圈持續檢查 UART 接收暫存器，不中斷相機取像流程。

### 2.3 推論狀態機 (Inference State Machine, ISM)

將模型載入、推論、結果輸出解耦為狀態轉換，避免阻塞主迴圈：

```
ISM_IDLE → ISM_LOAD_MODEL → ISM_INFERENCE
                           → ISM_ZO_INIT → ISM_ZO_TRAIN
```

- 每個狀態在一次 `ISM_Process()` 呼叫中只執行一個步驟
- UART 命令僅設定「請求狀態」，實際執行由主迴圈驅動

### 2.4 記憶體佈局設計

此階段解決了「NPU 模型 vs CPU 模型」記憶體需求不同的問題：

| 區域 | 用途 | 位置 | 大小 |
|---|---|---|---|
| `tensorArena` | NPU 模型 tensor arena | SRAM `.bss.NoInit.activation_buf_sram` | 832 KB |
| `cpuTensorArena` | CPU 模型 tensor arena | HyperRAM (0x82000000) `.bss.NoInit.cpu_activation_buf` | 3–8 MB |

**自動偵測機制**：`DoLoadModel()` 掃描模型 FlatBuffer 中的算子類型，若發現包含 `ethos-u` custom op 則判定為 NPU 模型並使用 SRAM arena，否則自動切換至 HyperRAM arena。使用者無需手動指定。

**HyperRAM 健全性測試**：載入 CPU 模型前，在 arena 的起始、中間、末端三個位置寫入/讀回測試值 (`0xDEADBEEF`)，防止硬體問題導致難以診斷的推論錯誤。

---

## 三、階段二：模型拆分與 ZO 訓練器

### 3.1 模型拆分架構

為了實現裝置端訓練，將原本的單一模型拆分為兩部分：

```
Camera Frame (RGB565, 320×240)
    │
    │  imlib_nvt_scale (resize to model input)
    ▼
┌──────────────────────────┐
│  Feature Extractor       │  ← NPU 加速 (Vela 編譯, INT8)
│  (MobileNetV2 骨幹網路)  │    使用 SRAM tensor arena
└───────────┬──────────────┘
            │  Feature Vector (e.g. 448-dim, INT8)
            ▼
┌──────────────────────────┐
│  Classifier (FC 層)      │  ← CPU 執行 (INT8)
│  448 → 10 classes        │    使用 HyperRAM tensor arena
└───────────┬──────────────┘
            │
            ▼
       分類結果 (Top-5)
```

**為什麼要拆分？**

NPU 編譯後的模型權重嵌入在 Flash 中且為唯讀，無法直接修改。拆分後：
- **Feature Extractor**：仍由 NPU 高速執行，權重凍結不變
- **Classifier（FC 層）**：以 CPU 模型獨立執行，權重可被複製到 HyperRAM 進行修改與訓練

### 3.2 ZOTrainer — 零階優化器

這是本實驗的核心方法。基於論文 *"Poor Man's Training on MCUs"* (Zhao et al. 2024) 實作了 **Zeroth-Order SGD (ZO-SGD)**。

#### 為什麼不用反向傳播 (Backpropagation)?

| 原因 | 說明 |
|---|---|
| 記憶體不足 | 標準 BP 需要儲存每層的中間激活值，記憶體開銷遠超 MCU 容量 |
| 框架不支援 | TFLM 是純推論引擎，不支援自動微分 |
| NPU 不透明 | Ethos-U 加速的算子無法取得中間梯度 |

**ZO-SGD 的優勢**：僅需前向推論即可估計梯度方向，完全在 TFLM 框架內實現。

#### 演算法流程（每一個 Training Step）

```
輸入：當前 FC 權重 w, 目標標籤 y, 學習率 η, 擾動次數 Q

1. 計算基準損失 L₀ = CrossEntropy(forward(w), y)

2. For q = 1 to Q:
   a. 用 XORShift PRNG (seed = step×1000+q) 產生 Rademacher 向量 z ∈ {-1,+1}^d
   b. 擾動權重：w'ᵢ = clamp(wᵢ + zᵢ, -128, 127)
   c. 計算擾動損失 Lq = CrossEntropy(forward(w'), y)
   d. 累積梯度估計：gᵢ += (Lq - L₀) × zᵢ
   e. 用相同 seed 還原權重（PRNG 重播）

3. 計算縮放學習率：
   η' = η × [Q / (Q + d - 1)] × [1 / s_w²] × [1 / Q]
   其中 s_w = 權重量化 scale

4. 更新權重：wᵢ ← clamp(wᵢ - η' × gᵢ, -128, 127)
```

#### 記憶體管理技巧

| 技巧 | 說明 |
|---|---|
| Tensor pointer 重定向 | 將 TFLM interpreter 的 FC 權重 tensor data pointer 直接指向 HyperRAM 中的 mutable buffer，無需修改框架原始碼 |
| PRNG seed 重播 | 擾動向量 z 不需要儲存——用相同 seed 重新產生即可還原，節省 d bytes 記憶體 |
| INT16 gradient buffer | 使用 INT16 而非 float32 累積梯度，在 Q 次累加中足夠且節省一半記憶體 |
| Scratch buffer 分區 | 所有 mutable buffer（working weights + backup + gradient）從一塊連續 HyperRAM 區塊中切割 |

#### Scratch Buffer 記憶體佈局

```
scratchBuf (HyperRAM):
┌──────────────────┐
│ mutableWeights   │  d bytes (INT8, 工作副本)
├──────────────────┤
│ mutableBias      │  C×4 bytes (INT32, 工作副本)
├──────────────────┤
│ originalWeights  │  d bytes (INT8, 備份用於 Reset)
├──────────────────┤
│ originalBias     │  C×4 bytes (INT32, 備份)
├──────────────────┤
│ gradientBuf      │  d×2 bytes (INT16, 梯度累積)
└──────────────────┘
總計 ≈ d×4 + C×8 bytes（例 d=4480, C=10: ~18 KB）
```

### 3.3 Snapshot 持久化：Flash 權重保存

訓練好的 FC 權重需要在斷電後保留，設計了 Flash snapshot 機制：

**儲存位置**：APROM Flash 尾端 4 頁（保留區域）

```
APROM Flash:
┌─────────────────────────────┐  0x00000000
│  Application Code + Models  │
│         ...                 │
├─────────────────────────────┤  FMC_APROM_END - 4×PAGE_SIZE
│  ZoFlashHeader (32 bytes)   │  ← magic, version, checksum, stepCount
│  FC Weights (INT8)          │
│  FC Bias (INT32)            │
│  (padding to page boundary) │
└─────────────────────────────┘  FMC_APROM_END
```

**完整性驗證**：使用 FNV-1a checksum 校驗 payload，防止不完整寫入或 Flash 損壞

**自動載入**：開機載入 split model 後自動檢測是否存在有效 snapshot，若有則恢復訓練狀態

**相關 UART 命令**：

| 命令 | 功能 |
|---|---|
| `zo_init` | 初始化 ZO 訓練器 |
| `tra=<label>` | 以指定標籤執行一步訓練（如 `tra=cat`） |
| `zo_reset` | 還原為原始權重 |
| `zo_save` | 手動儲存 snapshot 至 Flash |
| `zo_status` | 顯示訓練器狀態（步數、損失、來源） |
| `zo_lr <val>` | 設定學習率（預設 0.01） |
| `zo_q <val>` | 設定擾動次數 Q（預設 20） |
| `zo_method <np\|wp>` | 選擇 ZO 梯度估計法：NP（節點擾動，預設）或 WP（權重擾動）。僅能在第一步訓練前設定，開始後鎖定，須 `zo_reset` 才能更換——確保單次比較實驗只用單一方法 |

---

## 四、階段三：系統整合與可用性

### 4.1 條件式日誌系統

設計 bitmask-based 的 `LogToken` 系統，避免除錯訊息影響正常運行：

```c
enum LogToken : uint16_t {
    LOG_MODEL_LOAD      = 0x0001,
    LOG_MODEL_INIT      = 0x0002,
    LOG_HYPERRAM_TEST   = 0x0004,
    LOG_INFERENCE_DETAIL = 0x0008,
    LOG_ZO_TRAINING     = 0x0010,
};
```

透過 UART 命令即時控制：`log_on MODEL_LOAD`, `log_off MODEL_LOAD`, `log_all`, `log_none`

### 4.2 幀同步機制

解決「推論期間相機 DMA 覆寫 frame buffer」的問題：

1. 推論開始前設定 `frameBufferFrozen = true`
2. 主迴圈檢查此旗標，若為 true 則跳過相機觸發與等待
3. 推論完成後設定 `frameBufferFrozen = false`
4. MPU 設定 frame buffer 區域為 Non-cacheable，確保 DMA 與 CPU 資料一致

推論結果改為 **Top-5 輸出**，格式為 `[FRAME: N] [RESULT] label1 score1, label2 score2, ... (time)`。

### 4.3 `MobileNetModel` 統一算子註冊

單一 `MobileNetModel` 類別同時支援 NPU 與 CPU 模型：

- `EnlistOperations()` 無條件註冊所有可能用到的算子（包括 `DEQUANTIZE`, `EthosU`）
- 未使用的算子不會造成額外開銷（TFLM 只分配有引用到的算子）
- `m_opsEnlisted` 旗標防止 `Init()` 重試時重複註冊算子

---

## 五、整體系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Loop                            │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │ Camera   │→ │ Display  │  │ UART Cmd │  │    ISM     │ │
│  │ Capture  │  │ Update   │  │ Parser   │  │  Process   │ │
│  └──────────┘  └──────────┘  └────┬─────┘  └─────┬──────┘ │
│       ↑ (frameBufferFrozen)       │               │        │
│       └───────────────────────────┼───────────────┘        │
└───────────────────────────────────┼────────────────────────-┘
                                    │
               ┌────────────────────┼────────────────────┐
               │  ISM States:      ▼                     │
               │   IDLE → LOAD_MODEL → INFERENCE         │
               │                     → ZO_INIT → ZO_TRAIN│
               └─────────────────────────────────────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               │  Memory Layout:   ▼                     │
               │                                         │
               │   SRAM (832 KB)                         │
               │     └─ NPU Feature Extractor arena      │
               │                                         │
               │   HyperRAM (3–8 MB)                     │
               │     ├─ CPU Classifier arena              │
               │     └─ ZO scratch buffers               │
               │                                         │
               │   Flash (APROM tail 4 pages)            │
               │     └─ ZO weight snapshot               │
               └─────────────────────────────────────────┘
```

---

## 六、關鍵設計決策總結

| 問題 | 解決方案 | 相關 Commit |
|---|---|---|
| NPU 模型權重唯讀，無法訓練 | 拆分模型：NPU 做特徵提取，CPU 執行可修改的 FC 層 | `44e10bf2`, `85086902` |
| MCU 無法執行反向傳播 | 使用零階優化 (ZO-SGD)，僅需前向推論即可估計梯度 | `85086902` |
| 擾動向量太大無法全部存下 | PRNG seed 重播技巧：相同 seed 產生相同擾動序列 | `85086902` |
| INT8 量化空間下梯度更新失真 | 量化感知縮放因子 1/s_w² 校正 | `85086902` |
| 斷電後訓練成果丟失 | Flash snapshot 持久化（APROM 尾端保留頁 + FNV-1a 校驗） | `95cbf785` |
| NPU/CPU 模型 arena 需求不同 | 自動偵測 EthosU custom op → 動態選擇 SRAM 或 HyperRAM | `30357c50` |
| 推論時相機 DMA 覆蓋 frame buffer | `frameBufferFrozen` 旗標凍結取像 | `f541a2e0` |
| 除錯資訊過多影響正常輸出 | Bitmask log token 系統，可透過 UART 即時開關 | `0f02edb3` |
| UART 命令被截斷 | 命令 buffer 從 32 擴容至 256 bytes | `4a5c06c4` |

---

## 七、檔案結構

```
ImageClassification/
├── main.cpp                 # 主程式：全域變數定義、主迴圈、相機/顯示/UART 整合
├── GlobalState.hpp          # 所有全域變數的 extern 宣告
├── ImgClassProcessing.cpp   # 前處理（resize→tensor）與後處理（softmax→Top-K）
├── LogConfig.hpp / .cpp     # 條件式日誌系統
├── Model/
│   ├── MobileNetModel.cpp / include/MobileNetModel.hpp  # 模型類別（算子註冊）
│   ├── inference_mngt.cpp / include/inference_mngt.h     # 推論狀態機 (ISM)
│   ├── ZOTrainer.cpp / include/ZOTrainer.hpp             # 零階優化訓練器
│   ├── feature_extractor_int8_vela.tflite.cc             # NPU 特徵提取模型 (C array)
│   └── classifier_int8.tflite.cc                         # CPU 分類器模型 (C array)
├── uart/
│   ├── uart_cmd.cpp / uart_cmd.h                         # UART 命令處理
│   └── UART_COMMANDS.md                                  # 命令文件
└── KEIL/
    ├── ImageClassification.uvprojx                       # Keil 專案（定義 arena 大小巨集）
    └── M55M1.scf                                         # Scatter file（HYPERRAM_BUF 區域）
```
