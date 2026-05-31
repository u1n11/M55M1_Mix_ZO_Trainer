# M55M1 ImageClassification — UART Command Reference

> **Target MCU**: Nuvoton M55M1 with Ethos-U NPU (H256)
> **Task**: CIFAR-10 image classification via on-board camera, controlled from a host PC over UART.

---

## Protocol

| Item | Detail |
|------|--------|
| Transport | UART (serial) |
| Direction | Host PC → MCU (command); MCU → Host PC (log/result) |
| Encoding | ASCII, newline (`\r\n`) terminated |
| Command format | Plain text string, no argument unless noted |
| Response | Human-readable log lines prefixed with `[TAG]` |

Commands are parsed in `UART_ProcessCommand()` ([uart_cmd.cpp](uart_cmd.cpp)).
Most inference/training commands are **non-blocking** on the host side: they set an ISM (Inference State Machine) state and return immediately. The MCU processes the state in the next `ISM_Process()` main-loop tick. Some commands execute synchronously in `UART_ProcessCommand()` (for example: `zo_reset`, `zo_status`, `zo_save`, and log control commands).

---

## Build Modes

Commands available depend on the compile-time flag `USE_SPLIT_MODEL`:

| Flag | Mode | Active Commands |
|------|------|-----------------|
| `USE_SPLIT_MODEL=0` (default) | Standard single-model | `load_model`, `this_is?`, `show_graph` |
| `USE_SPLIT_MODEL=1` | Split extractor+classifier with ZO training | `load_model`, `this_is?`, `zo_init`, `zo_reset`, `zo_status`, `zo_save`, `tra=<label>`, `zo_set_lr <val>`, `zo_set_q <val>` |

Common commands in both modes: `log_all`, `log_none`, `log_status`, `log_on <token>`, `log_off <token>`

---

## Standard Mode Commands (`USE_SPLIT_MODEL=0`)

### `load_model`

**Purpose**: Load the MobileNet model from Flash into tensor arena (SRAM for NPU, HyperRAM for CPU).

**ISM state**: `ISM_LOAD_MODEL`

**Expected MCU output**:
```
[LOAD] time=<ms> mem=<used>/<total>B zo=N/A
```

**Must be called before**: `this_is?`, `show_graph`

---

### `this_is?`

**Purpose**: Capture one camera frame and run inference. Prints Top-5 classification candidates.

**ISM state**: `ISM_INFERENCE`

**Frame synchronisation**: During inference, the frame buffer is frozen to prevent camera updates, ensuring the inference result matches the displayed frame.

**Expected MCU output**:
```
[FRAME: 1] [RESULT] airplane 0.920000, ship 0.012300, truck 0.009100, bird 0.008400, car 0.006500 (xx.xxms)
```

The `[FRAME: N]` counter increments with each inference/training operation, allowing you to verify that the displayed frame and inference result are synchronized.

**Prerequisite**: `load_model` must have been called first (`modelLoaded == true`).

---

### `show_graph`

**Purpose**: Dump the TFLite flatbuffer graph structure to UART — lists all ops, tensor shapes, and types.

**ISM state**: `ISM_DUMP_GRAPH`

**Use case**: Debugging model structure; verifying that NPU delegate ops are correctly registered.

---

## Split-Model Mode Commands (`USE_SPLIT_MODEL=1`)

In this mode the pipeline is split into two models:
- **Extractor model** (`extractorModel`): runs on NPU, produces a feature vector.
- **Classifier model** (`classifierModel`): INT8 FC layer, weights are mutable for ZO training.

### `load_model`

**Purpose**: Load both extractor and classifier models.

**ISM state**: `ISM_LOAD_SPLIT_MODEL`

**Expected MCU output**:
```
[LOAD] time=<ms> mem=<used>/<total>B zo=NO
```

If a valid persisted snapshot exists, `zo` will be `YES` and load will also print:
```
[ZO] Loaded classifier snapshot from flash (steps=<n>)
[ZO] Weight source: FLASH
```

**Must be called before**: any other split-mode command.

---

### `this_is?`

**Purpose**: Capture camera frame → run extractor → run classifier → print Top-5 classification candidates.

**ISM state**: `ISM_SPLIT_INFERENCE`

**Frame synchronisation**: During inference, the frame buffer is frozen to prevent camera updates, ensuring the inference result matches the displayed frame.

**Expected MCU output**:
```
[FRAME: 1] [RESULT] cat 0.850000, dog 0.073000, deer 0.021000, horse 0.018000, frog 0.011000 (xx.xxms) (ext:xx.xx+cls:xx.xx)
```

The `[FRAME: N]` counter increments with each inference/training operation, allowing you to verify frame consistency.

---

### `zo_init`

**Purpose**: Initialise the ZOTrainer — locates the FC layer in the classifier FlatBuffer, copies const Flash weights to a mutable HyperRAM buffer, then tries loading a persisted snapshot from APROM flash.

**ISM state**: `ISM_ZO_INIT`

**Expected MCU output**:
```
[ZO] Init OK: FC 448→10, weights 4480 B, bias 40 B
[ZO] Weight source: FLASH
```
or
```
[ZO] Weight source: BUILTIN (no valid flash snapshot)
```

**Must be called before**: `tra=<label>`, `zo_reset`, `zo_status`, `zo_save`.

---

### `zo_reset`

**Purpose**: Restore FC weights to their original Flash values in RAM (undo all training), and erase the persisted ZO snapshot in APROM flash.

**Direct execution** (no ISM state change; runs synchronously).

**Expected MCU output**:
```
[ZO] Weights reset to original
[ZO] Persisted snapshot erased from flash @0x........
```

---

### `tra=<label_name>`

**Purpose**: Capture camera frame, extract features, then perform one ZO-SGD training step targeting the given CIFAR-10 class label.

**Argument**: `<label_name>` — one of the CIFAR-10 class names (case-sensitive, must match `labels[]`).

**ISM state**: `ISM_ZO_TRAIN`

**CIFAR-10 valid labels**:
```
airplane  automobile  bird  cat  deer
dog  frog  horse  ship  truck
```

**Example**:
```
tra=cat
```

**Expected MCU output**:
```
[ZO] Step 1 | target=cat | loss=2.3021 → 1.8754 | lr=0.010000 | Q=20
```

Training steps do **not** auto-save to flash. Use `zo_save` when you decide to persist.

**Error output** (invalid label):
```
[ZO] Unknown label 'kitten'. Valid labels:
  [0] airplane
  [1] automobile
  ...
```

---

### `zo_set_lr <value>`

**Purpose**: Set the ZO-SGD learning rate (float, must be > 0).

**Example**:
```
zo_set_lr 0.005
```

**Expected MCU output**:
```
[ZO] Learning rate set to 0.005000
```

**Default**: As initialised in `main.cpp` (`zoLearningRate`).

---

### `zo_set_q <value>`

**Purpose**: Set the number of Rademacher perturbations Q per training step (integer, must be > 0). Higher Q → better gradient estimate, slower step.

**Example**:
```
zo_set_q 50
```

**Expected MCU output**:
```
[ZO] Perturbations Q set to 50
```

**Default**: As initialised in `main.cpp` (`zoNumPerturbations`).

---

### `zo_status`

**Purpose**: Show trainer readiness, current model source, runtime step count, and persisted flash step count.

**Expected MCU output**:
```
[ZO] Status: trainer=READY, source=FLASH, runtime_steps=12, flash_steps=12
```
or
```
[ZO] Status: trainer=NOT_INITIALISED
```

---

### `zo_save`

**Purpose**: Force-save current trainer snapshot to APROM flash (manual save).

**Expected MCU output**:
```
[ZO] Snapshot saved to flash @0x........ (steps=12)
```

---

## Typical Workflows

### A. Standard Inference Loop (host PC automation)

```
1. load_model          # One-time: load model on boot
2. this_is?            # Repeated: capture + classify
3. this_is?
4. ...
```

Host reads back each `[RESULT] ...` line after sending `this_is?`.

---

### B. ZO On-Device Training Session

```
1. load_model          # Load extractor + classifier
2. zo_init             # Init ZO trainer (copies weights to HyperRAM)
3. zo_set_lr 0.01      # (optional) tune hyperparams
4. zo_set_q 20         # (optional)
5. this_is?            # Check baseline accuracy before training

# Training loop (repeat N times per class, per epoch):
6. tra=cat             # Point camera at cat, train one step
7. tra=cat             # ...
8. tra=dog
9. ...

# Evaluate:
10. this_is?           # Check updated accuracy

# Persist only when needed:
11. zo_save            # Save current trained snapshot to APROM flash

# Reset if needed:
12. zo_reset           # Restore original weights in RAM and clear persisted flash snapshot
```

---

### C. Host-Side Automation Script Skeleton (Python pseudocode)

```python
import serial, time

ser = serial.Serial('COM_PORT', 115200, timeout=2)

def send_cmd(cmd: str) -> str:
    ser.write((cmd + '\r\n').encode())
    time.sleep(0.1)
    return ser.read_all().decode(errors='replace')

# --- Standard mode ---
print(send_cmd('load_model'))
for _ in range(10):
    result = send_cmd('this_is?')
    print(result)

# --- ZO training mode ---
print(send_cmd('load_model'))
print(send_cmd('zo_init'))
print(send_cmd('zo_set_lr 0.01'))
for label in ['cat', 'dog', 'airplane']:
    for step in range(30):
        print(send_cmd(f'tra={label}'))
print(send_cmd('this_is?'))
```

> **Note**: The MCU may take hundreds of milliseconds to complete an inference or training step. Wait for the `[RESULT]`/`[ZO]` response line before sending the next command, or use a line-based read loop.

---

## Log Control Commands (Debug)

By default, only **critical** logs are shown (results, memory usage, timing). Detailed diagnostic logs for model loading are hidden. Use these commands to enable/disable specific log tokens:

### `log_all`

**Purpose**: Enable all verbose logging (model loading, initialization, HyperRAM tests, etc.)

**Usage**:
```
log_all
```

**Effect**: All debug tokens are enabled. MCU output becomes verbose.

---

### `log_none`

**Purpose**: Disable all verbose logging. Keep only critical output (results, memory, timing).

**Usage**:
```
log_none
```

**Expected MCU output**:
```
[LOG] All debug tokens disabled
```

---

### `log_status`

**Purpose**: Print current log token configuration.

**Usage**:
```
log_status
```

**Expected MCU output**:
```
[LOG CONFIG] Current log tokens:
  LOG_MODEL_LOAD:       OFF
  LOG_MODEL_INIT:       OFF
  LOG_HYPERRAM_TEST:    OFF
  LOG_INFERENCE_DETAIL: OFF
  LOG_ZO_TRAINING:      OFF
```

---

### `log_on <token>`

**Purpose**: Enable a specific log token.

**Valid tokens**:
- `load` → Model binary and arena information
- `init` → Model initialization diagnostics
- `hyperram` → HyperRAM sanity checks (split model)
- `inference` → Per-layer inference timing (if available)
- `zo` → ZO trainer initialization details

**Usage**:
```
log_on load
log_on hyperram
```

**Expected MCU output**:
```
[LOG] LOG_MODEL_LOAD enabled
[LOG] LOG_HYPERRAM_TEST enabled
```

---

### `log_off <token>`

**Purpose**: Disable a specific log token.

**Usage**:
```
log_off load
```

**Expected MCU output**:
```
[LOG] LOG_MODEL_LOAD disabled
```

---

## Default Log Behavior

| Message Type | Default | Token | Example |
|-------------|---------|-------|---------|
| **Results** | ✅ Always shown | (none) | `[FRAME: 1] [RESULT] cat 0.850000, dog 0.073000, deer 0.021000, horse 0.018000, frog 0.011000 (45.23ms)` |
| **Memory** | ✅ Always shown | (none) | `[MEM] Tensor arena: 1024/2048 bytes` |
| **Timing** | ✅ Always shown | (none) | Time shown in result  |
| Deep TFLM model internals | ❌ Compile-time off by default | `IMGCLS_ENABLE_MODEL_LOAD_VERBOSE_LOGS` | Allocator/subgraph/tensor dump/operator list |
| Init diagnostics | ❌ Hidden | `LOG_MODEL_INIT` | Model setup messages |
| HyperRAM tests | ❌ Hidden | `LOG_HYPERRAM_TEST` | RAM sanity check messages |
| ZO training | ❌ Hidden | `LOG_ZO_TRAINING` | FC layer info during init |

> To enable deep model-load internals at build time, define `IMGCLS_ENABLE_MODEL_LOAD_VERBOSE_LOGS=1` in compiler defines.

---

## Usage Example: Minimal Output

```python
import serial, time

ser = serial.Serial('COM_PORT', 115200, timeout=2)

def send_cmd(cmd: str) -> str:
    ser.write((cmd + '\r\n').encode())
    time.sleep(0.1)
    return ser.read_all().decode(errors='replace')

# Ensure minimal output
print(send_cmd('log_none'))

# Training session - only critical logs shown
print(send_cmd('load_model'))       # Silent (log_on load to see details)
print(send_cmd('zo_init'))          # Shows: [MEM] Classifier arena, [ZO] Ready
print(send_cmd('zo_set_lr 0.01'))   # Silent
print(send_cmd('tra=cat'))          # Shows: [FRAME: 1] [ZO] Step 1 | Label: cat | ...
print(send_cmd('tra=dog'))          # Shows: [FRAME: 2] [ZO] Step 2 | Label: dog | ...
print(send_cmd('this_is?'))         # Shows: [FRAME: 3] [RESULT] cat 0.850000, dog 0.073000, deer 0.021000, horse 0.018000, frog 0.011000 (45.23ms) (ext:45.10+cls:0.13)

# Enable debug logs for troubleshooting
print(send_cmd('log_on load'))      # Shows [ISM] layer-level load diagnostics only
print(send_cmd('log_status'))       # Check current config
```

---

## Error Handling

| MCU output | Meaning |
|-----------|---------|
| `[ERR] Unknown command: <cmd>` | Typo or wrong build mode |
| `[LOG] Unknown token: <x>` | Invalid log token name |
| `[ZO] Trainer not initialised` | `zo_reset` called before `zo_init` |
| `[ZO] Unknown label '<x>'` | Label string doesn't match `labels[]` |
| `[ZO] Invalid learning rate` | `zo_set_lr` value ≤ 0 |
| `[ZO] Invalid Q value` | `zo_set_q` value ≤ 0 |

---

## Source Files

| File | Role |
|------|------|
| [uart_cmd.cpp](uart_cmd.cpp) | Command parser and dispatch table |
| [uart/include/uart_cmd.h](include/uart_cmd.h) | `UART_ProcessCommand()` declaration |
| [../Model/include/inference_mngt.h](../Model/include/inference_mngt.h) | ISM state enum and API |
| [../GlobalState.hpp](../GlobalState.hpp) | All shared global variables (`zoLearningRate`, `zoNumPerturbations`, etc.) |
| [../Model/include/ZOTrainer.hpp](../Model/include/ZOTrainer.hpp) | ZOTrainer class (FC layer ZO-SGD) |
