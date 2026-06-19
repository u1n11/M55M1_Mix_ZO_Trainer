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
| Command format | Plain text string; some commands take an argument (see below) |
| Response | Human-readable log lines prefixed with `[TAG]` |

Commands are parsed in `UART_ProcessCommand()` ([uart_cmd.cpp](uart_cmd.cpp)).

**Execution model**: Most inference/training commands are **non-blocking** — they set an ISM (Inference State Machine) state and return immediately. The MCU processes the state in the next `ISM_Process()` main-loop tick. A few commands run **synchronously** inside `UART_ProcessCommand()` itself: `zo_reset`, `zo_status`, `zo_save`, and all log control commands.

---

## Build Modes

The set of available commands depends on the compile-time flag `USE_SPLIT_MODEL`:

| Flag | Mode | Active Commands |
|------|------|-----------------|
| `USE_SPLIT_MODEL=0` | Standard single-model | `load_model`, `this_is?`, `show_graph` |
| `USE_SPLIT_MODEL=1` | Split extractor + classifier with ZO training | `load_model`, `this_is?`, `zo_init`, `zo_reset`, `zo_status`, `zo_save`, `tra=<label>`, `zo_lr <val>`, `zo_q <val>`, `zo_method <np\|wp>` |

Log control commands (`log_all`, `log_none`, `log_status`, `log_on <token>`, `log_off <token>`) are available in **both** modes.

---

## Standard Mode Commands (`USE_SPLIT_MODEL=0`)

### `load_model`

**Purpose**: Load the MobileNet model from Flash into the tensor arena.

**ISM state set**: `ISM_LOAD_MODEL` (processed on next main-loop tick)

**Expected MCU output**:
```
[LOAD] time=<ms> mem=<used>/<total>B zo=N/A
```

**Must be called before**: `this_is?`, `show_graph`

---

### `this_is?`

**Purpose**: Capture one camera frame, run inference, and print Top-5 classification results.

**ISM state set**: `ISM_INFERENCE`

**Expected MCU output**:
```
[FRAME: 1] [RESULT] airplane 0.920000, ship 0.012300, truck 0.009100, bird 0.008400, car 0.006500 (xx.xxms)
```

- `[FRAME: N]` — monotonic counter incremented on every inference/training operation; lets you correlate the displayed frame with the result.
- Scores are softmax probabilities (0.0–1.0).
- Timing in milliseconds shown at the end.

**Prerequisite**: `load_model` must be called first.

---

### `show_graph`

**Purpose**: Dump the TFLite flatbuffer graph structure to UART — lists all ops, tensor shapes, and data types.

**ISM state set**: `ISM_DUMP_GRAPH`

**Use case**: Verifying that NPU delegate ops are correctly registered; debugging model structure.

---

## Split-Model Mode Commands (`USE_SPLIT_MODEL=1`)

In split mode the inference pipeline uses two models:

- **Extractor model** (`extractorModel`): runs on the NPU; produces a feature vector from the camera frame.
- **Classifier model** (`classifierModel`): a small INT8 fully-connected layer whose weights are mutable at runtime for ZO training.

---

### `load_model`

**Purpose**: Load both the extractor and classifier models.

**ISM state set**: `ISM_LOAD_SPLIT_MODEL`

**Expected MCU output** (fresh boot, no flash snapshot):
```
[LOAD] time=<ms> mem=<used>/<total>B zo=NO
[ZO] Weight source: BUILTIN (no valid flash snapshot)
```

**Expected MCU output** (valid persisted snapshot found):
```
[LOAD] time=<ms> mem=<used>/<total>B zo=YES
[ZO] Loaded classifier snapshot from flash (steps=<n>)
[ZO] Weight source: FLASH
```

**Must be called before**: any other split-mode command.

---

### `this_is?`

**Purpose**: Capture one camera frame → run extractor → run classifier → print Top-5 results.

**ISM state set**: `ISM_SPLIT_INFERENCE`

**Expected MCU output**:
```
[FRAME: 1] [RESULT] cat 0.850000, dog 0.073000, deer 0.021000, horse 0.018000, frog 0.011000 (xx.xxms) (ext:xx.xx+cls:xx.xx)
```

- `(ext:xx.xx+cls:xx.xx)` — extractor and classifier timing shown separately, in milliseconds.

**Prerequisite**: `load_model` must be called first.

---

### `zo_init`

**Purpose**: Initialise the ZOTrainer — locates the FC layer in the classifier FlatBuffer, copies the const Flash weights to a mutable RAM buffer, and attempts to load a persisted snapshot from APROM flash.

**ISM state set**: `ISM_ZO_INIT`

**Expected MCU output** (no valid flash snapshot):
```
[ZO] Init OK: FC 448→10, weights 4480 B, bias 40 B
[ZO] Weight source: BUILTIN (no valid flash snapshot)
```

**Expected MCU output** (snapshot restored from flash):
```
[ZO] Init OK: FC 448→10, weights 4480 B, bias 40 B
[ZO] Loaded classifier snapshot from flash (steps=<n>)
[ZO] Weight source: FLASH
```

**Must be called before**: `tra=<label>`, `zo_reset`, `zo_status`, `zo_save`.

---

### `zo_reset`

**Purpose**: Restore FC weights in RAM to their original built-in values (undoing all training), then erase the persisted ZO snapshot from APROM flash.

**Execution**: Synchronous — runs immediately inside `UART_ProcessCommand()`, no ISM state change.

**Expected MCU output** (trainer was initialised):
```
[ZO] Weights reset to original
[ZO] Persisted snapshot erased from flash @0x........
```

**Error output** (trainer not yet initialised):
```
[ZO] Trainer not initialised
```

---

### `zo_status`

**Purpose**: Print current trainer readiness, weight source, runtime step count, and persisted flash step count.

**Execution**: Synchronous.

**Expected MCU output** (ready):
```
[ZO] Status: trainer=READY, source=FLASH, runtime_steps=12, flash_steps=12
```

**Expected MCU output** (not initialised):
```
[ZO] Status: trainer=NOT_INITIALISED
```

---

### `zo_save`

**Purpose**: Force-save the current classifier weight snapshot to APROM flash (manual save). Training steps do **not** auto-save; call this when you decide to persist.

**Execution**: Synchronous.

**Expected MCU output**:
```
[ZO] Snapshot saved to flash @0x........ (steps=12)
```

**Error output**:
```
[ZO] Manual save to flash failed
```

---

### `tra=<label_name>`

**Purpose**: Capture one camera frame, extract features, then perform one ZO-SGD training step targeting the named CIFAR-10 class.

**Argument**: `<label_name>` — one of the ten CIFAR-10 class names (case-sensitive, must match an entry in `labels[]`).

**ISM state set**: `ISM_ZO_TRAIN`

**Valid labels** (CIFAR-10):
```
airplane  automobile  bird  cat  deer
dog       frog        horse ship truck
```

**Example**:
```
tra=cat
```

**Expected MCU output**:
```
[FRAME: 2] [ZO] Step 1 | method=NP | target=cat | loss=2.3021->1.8754 | loss_ema=2.3021 | delta_params=3.27% | grad_norm=12.8431 | lr=0.010000 | Q=50 | Time: 142.50ms | Mem: 93184 bytes
```

- `method=NP|WP` — the gradient-estimation method used this step (set via
  `zo_method`; NP by default).

The device reports **raw per-step measurements only**. It performs no
convergence judgement and never changes training behaviour based on these
values — the external host decides convergence/stopping from this log series.

- `loss=A->B` — cross-entropy loss before (A) and after (B) the update this step.
- `loss_ema` — exponential moving average of the pre-update loss, smoothing
  factor α=0.1 (seeded with the first step's loss). Reported for trend
  reconstruction; not used for any on-device decision.
- `delta_params` — percentage of INT8 FC weight elements that actually changed
  this step (raw quantization-saturation measurement: INT8 updates round to
  integers, so sub-LSB steps leave many weights unchanged).
- `grad_norm` — L2 norm of the ZO node-gradient estimate (auxiliary host signal).
- `lr` — current learning rate (host-set; echoed verbatim).
- `Q` — number of Rademacher perturbations used (host-set; echoed verbatim).

**Error output** (unknown label):
```
[ZO] Unknown label 'kitten'. Valid labels:
  [0] airplane
  [1] automobile
  ...
```

---

### `zo_lr <value>`

**Purpose**: Set the ZO-SGD learning rate. Must be a positive float.

**Execution**: Synchronous (takes effect on the next `tra=` step).

**Example**:
```
zo_lr 0.005
```

**Expected MCU output**:
```
[ZO] Learning rate set to 0.005000
```

**Error output** (value ≤ 0):
```
[ZO] Invalid learning rate (must be > 0)
```

**Default**: Value of `zoLearningRate` as initialised in `main.cpp`.

---

### `zo_q <value>`

**Purpose**: Set the number of Rademacher perturbations Q per training step. Must be a positive integer. Higher Q gives a better gradient estimate but each step takes longer.

**Execution**: Synchronous (takes effect on the next `tra=` step).

**Example**:
```
zo_q 50
```

**Expected MCU output**:
```
[ZO] Perturbations Q set to 50
```

**Error output** (value ≤ 0):
```
[ZO] Invalid Q value (must be > 0)
```

**Default**: Value of `zoNumPerturbations` as initialised in `main.cpp`.

---

### `zo_method <np|wp>`

**Purpose**: Select the zeroth-order gradient-estimation method for `tra=` steps.

| Value | Method | Perturbs | Perturbation dim |
|-------|--------|----------|------------------|
| `np` | **Node Perturbation** (default) | the C logits, reconstructs ∇W = ∇z·aᵀ analytically | C = 10 |
| `wp` | **Weight Perturbation** | every FC weight + bias by ±1 LSB (classic SPSA) | C·F + C ≈ 12 810 |

**Execution**: Synchronous.

**Locking**: A comparison run must stay on a single method, so switching is
**only allowed before the first training step**. Once `GetStepCount() > 0`, the
command is rejected until you `zo_reset` (which clears the step count and
releases the lock). With no command issued, the method defaults to **NP**.

**Example**:
```
zo_method wp
```

**Expected MCU output**:
```
[ZO] Method set to WP
```

**Output when already on that method**:
```
[ZO] Method already WP; unchanged
```

**Error output** (switch attempted mid-run):
```
[ZO] Cannot switch method after 12 step(s). Send 'zo_reset' first.
```

**Error output** (bad value):
```
[ZO] Unknown method 'foo'. Valid: np, wp
```

> **Note on hyper-parameters**: WP applies the GNS factor with the full
> perturbation dimension `d = C·F + C` (≫ C), so the *same* `zo_lr` value
> moves the INT8 weights far less under WP than under NP. Expect WP to need a
> larger LR and/or larger Q to register non-zero `delta_params`. This is the
> dimensionality/variance penalty WP pays — and the headline result of the
> NP-vs-WP comparison.

**Default**: `ZO_METHOD_NP`, as initialised in `main.cpp`.

---

## Log Control Commands

By default, only essential output is shown (results, timing, memory usage). Verbose diagnostic messages are suppressed. These commands let you toggle log tokens at runtime without recompiling.

### `log_all`

**Purpose**: Enable all verbose log tokens at once.

**Execution**: Synchronous.

**Expected MCU output**:
```
[LOG] All debug tokens enabled
```

---

### `log_none`

**Purpose**: Disable all verbose log tokens. Restores minimal output (results, timing, errors only).

**Execution**: Synchronous.

**Expected MCU output**:
```
[LOG] All debug tokens disabled
```

---

### `log_status`

**Purpose**: Print the current on/off state of every log token.

**Execution**: Synchronous.

**Expected MCU output**:
```
[LOG CONFIG] Current log tokens:
  LOG_MODEL_LOAD:       OFF
  LOG_MODEL_INIT:       OFF
  LOG_INFERENCE_DETAIL: OFF
  LOG_ZO_TRAINING:      OFF
```

---

### `log_on <token>`

**Purpose**: Enable a single log token.

**Execution**: Synchronous.

**Valid tokens**:

| Token | Enables |
|-------|---------|
| `load` | Model binary and arena information during load (`LOG_MODEL_LOAD`) |
| `init` | Model initialisation diagnostics (`LOG_MODEL_INIT`) |
| `inf` | Per-layer inference detail (`LOG_INFERENCE_DETAIL`) |
| `zo` | ZO trainer initialisation and step detail (`LOG_ZO_TRAINING`) |

**Example**:
```
log_on zo
```

**Expected MCU output**:
```
[LOG] LOG_ZO_TRAINING enabled
```

**Error output** (unknown token):
```
[LOG] Unknown token: hyperram
[LOG] Valid tokens: load, init, inf, zo
```

---

### `log_off <token>`

**Purpose**: Disable a single log token (same valid tokens as `log_on`).

**Execution**: Synchronous.

**Example**:
```
log_off load
```

**Expected MCU output**:
```
[LOG] LOG_MODEL_LOAD disabled
```

---

## Default Log Behavior

| Output type | Shown by default | Controlled by |
|-------------|-----------------|---------------|
| Inference results (`[RESULT]`) | Yes | Always on |
| ZO training step summary (`[ZO] Step …`) | Yes | Always on |
| Timing and memory usage | Yes | Always on |
| Model load / arena detail | No | `log_on load` |
| Model init diagnostics | No | `log_on init` |
| Per-layer inference detail | No | `log_on inf` |
| ZO trainer init / step detail | No | `log_on zo` |
| Deep TFLM allocator / subgraph dump | No | Build-time: `IMGCLS_ENABLE_MODEL_LOAD_VERBOSE_LOGS=1` |

---

## Error Reference

| MCU output | Cause |
|-----------|-------|
| `[ERR] Unknown command: <cmd>` | Typo, or command not compiled into this build mode |
| `[LOG] Unknown token: <x>` | Token name is not one of `load`, `init`, `inf`, `zo` |
| `[ZO] Trainer not initialised` | `zo_reset`/`zo_status`/`zo_save` called before `zo_init` |
| `[ZO] Unknown label '<x>'` | Label string does not match any entry in `labels[]` |
| `[ZO] Invalid learning rate (must be > 0)` | `zo_lr` value is zero or negative |
| `[ZO] Invalid Q value (must be > 0)` | `zo_q` value is zero or negative |
| `[ZO] Unknown method '<x>'` | `zo_method` value is not `np` or `wp` |
| `[ZO] Cannot switch method after <n> step(s). Send 'zo_reset' first.` | `zo_method` issued mid-run (step count > 0); locked for comparison integrity |
| `[ZO] Manual save to flash failed` | Flash write error during `zo_save` |
| `[ZO] Reset done in RAM, but clearing flash snapshot failed` | `zo_reset` succeeded in RAM but flash erase failed |

---

## Typical Workflows

### A. Standard Inference Loop

```
load_model      ← one-time on boot
this_is?        ← capture + classify; read back [RESULT] line
this_is?
...
```

---

### B. ZO On-Device Training Session

```
load_model          ← load extractor + classifier
zo_init             ← copy weights to mutable RAM; restore flash snapshot if present
zo_method np        ← (optional) pick NP or WP; MUST be set before the first tra=
zo_lr 0.01          ← (optional) tune learning rate
zo_q 20             ← (optional) tune perturbation count
this_is?            ← baseline accuracy before training

# Training loop — repeat per class, per epoch:
tra=cat             ← point camera at a cat, train one step
tra=cat
tra=dog
...

this_is?            ← evaluate updated accuracy

zo_save             ← persist when satisfied
zo_status           ← verify step count matches expectation

zo_reset            ← (if needed) undo all training and clear flash snapshot
```

---

### C. Host-Side Automation (Python)

```python
import serial, time

ser = serial.Serial('COM_PORT', 115200, timeout=5)

def send_cmd(cmd: str) -> str:
    ser.write((cmd + '\r\n').encode())
    time.sleep(0.2)          # Wait for ISM tick to complete
    return ser.read_all().decode(errors='replace')

# Standard inference
print(send_cmd('load_model'))
for _ in range(5):
    print(send_cmd('this_is?'))

# ZO training
print(send_cmd('load_model'))
print(send_cmd('zo_init'))
print(send_cmd('zo_lr 0.01'))
print(send_cmd('zo_q 20'))
for label in ['cat', 'dog', 'airplane']:
    for _ in range(30):
        print(send_cmd(f'tra={label}'))
print(send_cmd('this_is?'))
print(send_cmd('zo_save'))
```

> **Timing note**: Inference and training steps can take hundreds of milliseconds. Wait for the `[RESULT]` or `[ZO] Step` response line before sending the next command, or implement a line-based read loop rather than a fixed `time.sleep`.

---

## Source Files

| File | Role |
|------|------|
| [uart_cmd.cpp](uart_cmd.cpp) | Command parser and dispatch table |
| [include/uart_cmd.h](include/uart_cmd.h) | `UART_ProcessCommand()` declaration |
| [../Model/include/inference_mngt.h](../Model/include/inference_mngt.h) | ISM state enum and API |
| [../GlobalState.hpp](../GlobalState.hpp) | Shared globals (`zoLearningRate`, `zoNumPerturbations`, `labels`, etc.) |
| [../Model/include/ZOTrainer.hpp](../Model/include/ZOTrainer.hpp) | ZOTrainer class (FC layer ZO-SGD) |
