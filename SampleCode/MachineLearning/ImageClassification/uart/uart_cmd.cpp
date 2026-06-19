#include <cstdio>
#include <cstring>
#include <string>

/* GlobalState.hpp → MobileNetModel.hpp defines USE_SPLIT_MODEL — must come
 * before inference_mngt.h so the conditional enum values are compiled in. */
#include "../GlobalState.hpp"
#include "uart_cmd.h"
#include "inference_mngt.h" /* Use ISM API (USE_SPLIT_MODEL already defined) */
#include "log_macros.h"
#include "LogConfig.hpp"

/* ------------------------------------------------------------------ */
/*  Command Functions                                                 */
/* ------------------------------------------------------------------ */

#if !(defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1))
static void Cmd_LoadModel(void)
{
    /* Just set state, execution happens in ISM_Process() */
    ISM_SetState(ISM_LOAD_MODEL);
}

static void Cmd_Inference(void)
{
    /* Just set state, execution happens in ISM_Process() */
    ISM_SetState(ISM_INFERENCE);
}

static void Cmd_DumpGraph(void)
{
    /* Just set state, execution happens in ISM_Process() */
    ISM_SetState(ISM_DUMP_GRAPH);
}
#endif /* !USE_SPLIT_MODEL */

#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
static void Cmd_LoadSplitModel(void)
{
    ISM_SetState(ISM_LOAD_SPLIT_MODEL);
}

static void Cmd_SplitInference(void)
{
    ISM_SetState(ISM_SPLIT_INFERENCE);
}

static void Cmd_ZOInit(void)
{
    ISM_SetState(ISM_ZO_INIT);
}

static void Cmd_ZOReset(void)
{
    if (zoTrainer && zoTrainer->IsInitialized()) {
        zoTrainer->Reset(classifierModel);
        if (!ISM_ZO_ClearFlash()) {
            printf_err("[ZO] Reset done in RAM, but clearing flash snapshot failed\r\n");
        }
    } else {
        info("[ZO] Trainer not initialised\r\n");
    }
}

static void Cmd_ZOStatus(void)
{
    ISM_ZO_PrintStatus();
}

static void Cmd_ZOSave(void)
{
    if (!ISM_ZO_SaveToFlash()) {
        printf_err("[ZO] Manual save to flash failed\r\n");
    }
}
#endif /* USE_SPLIT_MODEL */

/* ------------------------------------------------------------------ */
/*  Command Table (no-argument commands)                             */
/* ------------------------------------------------------------------ */

typedef void (*CmdHandler)(void);

typedef struct {
    const char *cmd;
    CmdHandler handler;
} CmdEntry;

const CmdEntry g_cmd_table[] = {
#if !(defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1))
    { "load_model",  Cmd_LoadModel  },
    { "this_is?",    Cmd_Inference  },
    { "show_graph",  Cmd_DumpGraph  },
#endif /* !USE_SPLIT_MODEL */
#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
    /* Unified command names (same as non-split mode) */
    { "load_model",  Cmd_LoadSplitModel  },
    { "this_is?",    Cmd_SplitInference  },
    { "zo_init",     Cmd_ZOInit     },
    { "zo_reset",    Cmd_ZOReset    },
    { "zo_status",   Cmd_ZOStatus   },
    { "zo_save",     Cmd_ZOSave     },
#endif /* USE_SPLIT_MODEL */
    { NULL,          NULL }        /* Terminator */
};

/* ------------------------------------------------------------------ */
/*  Process Command                                                   */
/* ------------------------------------------------------------------ */

/* strcmp helper */
static bool cmd_equals(const char *a, const char *b)
{
    while (*a && *b)
    {
        if (*a != *b) return false;
        a++; b++;
    }
    return (*a == *b);
}

void UART_ProcessCommand(char *cmdBuf)
{
    /* ---- Check no-argument command table first ---- */
    bool found = false;
    for (int i = 0; g_cmd_table[i].cmd != NULL; i++)
    {
        if (cmd_equals(cmdBuf, g_cmd_table[i].cmd))
        {
            g_cmd_table[i].handler();
            found = true;
            break;
        }
    }

    if (found) return;

#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
    /* ---- "tra=<label_name>" ---- */
    if (std::strncmp(cmdBuf, "tra=", 4) == 0)
    {
        const char *labelStr = cmdBuf + 4;
        /* Skip leading spaces */
        while (*labelStr == ' ') labelStr++;

        int foundIdx = -1;
        for (int i = 0; i < (int)labels.size(); i++) {
            if (labels[i] == labelStr) {
                foundIdx = i;
                break;
            }
        }

        if (foundIdx < 0) {
            info("[ZO] Unknown label '%s'. Valid labels:\r\n", labelStr);
            for (int i = 0; i < (int)labels.size(); i++) {
                info("  [%d] %s\r\n", i, labels[i].c_str());
            }
        } else {
            zoTargetLabel = foundIdx;
            ISM_SetState(ISM_ZO_TRAIN);
        }
        return;
    }

    /* ---- "zo_lr <value>" ---- */
    if (std::strncmp(cmdBuf, "zo_lr ", 6) == 0)
    {
        float val = (float)std::atof(cmdBuf + 6);
        if (val > 0.0f) {
            zoLearningRate = val;
            info("[ZO] Learning rate set to %.6f\r\n", zoLearningRate);
        } else {
            info("[ZO] Invalid learning rate (must be > 0)\r\n");
        }
        return;
    }

    /* ---- "zo_q <value>" ---- */
    if (std::strncmp(cmdBuf, "zo_q ", 5) == 0)
    {
        int val = std::atoi(cmdBuf + 5);
        if (val > 0) {
            zoNumPerturbations = val;
            info("[ZO] Perturbations Q set to %d\r\n", zoNumPerturbations);
        } else {
            info("[ZO] Invalid Q value (must be > 0)\r\n");
        }
        return;
    }

    /* ---- "zo_method np|wp" ----
     * Locked once a training run has started: a comparison run must stay on one
     * method. Switch is only accepted before the first step (or after zo_reset).
     * No command at all → default NP (zoMethod initialised to ZO_METHOD_NP). */
    if (std::strncmp(cmdBuf, "zo_method ", 10) == 0)
    {
        const char *methodStr = cmdBuf + 10;
        while (*methodStr == ' ') methodStr++; /* skip spaces */

        int requested;
        if (std::strcmp(methodStr, "np") == 0) {
            requested = ZO_METHOD_NP;
        } else if (std::strcmp(methodStr, "wp") == 0) {
            requested = ZO_METHOD_WP;
        } else {
            info("[ZO] Unknown method '%s'. Valid: np, wp\r\n", methodStr);
            return;
        }

        if (requested == zoMethod) {
            info("[ZO] Method already %s; unchanged\r\n",
                 (zoMethod == ZO_METHOD_WP) ? "WP" : "NP");
            return;
        }

        /* Lock: refuse to switch mid-run so a comparison stays single-method. */
        if (zoTrainer && zoTrainer->IsInitialized() && zoTrainer->GetStepCount() > 0) {
            info("[ZO] Cannot switch method after %d step(s). Send 'zo_reset' first.\r\n",
                 zoTrainer->GetStepCount());
            return;
        }

        zoMethod = requested;
        info("[ZO] Method set to %s\r\n", (zoMethod == ZO_METHOD_WP) ? "WP" : "NP");
        return;
    }
#endif /* USE_SPLIT_MODEL */

    /* ---- Log Control Commands ---- */
    /* log_all: enable all verbose logs */
    if (cmd_equals(cmdBuf, "log_all"))
    {
        LogConfig_EnableAll();
        info("[LOG] All debug tokens enabled\r\n");
        return;
    }

    /* log_none: disable all verbose logs */
    if (cmd_equals(cmdBuf, "log_none"))
    {
        LogConfig_DisableAll();
        info("[LOG] All debug tokens disabled\r\n");
        return;
    }

    /* log_status: print current log configuration */
    if (cmd_equals(cmdBuf, "log_status"))
    {
        LogConfig_Print();
        return;
    }

    /* log_on <token>: enable specific token */
    if (std::strncmp(cmdBuf, "log_on ", 7) == 0)
    {
        const char *tokenStr = cmdBuf + 7;
        while (*tokenStr == ' ') tokenStr++;  /* skip spaces */

        if (std::strcmp(tokenStr, "load") == 0) {
            LogConfig_Enable(LOG_MODEL_LOAD);
            info("[LOG] LOG_MODEL_LOAD enabled\r\n");
        } else if (std::strcmp(tokenStr, "init") == 0) {
            LogConfig_Enable(LOG_MODEL_INIT);
            info("[LOG] LOG_MODEL_INIT enabled\r\n");
        } else if (std::strcmp(tokenStr, "inf") == 0) {
            LogConfig_Enable(LOG_INFERENCE_DETAIL);
            info("[LOG] LOG_INFERENCE_DETAIL enabled\r\n");
        } else if (std::strcmp(tokenStr, "zo") == 0) {
            LogConfig_Enable(LOG_ZO_TRAINING);
            info("[LOG] LOG_ZO_TRAINING enabled\r\n");
        } else {
            info("[LOG] Unknown token: %s\r\n", tokenStr);
            info("[LOG] Valid tokens: load, init, inf, zo\r\n");
        }
        return;
    }

    /* log_off <token>: disable specific token */
    if (std::strncmp(cmdBuf, "log_off ", 8) == 0)
    {
        const char *tokenStr = cmdBuf + 8;
        while (*tokenStr == ' ') tokenStr++;  /* skip spaces */

        if (std::strcmp(tokenStr, "load") == 0) {
            LogConfig_Disable(LOG_MODEL_LOAD);
            info("[LOG] LOG_MODEL_LOAD disabled\r\n");
        } else if (std::strcmp(tokenStr, "init") == 0) {
            LogConfig_Disable(LOG_MODEL_INIT);
            info("[LOG] LOG_MODEL_INIT disabled\r\n");
        } else if (std::strcmp(tokenStr, "inf") == 0) {
            LogConfig_Disable(LOG_INFERENCE_DETAIL);
            info("[LOG] LOG_INFERENCE_DETAIL disabled\r\n");
        } else if (std::strcmp(tokenStr, "zo") == 0) {
            LogConfig_Disable(LOG_ZO_TRAINING);
            info("[LOG] LOG_ZO_TRAINING disabled\r\n");
        } else {
            info("[LOG] Unknown token: %s\r\n", tokenStr);
            info("[LOG] Valid tokens: load, init, inf, zo\r\n");
        }
        return;
    }

    info("[ERR] Unknown command: %s\r\n", cmdBuf);
}
