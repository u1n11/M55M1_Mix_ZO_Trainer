#ifndef LOG_CONFIG_HPP
#define LOG_CONFIG_HPP

#include <cstdint>

/* ================================================================== */
/*  Log Control Tokens                                               */
/* ================================================================== */
/* These tokens control verbose logging for different subsystems.
 * Use them to selectively enable/disable detailed logs during development. */

enum LogToken : uint16_t {
    LOG_MODEL_LOAD     = 0x0001,  /* Detailed model loading info (arena, opcodes, quantization) */
    LOG_MODEL_INIT     = 0x0002,  /* Model initialization diagnostics */
    LOG_HYPERRAM_TEST  = 0x0004,  /* HyperRAM sanity test details */
    LOG_INFERENCE_DETAIL = 0x0008, /* Detailed inference per-layer times */
    LOG_ZO_TRAINING    = 0x0010,  /* ZO trainer initialization & step details */
    
    LOG_ALL_VERBOSE    = 0xFFFF,  /* Enable all debug logs */
};

/* ================================================================== */
/*  Global Log Configuration                                         */
/* ================================================================== */

extern uint16_t g_logTokens;  /* Bitmask of active log tokens */

/* Check if a specific log token is enabled */
inline bool IsLogTokenEnabled(LogToken token) {
    return (g_logTokens & token) != 0;
}

/* ================================================================== */
/*  Conditional Log Macros                                           */
/* ================================================================== */

/* Log only if specific token is enabled */
#define info_if_token(token, fmt, ...) \
    do { \
        if (IsLogTokenEnabled(token)) { \
            info(fmt, ##__VA_ARGS__); \
        } \
    } while(0)

/* Always log (critical info) */
#define info_critical(fmt, ...) \
    info(fmt, ##__VA_ARGS__)

/* ================================================================== */
/*  Token Control Functions                                          */
/* ================================================================== */

void LogConfig_Enable(LogToken token);
void LogConfig_Disable(LogToken token);
void LogConfig_EnableAll(void);
void LogConfig_DisableAll(void);
void LogConfig_Print(void);  /* Print current log configuration */

#endif /* LOG_CONFIG_HPP */
