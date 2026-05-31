#include "LogConfig.hpp"
#include "log_macros.h"

/* ================================================================== */
/*  Global Log Configuration State                                   */
/* ================================================================== */

uint16_t g_logTokens = 0;  /* Default: all verbose logs disabled */

/* ================================================================== */
/*  Log Control Functions Implementation                              */
/* ================================================================== */

void LogConfig_Enable(LogToken token)
{
    g_logTokens |= token;
}

void LogConfig_Disable(LogToken token)
{
    g_logTokens &= ~token;
}

void LogConfig_EnableAll(void)
{
    g_logTokens = LOG_ALL_VERBOSE;
}

void LogConfig_DisableAll(void)
{
    g_logTokens = 0;
}

void LogConfig_Print(void)
{
    info("\r\n[LOG CONFIG] Current log tokens:\r\n");
    info("  LOG_MODEL_LOAD:       %s\r\n", (g_logTokens & LOG_MODEL_LOAD)      ? "ON" : "OFF");
    info("  LOG_MODEL_INIT:       %s\r\n", (g_logTokens & LOG_MODEL_INIT)      ? "ON" : "OFF");
    info("  LOG_HYPERRAM_TEST:    %s\r\n", (g_logTokens & LOG_HYPERRAM_TEST)   ? "ON" : "OFF");
    info("  LOG_INFERENCE_DETAIL: %s\r\n", (g_logTokens & LOG_INFERENCE_DETAIL)? "ON" : "OFF");
    info("  LOG_ZO_TRAINING:      %s\r\n", (g_logTokens & LOG_ZO_TRAINING)     ? "ON" : "OFF");
    info("\r\n");
}
