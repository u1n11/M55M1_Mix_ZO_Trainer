#ifndef INFERENCE_MNGT_H
#define INFERENCE_MNGT_H

#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    ISM_IDLE,
    ISM_LOAD_MODEL,
    ISM_INFERENCE,
    ISM_DUMP_GRAPH,
#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
    ISM_LOAD_SPLIT_MODEL,
    ISM_SPLIT_INFERENCE,
    ISM_ZO_INIT,
    ISM_ZO_TRAIN,
#endif
    ISM_UNKNOWN
} ISM_State_t;

/* Initialize the Inference State Machine */
void ISM_Init(void);

/* Set the next state request (e.g. from UART command) */
void ISM_SetState(ISM_State_t newState);

/* Main processing function to be called in the main loop */
void ISM_Process(void);

/* Get current state */
ISM_State_t ISM_GetState(void);

#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
/* Persist current trained classifier FC parameters to APROM flash. */
int ISM_ZO_SaveToFlash(void);

/* Clear persisted classifier snapshot from APROM flash. */
int ISM_ZO_ClearFlash(void);

/* Print training status including train step count and model source. */
void ISM_ZO_PrintStatus(void);
#endif

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_MNGT_H */
