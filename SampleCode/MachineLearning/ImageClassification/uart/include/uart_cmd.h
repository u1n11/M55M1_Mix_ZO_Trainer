#ifndef UART_CMD_H
#define UART_CMD_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Process a received command string */
void UART_ProcessCommand(char *cmdBuf);

#ifdef __cplusplus
}
#endif

#endif /* UART_CMD_H */
