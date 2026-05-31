#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "NuMicro.h"
#include "hyperflash_code.h"

#define HFLH_TRIM_SIZE              (0x1000)
#define HFLH_TRIM_ADDR              (0x800000 - 0x1000)

extern void SPIM_HYPER_AXQContMode(SPIM_T *spim, uint32_t u32SeqCNT);

//------------------------------------------------------------------------------
void HyperFlash_ResetModule(SPIM_T *spim)
{
    volatile uint32_t u32Delay = 0;

    SPIM_HYPER_Reset(spim);

    for (u32Delay = 0; u32Delay < 0x3000; u32Delay++) {}
}

void HyperFlash_ClearECCError(SPIM_T *spim)
{
    HyperFlash_WriteOPCMD(spim, HF_CMD_NOOP_CODE, HF_CMD_NOOP_CODE);
    //HyperFlash_WriteOPCMD(spim, HF_CMD_NOOP_CODE, HF_CMD_NOOP_CODE);
    //HyperFlash_WriteOPCMD(spim, HF_CMD_NOOP_CODE, HF_CMD_NOOP_CODE);
}

/**
 * @brief Send Hyper Flash Operation Command
 *
 * @param spim
 * @param u32CMD
 * @param u32CMDData
 */
void HyperFlash_WriteOPCMD(SPIM_T *spim, uint32_t u32CMD, uint32_t u32Addr)
{
    SPIM_HYPER_Write2Byte(spim, (u32CMD * 2), u32Addr);
}

void HyperFlash_WriteConfigRegister(SPIM_T *spim, uint32_t u32Reg, uint16_t u16WrData)
{

    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_COMMON_AA);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_2AA, HF_CMD_COMMON_55);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, u32Reg);

    SPIM_HYPER_Write2Byte(spim, HF_CMD_NOOP_CODE, u16WrData);

    //HyperFlash_WaitBusBusy(spim);
}

/**
 * @brief Read HyperFlash Non-volatile config register
 *
 * @param spim
 * @return uint16_t Register value
 */
uint16_t HyperFlash_ReadConfigRegister(SPIM_T *spim, uint32_t u32Reg)
{
    volatile uint16_t u16RdData = 0;

    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_COMMON_AA);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_2AA, HF_CMD_COMMON_55);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, u32Reg);

    u16RdData = SPIM_HYPER_Read1Word(spim, HF_CMD_NOOP_CODE);

    return u16RdData;
}

/**
 * @brief Wait Hyper Flash Program Busy
 *
 * @param spim
 */
int32_t HyperFlash_WaitBusBusy(SPIM_T *spim)
{
    volatile uint32_t u32Status = 0;

    while (u32Status != 0x80)
    {
        HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_70);

        u32Status = (SPIM_HYPER_Read1Word(spim, HF_CMD_NOOP_CODE) & 0x80);

        //if (u32Status == 0x80)
        //{
        //    log_printf("u32Status = %x\r\n", u32Status);
        //}

        //if (--i32Timeout <= 0)
        //{
        //    return SPIM_ERR_TIMEOUT;
        //}
    }


    return SPIM_OK;
}

void HyperFlash_EraseSector(SPIM_T *spim, uint32_t u32SAddr)
{

    if ((u32SAddr != 0) && (u32SAddr >= 2))
    {
        u32SAddr /= 2;
    }

    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_COMMON_AA);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_2AA, HF_CMD_COMMON_55);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_80);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_COMMON_AA);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_2AA, HF_CMD_COMMON_55);
    HyperFlash_WriteOPCMD(spim, u32SAddr, HF_CMD_30);

    HyperFlash_WaitBusBusy(spim);
}

void HyperFlash_EraseChip(SPIM_T *spim)
{
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_COMMON_AA);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_2AA, HF_CMD_COMMON_55);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_80);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_COMMON_AA);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_2AA, HF_CMD_COMMON_55);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_10);

    HyperFlash_WaitBusBusy(spim);
}

void do_dmm_writepage(SPIM_T *spim, uint32_t u32SAddr, uint8_t *pu8WrBuf, uint32_t u32NTx)
{
    int32_t *pi32Saddr = (int32_t *)u32SAddr;

    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_COMMON_AA);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_2AA, HF_CMD_COMMON_55);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_A0);

    SPIM_HYPER_EnterDirectMapMode(spim); // Hyper Mode Switch to Direct Map mode.

    memcpy(pi32Saddr, pu8WrBuf, u32NTx);

    SPIM_HYPER_WAIT_DMMDONE(spim);
    HyperFlash_WaitBusBusy(spim);
}

void HyperFlash_DMMWrite(SPIM_T *spim, uint32_t u32SAddr, uint8_t *pu8WrBuf, uint32_t u32NTx)
{
    uint32_t pageOffset = 0;

    pageOffset = u32SAddr % HFLH_PAGE_SIZE;

    if ((pageOffset + u32NTx) <= HFLH_PAGE_SIZE)
    {
        /* Do all the bytes fit onto one page ? */
        do_dmm_writepage(spim, u32SAddr, pu8WrBuf, u32NTx);
    }
    else
    {
        uint32_t toWr = 0;
        uint32_t buf_idx = 0;

        toWr = HFLH_PAGE_SIZE - pageOffset;               /* Size of data remaining on the first page. */

        do_dmm_writepage(spim, u32SAddr, &pu8WrBuf[buf_idx], toWr);

        u32SAddr += toWr;                         /* Advance indicator. */
        u32NTx -= toWr;
        buf_idx += toWr;

        while (u32NTx)
        {
            toWr = HFLH_PAGE_SIZE;

            if (toWr > u32NTx)
            {
                toWr = u32NTx;
            }

            do_dmm_writepage(spim, u32SAddr, &pu8WrBuf[buf_idx], toWr);

            u32SAddr += toWr;                 /* Advance indicator. */
            u32NTx -= toWr;
            buf_idx += toWr;
        }
    }
}

void HyperFlash_DMA_WriteByPage(SPIM_T *spim, uint32_t u32SAddr, uint8_t *pu8WrBuf, uint32_t u32NTx)
{

    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_COMMON_AA);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_2AA, HF_CMD_COMMON_55);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_A0);

    SPIM_HYPER_DMAWrite(spim, u32SAddr, pu8WrBuf, u32NTx);

    HyperFlash_WaitBusBusy(spim);
}

void HyperFlash_DMAWrite(SPIM_T *spim, uint32_t u32SAddr, uint8_t *pu8WrBuf, uint32_t u32NTx)
{
    uint32_t pageOffset = 0;

    pageOffset = u32SAddr % HFLH_PAGE_SIZE;

    if ((pageOffset + u32NTx) <= HFLH_PAGE_SIZE)
    {
        /* Do all the bytes fit onto one page ? */
        HyperFlash_DMA_WriteByPage(spim, u32SAddr, pu8WrBuf, u32NTx);
    }
    else
    {
        uint32_t toWr = 0;
        uint32_t buf_idx = 0;

        toWr = HFLH_PAGE_SIZE - pageOffset;               /* Size of data remaining on the first page. */

        HyperFlash_DMA_WriteByPage(spim, u32SAddr, &pu8WrBuf[buf_idx], toWr);

        u32SAddr += toWr;                         /* Advance indicator. */
        u32NTx -= toWr;
        buf_idx += toWr;

        while (u32NTx)
        {
            toWr = HFLH_PAGE_SIZE;

            if (toWr > u32NTx)
            {
                toWr = u32NTx;
            }

            HyperFlash_DMA_WriteByPage(spim, u32SAddr, &pu8WrBuf[buf_idx], toWr);

            u32SAddr += toWr;                 /* Advance indicator. */
            u32NTx -= toWr;
            buf_idx += toWr;
        }
    }
}

void HyperFlash_DMARead(SPIM_T *spim, uint32_t u32SAddr, uint8_t *pu8RdBuf, uint32_t u32NRx)
{
    SPIM_HYPER_DMARead(spim, u32SAddr, pu8RdBuf, u32NRx);

    HyperFlash_WaitBusBusy(spim);
}

void HyperFlash_IO_Write2Byte(SPIM_T *spim, uint32_t u32WrAddr, uint16_t u16WrData)
{

    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_COMMON_AA);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_2AA, HF_CMD_COMMON_55);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_A0);

    SPIM_HYPER_Write2Byte(spim, u32WrAddr, u16WrData);

    HyperFlash_WaitBusBusy(spim);
}

void HyperFlash_IO_Write4Byte(SPIM_T *spim, uint32_t u32WrAddr, uint32_t u32WrData)
{

    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_COMMON_AA);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_2AA, HF_CMD_COMMON_55);
    HyperFlash_WriteOPCMD(spim, HF_CMD_COMMON_555, HF_CMD_A0);

    SPIM_HYPER_Write4Byte(spim, u32WrAddr, u32WrData);

    HyperFlash_WaitBusBusy(spim);
}

uint16_t HyperFlash_IO_Read2Byte(SPIM_T *spim, uint32_t u32SAddr)
{
    volatile uint16_t u16RdData = 0;


    u16RdData = SPIM_HYPER_Read1Word(spim, u32SAddr);

    return u16RdData;
}

uint32_t HyperFlash_IO_Read4Byte(SPIM_T *spim, uint32_t u32SAddr)
{
    volatile uint32_t u32RdData = 0;


    u32RdData = SPIM_HYPER_Read2Word(spim, u32SAddr);

    return u32RdData;
}

void HyperFlash_IO_Read(SPIM_T *spim, uint32_t u32SAddr, void *pvRdBuf, uint32_t u32NRx)
{
    uint8_t *pu8RxBuf = NULL;
    uint32_t u32DataCnt = 0;
    uint16_t *pu16RxBuf = (uint16_t *)pvRdBuf;
    volatile uint32_t u32i = 0;
    uint32_t u32RemainSize = (u32NRx % 2);

    for (u32i = 0; u32i < (u32NRx - u32RemainSize); u32i += 2)
    {
        pu16RxBuf[u32DataCnt++] = HyperFlash_IO_Read2Byte(spim, u32SAddr + u32i);
    }

    if (u32RemainSize != 0)
    {
        uint8_t *pu8Temp = (void *)pvRdBuf;
        pu8RxBuf = (uint8_t *)&pu8Temp[(u32NRx - u32RemainSize)];
        *pu8RxBuf = ((HyperFlash_IO_Read2Byte(spim, ((u32SAddr + u32NRx) - u32RemainSize)) >> 8) & 0xFF);
    }
}

#define DMM_MODE_TRIM

/**
 * @brief Training DLL component delay stop number
 *
 * @param spim
 */
void HyperFlash_TrainingDLLDelayTime(SPIM_T *spim)
{
    volatile uint8_t u8RdDelay = 0;
    uint8_t u8RdDelayIdx = 0;
    uint8_t u8RdDelayRes[SPIM_HYPER_MAX_LATENCY] = {0};
    uint32_t u32SrcAddr = HFLH_TRIM_ADDR;
    uint32_t u32TestSize = 32;
    volatile uint32_t u32i = 0;
    uint8_t au8TrimPatten[32] =
    {
        0xff, 0x0F, 0xFF, 0x00, 0xFF, 0xCC, 0xC3, 0xCC,
        0xC3, 0x3C, 0xCC, 0xFF, 0xFE, 0xFF, 0xFE, 0xEF,
        0xFF, 0xDF, 0xFF, 0xDD, 0xFF, 0xFB, 0xFF, 0xFB,
        0xBF, 0xFF, 0x7F, 0xFF, 0x77, 0xF7, 0xBD, 0xEF,
    };
    uint8_t au8DestBuf[32] = {0};
#ifdef DMM_MODE_TRIM
    uint32_t u32RdDataCnt = 0;
    uint32_t *pu32RdBuf = NULL;
#endif
    //popDat(au8TrimPatten, u32TestSize);

    SPIM_HYPER_INIT_DLL(spim);

    /* Erase HyperFlash */
    HyperFlash_EraseSector(spim, u32SrcAddr); //one sector = 256KB

    HyperFlash_DMAWrite(spim, u32SrcAddr, au8TrimPatten, u32TestSize);

    //for (u8RdDelay = 0; u8RdDelay < u32TestSize; u8RdDelay += 4)
    //{
    //    memcpy(&u32WrData, &au8TrimPatten[u8RdDelay], sizeof(uint32_t));
    //    HyperFlash_IO_Write4Byte(spim, u32SrcAddr + u8RdDelay, u32WrData);
    //}

#ifdef DMM_MODE_TRIM
    /* Enter direct-mapped mode to run new applications */
    //SPIM_HYPER_EnterDirectMapMode(spim);
    pu32RdBuf = (uint32_t *)au8DestBuf;
#endif

    for (u8RdDelay = 0; u8RdDelay <= SPIM_HYPER_MAX_LATENCY; u8RdDelay++)
    {
        /* Set DLL calibration to select the valid delay step number */
        SPIM_HYPER_SetDLLDelayNum(spim, u8RdDelay);

        memset(au8DestBuf, 0, sizeof(au8DestBuf));

#ifndef DMM_MODE_TRIM
        /* Read Data from HyperFlash */
        HyperFlash_DMARead(spim, u32SrcAddr, tstbuf2, u32TestSize);
#else
        u32RdDataCnt = 0;

        //for (u32i = u32SrcAddr; u32i < (u32SrcAddr + u32TestSize); u32i += 4)
        //{
        //    pu32RdBuf[u32RdDataCnt++] = inpw(u32DMMAddr + u32i);
        //}

        for (u32i = 0; u32i < u32TestSize; u32i += 4)
        {
            pu32RdBuf[u32RdDataCnt++] = HyperFlash_IO_Read4Byte(spim, u32SrcAddr + u32i);
        }

#endif//

        /* Verify the data and save the number of successful delay steps */
        if (memcmp(au8TrimPatten, au8DestBuf, u32TestSize))
        {
            printf("Data compare failed at block 0x%x\n", u32SrcAddr);
            //dump_compare_error(u32SrcAddr, au8TrimPatten, au8DestBuf, u32TestSize);
        }
        else
        {
            printf("RX Delay: %d = Pass\r\n", u8RdDelay);
            u8RdDelayRes[u8RdDelayIdx++] = u8RdDelay;
        }
    }

    //HyperFlash_WaitBusBusy(spim);

    if (u8RdDelayIdx <= 1)
    {
        u8RdDelayIdx = 0;
        u8RdDelayRes[u8RdDelayIdx] = 12;
    }
    else
    {
        if (u8RdDelayIdx >= 2)
        {
            u8RdDelayIdx = ((u8RdDelayIdx / 2)) - 1;
        }
        else
        {
            u8RdDelayIdx = 0;
            u8RdDelayRes[u8RdDelayIdx] = 12;
        }
    }

    /* Set the number of intermediate delay steps */
    SPIM_HYPER_SetDLLDelayNum(spim, u8RdDelayRes[u8RdDelayIdx]);
    printf("Set Delay Step Num : %d\r\n", u8RdDelayRes[u8RdDelayIdx]);
}

/**
  * @brief      SPIM Default Config HyperBus Access Module Parameters.
  * @param      spim
  * @param      u32CSMaxLT Chip Select Maximum Low Time 0 ~ 0xFFFF, Default Set 0x02ED
  * @param      u32AcctRD Initial Read Access Time 1 ~ 0x1F, Default Set 0x04
  * @param      u32AcctWR Initial Write Access Time 1 ~ 0x1F, Default Set 0x04
  * @return     None.
  */
void SPIM_HyperFlash_DefaultConfig(SPIM_T *spim, uint32_t u32CSMaxLow,
                                   uint32_t u32AcctRD, uint32_t u32AcctWR)
{
    /* Chip Select Setup Time 2.5 */
    SPIM_HYPER_SET_CSST(spim, SPIM_HYPER_CSST_3_5_HCLK);

    /* Chip Select Hold Time 3.5 HCLK */
    SPIM_HYPER_SET_CSH(spim, SPIM_HYPER_CSH_3_5_HCLK);

    /* Chip Select High between Transaction as 2 HCLK cycles */
    SPIM_HYPER_SET_CSHI(spim, 2);

    /* Chip Select Masximum low time HCLK */
    SPIM_HYPER_SET_CSMAXLT(spim, u32CSMaxLow);

    /* Initial Device RESETN Low Time 255 */
    SPIM_HYPER_SET_RSTNLT(spim, 0xFF);

    /* Initial Read Access Time Clock cycle*/
    SPIM_HYPER_SET_ACCTRD(spim, u32AcctRD);

    /* Initial Write Access Time Clock cycle*/
    SPIM_HYPER_SET_ACCTWR(spim, u32AcctWR);
}

static uint32_t HyperFlash_GetLatencyNum(uint32_t u32Latency)
{
    if (u32Latency < 5)
    {
        u32Latency = 5;
    }
    else if (u32Latency > 16)
    {
        u32Latency = 16;
    }

    return (u32Latency - 5);
}

void HyperFlash_SetReadLatency(SPIM_T *spim, uint32_t u32Latency)
{
    uint32_t u32RegValue = 0;
    uint32_t u32VCRValue = 0x8E0B;

    /* HyperFlash default read latency is 16 and write is always 1 */
    SPIM_HyperFlash_DefaultConfig(spim, HFLH_MAX_CS_LOW, HFLH_DEFRD_LTCY_NUM, HFLH_WR_LTCY_NUM);

    SPIM_HYPER_SetDLLDelayNum(spim, 2);

    HyperFlash_ResetModule(spim);

    //u32RegValue = HyperFlash_ReadConfigRegister(spim, READ_VCR_REG);
    //log_printf("1 VCReg = %x\r\n", u32RegValue);

    u32VCRValue |= (HyperFlash_GetLatencyNum(u32Latency) << 4);

    HyperFlash_WriteConfigRegister(spim, LOAD_VCR_REG, u32VCRValue);

    SPIM_HyperFlash_DefaultConfig(spim, HFLH_MAX_CS_LOW, u32Latency, HFLH_WR_LTCY_NUM);

    //u32RegValue = HyperFlash_ReadConfigRegister(spim, READ_VCR_REG);
    //log_printf("2 VCReg = %x\r\n", u32RegValue);

    HyperFlash_TrainingDLLDelayTime(spim);

    u32RegValue = HyperFlash_ReadConfigRegister(spim, READ_VCR_REG);
    printf("2 VCReg = %x\r\n", u32RegValue);
}

void SPIM_NVIC_Disable(SPIM_T *pSPIMx)
{
    if (pSPIMx == SPIM0)
    {
        NVIC_DisableIRQ(SPIM0_IRQn);
    }
}

void InitSPIMPort(SPIM_T *pSPIMx)
{
    if (pSPIMx == SPIM0)
    {
        /* Enable SPIM0 module clock */
        CLK_EnableModuleClock(SPIM0_MODULE);
        /* Enable OTFC0 module clock */
        CLK_EnableModuleClock(OTFC0_MODULE);

        /* Set the slew rate of HyperRAM IO pins to FAST1 by default */
        /* This default setting is based on Infineon S27KS0642 and Winbond W958D8NBYA. */
        /*
         * If you encounter issues such as signal instability, timing errors, or read/write failures during testing,
         * you may refer to the HyperRAM datasheet and adjust this setting accordingly.
        */
        uint32_t u32SlewRate = GPIO_SLEWCTL_FAST1;

        /* Init SPIM multi-function pins */
        SET_SPIM0_CLKN_PH12();
        SET_SPIM0_CLK_PH13();
        SET_SPIM0_D2_PJ5();
        SET_SPIM0_D3_PJ6();
        SET_SPIM0_D4_PH14();
        SET_SPIM0_D5_PH15();
        SET_SPIM0_D6_PG13();
        SET_SPIM0_D7_PG14();
        SET_SPIM0_MISO_PJ4();
        SET_SPIM0_MOSI_PJ3();
        SET_SPIM0_RESETN_PJ2();
        SET_SPIM0_RWDS_PG15();
        SET_SPIM0_SS_PJ7();

        PG->SMTEN |= (GPIO_SMTEN_SMTEN13_Msk |
                      GPIO_SMTEN_SMTEN14_Msk |
                      GPIO_SMTEN_SMTEN15_Msk);
        PH->SMTEN |= (GPIO_SMTEN_SMTEN12_Msk |
                      GPIO_SMTEN_SMTEN13_Msk |
                      GPIO_SMTEN_SMTEN14_Msk |
                      GPIO_SMTEN_SMTEN15_Msk);
        PJ->SMTEN |= (GPIO_SMTEN_SMTEN2_Msk |
                      GPIO_SMTEN_SMTEN3_Msk |
                      GPIO_SMTEN_SMTEN4_Msk |
                      GPIO_SMTEN_SMTEN5_Msk |
                      GPIO_SMTEN_SMTEN6_Msk |
                      GPIO_SMTEN_SMTEN7_Msk);

        /* Set SPIM I/O pins as high slew rate up to 80 MHz. */
        GPIO_SetSlewCtl(PG, BIT13, u32SlewRate);
        GPIO_SetSlewCtl(PG, BIT14, u32SlewRate);
        GPIO_SetSlewCtl(PG, BIT15, u32SlewRate);

        GPIO_SetSlewCtl(PH, BIT12, u32SlewRate);
        GPIO_SetSlewCtl(PH, BIT13, u32SlewRate);
        GPIO_SetSlewCtl(PH, BIT14, u32SlewRate);
        GPIO_SetSlewCtl(PH, BIT15, u32SlewRate);

        GPIO_SetSlewCtl(PJ, BIT2, u32SlewRate);
        GPIO_SetSlewCtl(PJ, BIT3, u32SlewRate);
        GPIO_SetSlewCtl(PJ, BIT4, u32SlewRate);
        GPIO_SetSlewCtl(PJ, BIT5, u32SlewRate);
        GPIO_SetSlewCtl(PJ, BIT6, u32SlewRate);
        GPIO_SetSlewCtl(PJ, BIT7, u32SlewRate);
    }
}

void SPIM_HyperFlash_Init(SPIM_T *pSPIMx)
{

    SPIM_NVIC_Disable(pSPIMx);

    InitSPIMPort(pSPIMx);

    SPIM_HYPER_Init(pSPIMx, SPIM_HYPERFLASH_MODE, 1); // Enable HyperBus Mode


    SPIM_HyperFlash_DefaultConfig(pSPIMx, HFLH_MAX_CS_LOW, HFLH_DEFRD_LTCY_NUM, HFLH_WR_LTCY_NUM);
    HyperFlash_ResetModule(pSPIMx);
    SPIM_HYPER_SetDLLDelayNum(pSPIMx, 1);
    SPIM_HYPER_INIT_DLL(pSPIMx);

    HyperFlash_TrainingDLLDelayTime(pSPIMx);
    //HyperFlash_SetReadLatency(pSPIMx, 10);
    //Must set DLL directly, because call HyperFlash_SetReadLatency() will erase hyperflash.
    //SPIM_HYPER_SetDLLDelayNum(pSPIMx, 5);

    SPIM_HYPER_EnterDirectMapMode(pSPIMx);
}

/*---------------------------------------------------------------------------*/
/*  HyperRAM Initialisation                                                  */
/*---------------------------------------------------------------------------*/

/* HyperRAM timing constants (Winbond W958D8NBYA / Infineon S27KS0642) */
#define HYPERRAM_CSM_TIME           4000    /* tCSM in nanoseconds */
#define HYPERRAM_RD_LTCY            7       /* Initial read latency (clock cycles) */
#define HYPERRAM_WR_LTCY            7       /* Initial write latency (clock cycles) */
#define HYPERRAM_CSHI_CYCLE         4       /* CS High time between transactions */
#define HYPERRAM_RST_CNT            0xFF    /* RESETN low time */
#define CSMAXLT_CIPHER_OFF          21      /* CSMAXLT overhead when cipher is off */
#define CSMAXLT_CIPHER_ON           54      /* CSMAXLT overhead when cipher is on */

/* DLL trimming parameters */
#define HRAM_TRIM_SAFE_OFFSET       0x10000 /* Trim at safe offset (not addr 0) */
#define HRAM_TRIM_SIZE              512     /* Must be divisible by 8 */
#define DLL_TRIM_MAX_RETRY          10
#define DLL_TRIM_PASS_COUNT         3
#define DLL_TRIM_SCORE_PER_BLOCK    8
#define DLL_TRIM_JUMP_STEP_1        0x08
#define DLL_TRIM_JUMP_STEP_2        0x10
#define DLL_TRIM_DEF_NUM            0x07
#define DMM_VERIFY_MAX_RETRY        3

/* HyperRAM register structure (for CONFIG0 drive strength) */
typedef struct
{
    union
    {
        uint32_t u32REG;
        struct
        {
            uint32_t u2BurstLength         : 2;
            uint32_t u1HybridBurstEnable   : 1;
            uint32_t u1FixedLatencyEnable  : 1;
            uint32_t u4InitialLatency      : 4;
            uint32_t u4Reserved            : 4;
            uint32_t u3DriveStrength       : 3;
            uint32_t u1DeepPowerDownEnable : 1;
            uint32_t                       : 16;
        };
    } CONFIG0;
} HRAM_REG_T;

/**
 * @brief  Compute CSMAXLT from tCSM (ns), system clock, and cipher state,
 *         then program all HyperBus timing registers for HyperRAM.
 */
static void SPIM_HyperRAM_DefaultConfig(SPIM_T *spim,
                                        uint32_t u32CSM,
                                        uint32_t u32AcctRD,
                                        uint32_t u32AcctWR)
{
    uint32_t u32CoreFreq  = CLK_GetSCLKFreq() / 1000000;   /* MHz */
    float    fPeriod_ns   = 1000.0f / (float)u32CoreFreq;  /* ns per cycle */
    uint32_t u32DIV       = SPIM_HYPER_GET_CLKDIV(spim);
    uint32_t u32CipherEn  = SPIM_HYPER_GET_CIPHER(spim);
    uint32_t u32CSMAXLT   = (uint32_t)((u32CSM / fPeriod_ns)
                            - (2U * 8U * u32DIV)
                            - ((!u32CipherEn == SPIM_HYPER_OP_ENABLE)
                               ? CSMAXLT_CIPHER_ON : CSMAXLT_CIPHER_OFF));

    SPIM_HYPER_SET_CSST(spim,    SPIM_HYPER_CSST_3_5_HCLK);
    SPIM_HYPER_SET_CSH(spim,     SPIM_HYPER_CSH_3_5_HCLK);
    SPIM_HYPER_SET_CSHI(spim,    HYPERRAM_CSHI_CYCLE);
    SPIM_HYPER_SET_CSMAXLT(spim, u32CSMAXLT);
    SPIM_HYPER_SET_RSTNLT(spim,  HYPERRAM_RST_CNT);
    SPIM_HYPER_SET_ACCTRD(spim,  u32AcctRD);
    SPIM_HYPER_SET_ACCTWR(spim,  u32AcctWR);
}

/* --- DLL Trimming helper patterns --- */
static void GenPRBS7Pattern(uint8_t *buf, uint32_t size)
{
    uint8_t prbs = 0x5A;
    for (uint32_t i = 0; i < size; i++)
    {
        buf[i] = prbs;
        prbs = (prbs >> 1) ^ ((prbs & 1) ? 0xB8 : 0x00);
    }
}

static void GenWalking1sPattern(uint8_t *buf, uint32_t size)
{
    for (uint32_t i = 0; i < size; i++)
        buf[i] = (uint8_t)(1u << (i % 8));
}

static void GenOriginalPattern(uint8_t *buf, uint32_t size)
{
    for (uint32_t k = 0; k < size; k++)
    {
        uint32_t val = (k & 0x0F) ^ (k >> 4) ^ (k >> 3);
        if (k & 1) val = ~val;
        buf[k] = ~(uint8_t)(val ^ (k << 3) ^ (k >> 2));
    }
}

static int VerifyFinalRead_RAM(SPIM_T *spim, uint32_t dmmAddr,
                               uint8_t *expected, uint32_t size)
{
    SPIM_HYPER_EnterDirectMapMode(spim);

    for (uint32_t i = 0; i + 8 <= size; i += 8)
    {
        __IO uint64_t *p = (__IO uint64_t *)(dmmAddr + i);
        uint64_t val = *p;
        if (memcmp(&expected[i], &val, 8) != 0)
        {
            SPIM_HYPER_ExitDirectMapMode(spim);
            return 0;
        }
    }

    SPIM_HYPER_ExitDirectMapMode(spim);
    return 1;
}

/**
 * @brief  Robust DLL delay trimming for HyperRAM.
 *         Uses three patterns and a scoring scheme to find the
 *         best DLL delay step, then verifies via DMM.
 */
static void HyperRAM_TrimDLLDelayNumber(SPIM_T *spim)
{
    uint8_t  u8RdDelay = 0;
    uint16_t u16Score[SPIM_HYPER_MAX_LATENCY] = {0};
    uint32_t u32SrcAddr = HRAM_TRIM_SAFE_OFFSET;
    uint32_t u32DMMAddr = SPIM_HYPER_GET_DMMADDR(spim);

    /* 8-byte-aligned buffers (DMA requirement) */
    uint64_t au64Pattern[HRAM_TRIM_SIZE / 8] = {0};
    uint64_t au64Verify [HRAM_TRIM_SIZE / 8] = {0};
    uint8_t *pu8Pattern = (uint8_t *)au64Pattern;
    uint8_t *pu8Verify  = (uint8_t *)au64Verify;

    void (*generators[])(uint8_t *, uint32_t) =
    {
        GenPRBS7Pattern, GenWalking1sPattern, GenOriginalPattern
    };
    const uint32_t u32NumPat = sizeof(generators) / sizeof(generators[0]);

    /* Phase 1: score each DLL delay across all patterns */
    for (uint32_t pi = 0; pi < u32NumPat; pi++)
    {
        generators[pi](pu8Pattern, HRAM_TRIM_SIZE);
        SPIM_HYPER_DMAWrite(spim, u32SrcAddr, pu8Pattern, HRAM_TRIM_SIZE);

        for (uint32_t retry = 0; retry < DLL_TRIM_MAX_RETRY; retry++)
        {
            for (u8RdDelay = 0; u8RdDelay < SPIM_HYPER_MAX_LATENCY; u8RdDelay++)
            {
                SPIM_HYPER_SetDLLDelayNum(spim, u8RdDelay);
                memset(pu8Verify, 0, HRAM_TRIM_SIZE);

                for (uint32_t pass = 0; pass < DLL_TRIM_PASS_COUNT; pass++)
                {
                    uint32_t loopAddr = pass * 0x100;
                    uint32_t ri = 0;

                    for (; ri + 8 <= HRAM_TRIM_SIZE; )
                    {
                        if (loopAddr + 8 > HRAM_TRIM_SIZE) break;
                        SPIM_HYPER_DMARead(spim, u32SrcAddr + loopAddr, &pu8Verify[ri], 8);
                        if (memcmp(&pu8Pattern[loopAddr], &pu8Verify[ri], 8) == 0)
                            u16Score[u8RdDelay] += DLL_TRIM_SCORE_PER_BLOCK;
                        loopAddr += ((ri % 3) == 0) ? DLL_TRIM_JUMP_STEP_1 : DLL_TRIM_JUMP_STEP_2;
                        ri += 8;
                    }
                }
            }
        }
    }

    /* Find best contiguous region */
    uint16_t maxScore = 0;
    for (uint32_t i = 0; i < SPIM_HYPER_MAX_LATENCY; i++)
        if (u16Score[i] > maxScore) maxScore = u16Score[i];

    uint8_t bestStart = 0, bestLen = 0, curLen = 0, curStart = 0;
    for (uint32_t i = 0; i < SPIM_HYPER_MAX_LATENCY; i++)
    {
        if (u16Score[i] == maxScore)
        {
            if (curLen == 0) curStart = (uint8_t)i;
            curLen++;
        }
        else
        {
            if (curLen > bestLen) { bestLen = curLen; bestStart = curStart; }
            curLen = 0;
        }
    }
    if (curLen > bestLen) { bestLen = curLen; bestStart = curStart; }

    /* Phase 2: DMM verify the best-score range */
    /* Use last pattern written for verification */
    generators[u32NumPat - 1](pu8Pattern, HRAM_TRIM_SIZE);
    SPIM_HYPER_DMAWrite(spim, u32SrcAddr, pu8Pattern, HRAM_TRIM_SIZE);

    uint8_t verifiedList[SPIM_HYPER_MAX_LATENCY] = {0};
    uint8_t verifiedCount = 0;

    for (uint32_t i = bestStart; i < (uint32_t)(bestStart + bestLen); i++)
    {
        if (SPIM_HYPER_SetDLLDelayNum(spim, (uint8_t)i) != SPIM_HYPER_OK) continue;
        if (SPIM_HYPER_GET_DLLREADY(spim) != SPIM_HYPER_OP_ENABLE) continue;

        for (int r = 0; r < DMM_VERIFY_MAX_RETRY; r++)
        {
            if (VerifyFinalRead_RAM(spim, u32DMMAddr + u32SrcAddr,
                                    pu8Pattern, HRAM_TRIM_SIZE))
            {
                verifiedList[verifiedCount++] = (uint8_t)i;
                break;
            }
        }
    }

    if (verifiedCount > 0)
    {
        u8RdDelay = verifiedList[verifiedCount / 2];
    }
    else
    {
        u8RdDelay = bestStart + (bestLen / 2);
    }

    /* Fallback backup: try u8RdDelay, -1, -2 */
    uint8_t backupTry[3] = { u8RdDelay,
                              (uint8_t)(u8RdDelay - 1),
                              (uint8_t)(u8RdDelay - 2) };

    for (uint32_t i = 0; i < 3; i++)
    {
        if (SPIM_HYPER_SetDLLDelayNum(spim, backupTry[i]) != SPIM_HYPER_OK) continue;
        if (SPIM_HYPER_GET_DLLREADY(spim) != SPIM_HYPER_OP_ENABLE) continue;

        for (int r = 0; r < DMM_VERIFY_MAX_RETRY; r++)
        {
            if (VerifyFinalRead_RAM(spim, u32DMMAddr + u32SrcAddr,
                                    pu8Pattern, HRAM_TRIM_SIZE))
            {
                printf("HyperRAM DLL Delay: %d\r\n", backupTry[i]);
                SPIM_HYPER_SetDLLDelayNum(spim, backupTry[i]);
                return;
            }
        }
    }

    printf("HyperRAM DLL Delay fallback: %d\r\n", DLL_TRIM_DEF_NUM);
    SPIM_HYPER_SetDLLDelayNum(spim, DLL_TRIM_DEF_NUM);
}

/**
 * @brief  Initialise SPIM for HyperRAM in Direct-Mapped Mode.
 *         After this call the HyperRAM is accessible at 0x82000000.
 */
void SPIM_HyperRAM_Init(SPIM_T *pSPIMx)
{
    HRAM_REG_T sHRAMReg;

    SPIM_NVIC_Disable(pSPIMx);
    InitSPIMPort(pSPIMx);

    /* Enable HyperRAM mode, clock divider = 1 */
    SPIM_HYPER_Init(pSPIMx, SPIM_HYPERRAM_MODE, 1);

    /* Configure timing: tCSM=4000ns, RD/WR latency=7 (device default) */
    SPIM_HyperRAM_DefaultConfig(pSPIMx, HYPERRAM_CSM_TIME, HYPERRAM_RD_LTCY, HYPERRAM_WR_LTCY);

    /* Reset HyperRAM device */
    SPIM_HYPER_Reset(pSPIMx);

    /* DLL trimming (uses DMA — must NOT be in DMM mode here) */
    HyperRAM_TrimDLLDelayNumber(pSPIMx);

    /* Read CONFIG0, set drive strength to 34 ohm (0), write back */
    sHRAMReg.CONFIG0.u32REG = SPIM_HYPER_ReadHyperRAMReg(pSPIMx, SPIM_HYPER_HRAM_CONFIG_REG0);
    sHRAMReg.CONFIG0.u3DriveStrength = 0;
    SPIM_HYPER_WriteHyperRAMReg(pSPIMx, SPIM_HYPER_HRAM_CONFIG_REG0, sHRAMReg.CONFIG0.u32REG);

    /* Enter Direct-Map Mode — HyperRAM now visible at 0x82000000 */
    SPIM_HYPER_EnterDirectMapMode(pSPIMx);

    printf("HyperRAM initialised at 0x%08X\r\n", (unsigned int)SPIM_HYPER_GET_DMMADDR(pSPIMx));
}
