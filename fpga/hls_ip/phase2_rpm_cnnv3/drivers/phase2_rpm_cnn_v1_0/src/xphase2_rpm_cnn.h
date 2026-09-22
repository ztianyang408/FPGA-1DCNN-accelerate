// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.2 (64-bit)
// Tool Version Limit: 2024.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XPHASE2_RPM_CNN_H
#define XPHASE2_RPM_CNN_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xphase2_rpm_cnn_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Control_BaseAddress;
} XPhase2_rpm_cnn_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XPhase2_rpm_cnn;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XPhase2_rpm_cnn_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XPhase2_rpm_cnn_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XPhase2_rpm_cnn_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XPhase2_rpm_cnn_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XPhase2_rpm_cnn_Initialize(XPhase2_rpm_cnn *InstancePtr, UINTPTR BaseAddress);
XPhase2_rpm_cnn_Config* XPhase2_rpm_cnn_LookupConfig(UINTPTR BaseAddress);
#else
int XPhase2_rpm_cnn_Initialize(XPhase2_rpm_cnn *InstancePtr, u16 DeviceId);
XPhase2_rpm_cnn_Config* XPhase2_rpm_cnn_LookupConfig(u16 DeviceId);
#endif
int XPhase2_rpm_cnn_CfgInitialize(XPhase2_rpm_cnn *InstancePtr, XPhase2_rpm_cnn_Config *ConfigPtr);
#else
int XPhase2_rpm_cnn_Initialize(XPhase2_rpm_cnn *InstancePtr, const char* InstanceName);
int XPhase2_rpm_cnn_Release(XPhase2_rpm_cnn *InstancePtr);
#endif

void XPhase2_rpm_cnn_Start(XPhase2_rpm_cnn *InstancePtr);
u32 XPhase2_rpm_cnn_IsDone(XPhase2_rpm_cnn *InstancePtr);
u32 XPhase2_rpm_cnn_IsIdle(XPhase2_rpm_cnn *InstancePtr);
u32 XPhase2_rpm_cnn_IsReady(XPhase2_rpm_cnn *InstancePtr);
void XPhase2_rpm_cnn_EnableAutoRestart(XPhase2_rpm_cnn *InstancePtr);
void XPhase2_rpm_cnn_DisableAutoRestart(XPhase2_rpm_cnn *InstancePtr);

void XPhase2_rpm_cnn_Set_input_r(XPhase2_rpm_cnn *InstancePtr, u64 Data);
u64 XPhase2_rpm_cnn_Get_input_r(XPhase2_rpm_cnn *InstancePtr);
void XPhase2_rpm_cnn_Set_output_rpm(XPhase2_rpm_cnn *InstancePtr, u64 Data);
u64 XPhase2_rpm_cnn_Get_output_rpm(XPhase2_rpm_cnn *InstancePtr);

void XPhase2_rpm_cnn_InterruptGlobalEnable(XPhase2_rpm_cnn *InstancePtr);
void XPhase2_rpm_cnn_InterruptGlobalDisable(XPhase2_rpm_cnn *InstancePtr);
void XPhase2_rpm_cnn_InterruptEnable(XPhase2_rpm_cnn *InstancePtr, u32 Mask);
void XPhase2_rpm_cnn_InterruptDisable(XPhase2_rpm_cnn *InstancePtr, u32 Mask);
void XPhase2_rpm_cnn_InterruptClear(XPhase2_rpm_cnn *InstancePtr, u32 Mask);
u32 XPhase2_rpm_cnn_InterruptGetEnabled(XPhase2_rpm_cnn *InstancePtr);
u32 XPhase2_rpm_cnn_InterruptGetStatus(XPhase2_rpm_cnn *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
