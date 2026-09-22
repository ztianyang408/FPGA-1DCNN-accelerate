// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.2 (64-bit)
// Tool Version Limit: 2024.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xphase2_rpm_cnn.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XPhase2_rpm_cnn_CfgInitialize(XPhase2_rpm_cnn *InstancePtr, XPhase2_rpm_cnn_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XPhase2_rpm_cnn_Start(XPhase2_rpm_cnn *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPhase2_rpm_cnn_ReadReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_AP_CTRL) & 0x80;
    XPhase2_rpm_cnn_WriteReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XPhase2_rpm_cnn_IsDone(XPhase2_rpm_cnn *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPhase2_rpm_cnn_ReadReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XPhase2_rpm_cnn_IsIdle(XPhase2_rpm_cnn *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPhase2_rpm_cnn_ReadReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XPhase2_rpm_cnn_IsReady(XPhase2_rpm_cnn *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPhase2_rpm_cnn_ReadReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XPhase2_rpm_cnn_EnableAutoRestart(XPhase2_rpm_cnn *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPhase2_rpm_cnn_WriteReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XPhase2_rpm_cnn_DisableAutoRestart(XPhase2_rpm_cnn *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPhase2_rpm_cnn_WriteReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_AP_CTRL, 0);
}

void XPhase2_rpm_cnn_Set_input_r(XPhase2_rpm_cnn *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPhase2_rpm_cnn_WriteReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_INPUT_R_DATA, (u32)(Data));
    XPhase2_rpm_cnn_WriteReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_INPUT_R_DATA + 4, (u32)(Data >> 32));
}

u64 XPhase2_rpm_cnn_Get_input_r(XPhase2_rpm_cnn *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPhase2_rpm_cnn_ReadReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_INPUT_R_DATA);
    Data += (u64)XPhase2_rpm_cnn_ReadReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_INPUT_R_DATA + 4) << 32;
    return Data;
}

void XPhase2_rpm_cnn_Set_output_rpm(XPhase2_rpm_cnn *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPhase2_rpm_cnn_WriteReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_OUTPUT_RPM_DATA, (u32)(Data));
    XPhase2_rpm_cnn_WriteReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_OUTPUT_RPM_DATA + 4, (u32)(Data >> 32));
}

u64 XPhase2_rpm_cnn_Get_output_rpm(XPhase2_rpm_cnn *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPhase2_rpm_cnn_ReadReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_OUTPUT_RPM_DATA);
    Data += (u64)XPhase2_rpm_cnn_ReadReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_OUTPUT_RPM_DATA + 4) << 32;
    return Data;
}

void XPhase2_rpm_cnn_InterruptGlobalEnable(XPhase2_rpm_cnn *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPhase2_rpm_cnn_WriteReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_GIE, 1);
}

void XPhase2_rpm_cnn_InterruptGlobalDisable(XPhase2_rpm_cnn *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPhase2_rpm_cnn_WriteReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_GIE, 0);
}

void XPhase2_rpm_cnn_InterruptEnable(XPhase2_rpm_cnn *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XPhase2_rpm_cnn_ReadReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_IER);
    XPhase2_rpm_cnn_WriteReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_IER, Register | Mask);
}

void XPhase2_rpm_cnn_InterruptDisable(XPhase2_rpm_cnn *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XPhase2_rpm_cnn_ReadReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_IER);
    XPhase2_rpm_cnn_WriteReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_IER, Register & (~Mask));
}

void XPhase2_rpm_cnn_InterruptClear(XPhase2_rpm_cnn *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPhase2_rpm_cnn_WriteReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_ISR, Mask);
}

u32 XPhase2_rpm_cnn_InterruptGetEnabled(XPhase2_rpm_cnn *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XPhase2_rpm_cnn_ReadReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_IER);
}

u32 XPhase2_rpm_cnn_InterruptGetStatus(XPhase2_rpm_cnn *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XPhase2_rpm_cnn_ReadReg(InstancePtr->Control_BaseAddress, XPHASE2_RPM_CNN_CONTROL_ADDR_ISR);
}

