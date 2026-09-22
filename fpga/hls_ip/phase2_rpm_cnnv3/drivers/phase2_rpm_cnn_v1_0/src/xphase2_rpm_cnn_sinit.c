// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.2 (64-bit)
// Tool Version Limit: 2024.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#ifdef SDT
#include "xparameters.h"
#endif
#include "xphase2_rpm_cnn.h"

extern XPhase2_rpm_cnn_Config XPhase2_rpm_cnn_ConfigTable[];

#ifdef SDT
XPhase2_rpm_cnn_Config *XPhase2_rpm_cnn_LookupConfig(UINTPTR BaseAddress) {
	XPhase2_rpm_cnn_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XPhase2_rpm_cnn_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XPhase2_rpm_cnn_ConfigTable[Index].Control_BaseAddress == BaseAddress) {
			ConfigPtr = &XPhase2_rpm_cnn_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XPhase2_rpm_cnn_Initialize(XPhase2_rpm_cnn *InstancePtr, UINTPTR BaseAddress) {
	XPhase2_rpm_cnn_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XPhase2_rpm_cnn_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XPhase2_rpm_cnn_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XPhase2_rpm_cnn_Config *XPhase2_rpm_cnn_LookupConfig(u16 DeviceId) {
	XPhase2_rpm_cnn_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XPHASE2_RPM_CNN_NUM_INSTANCES; Index++) {
		if (XPhase2_rpm_cnn_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XPhase2_rpm_cnn_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XPhase2_rpm_cnn_Initialize(XPhase2_rpm_cnn *InstancePtr, u16 DeviceId) {
	XPhase2_rpm_cnn_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XPhase2_rpm_cnn_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XPhase2_rpm_cnn_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

