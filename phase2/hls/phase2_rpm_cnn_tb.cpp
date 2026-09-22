#include "phase2_weights_float.h"
#include "D:/FPGA/vitis/phase3cnn_onboard/halotest/src/phase3_test_vector.h"
#include <cstdio>
#include <cstring>
#include <cstdint>

extern "C" void phase2_rpm_cnn(
    const float input[PHASE2_INPUT_LEN],
    float *output_rpm
);

int main() {
    static_assert(TEST_INPUT_LEN == PHASE2_INPUT_LEN,
                  "The board and HLS input lengths must match.");

    float output = 0.0f;
    uint32_t output_bits = 0;

    phase2_rpm_cnn(test_fft_input, &output);
    std::memcpy(&output_bits, &output, sizeof(output));

    std::printf("=== Phase2 HLS C Simulation: board test vector ===\n");
    std::printf("test split                    = test_id, index = 0\n");
    std::printf("true_rpm_label                = %.6f\n", TEST_TRUE_RPM);
    std::printf("phase2_pytorch_expected       = %.6f\n", TEST_PHASE2_PYTORCH_RPM);
    std::printf("hls_int8_float_emu_expected   = %.6f\n", TEST_HLS_INT8_FLOAT_EMU_RPM);
    std::printf("hls_csim_output               = %.6f\n", output);
    std::printf("hls_csim_raw_output_bits      = 0x%08x\n", output_bits);
    std::printf("csim_minus_int8_float_emu     = %.6f\n",
                output - TEST_HLS_INT8_FLOAT_EMU_RPM);
    return 0;
}
