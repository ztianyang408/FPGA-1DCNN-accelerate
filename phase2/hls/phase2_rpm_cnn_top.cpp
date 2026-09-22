#include "phase2_weights_int8.h"
#include <ap_fixed.h>

typedef ap_fixed<18, 8, AP_RND, AP_SAT> data_t;
// Scales are much smaller than activations; keep enough fractional bits so
// layer dequantization constants are not quantized to coarse 1/1024 steps.
typedef ap_fixed<24, 4, AP_RND, AP_SAT> scale_t;
typedef ap_fixed<40, 24, AP_RND, AP_SAT> acc_t;
typedef ap_fixed<32, 16, AP_RND, AP_SAT> out_t;

extern "C" {

static inline data_t relu(data_t x) {
    return x > (data_t)0 ? x : (data_t)0;
}

static void normalize_input(const float input[PHASE2_INPUT_LEN], data_t output[PHASE2_INPUT_LEN]) {
    for (int i = 0; i < PHASE2_INPUT_LEN; ++i) {
#pragma HLS PIPELINE II=1
        const data_t x = (data_t)input[i];
        const data_t mean = (data_t)phase2_train_mean[i];
        const data_t std = (data_t)phase2_train_std[i];
        output[i] = (x - mean) / std;
    }
}

static void conv1d_same_q(
    const data_t *input,
    data_t *output,
    int in_len,
    int in_ch,
    int out_ch,
    int kernel_size,
    int pad,
    const int8_t *weight_q,
    scale_t weight_scale,
    const int8_t *bias_q,
    scale_t bias_scale
) {
    const int out_len = in_len;
    for (int oc = 0; oc < out_ch; ++oc) {
        for (int i = 0; i < out_len; ++i) {
#pragma HLS PIPELINE II=8//这里再快dsp要爆了
            acc_t acc = 0;
            for (int ic = 0; ic < in_ch; ++ic) {
                for (int k = 0; k < kernel_size; ++k) {
                    const int idx = i + k - pad;
                    if (idx >= 0 && idx < in_len) {
                        const int widx = ((oc * in_ch + ic) * kernel_size) + k;
                        const int xidx = (ic * in_len) + idx;
                        acc += (acc_t)input[xidx] * (acc_t)weight_q[widx];
                    }
                }
            }
            output[oc * out_len + i] = (data_t)(acc * (acc_t)weight_scale + (acc_t)((data_t)bias_q[oc] * bias_scale));
        }
    }
}

static void maxpool1d_2(
    const data_t *input,
    data_t *output,
    int channels,
    int in_len
) {
    const int out_len = in_len / 2;
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < out_len; ++i) {
#pragma HLS PIPELINE II=1//最大池化层对效率影响及大，优先分配资源
            const data_t a = input[c * in_len + 2 * i];
            const data_t b = input[c * in_len + 2 * i + 1];
            output[c * out_len + i] = a > b ? a : b;
        }
    }
}

static void global_mean_pool(
    const data_t *input,
    data_t output[32],
    int channels,
    int len
) {
    for (int c = 0; c < channels; ++c) {
        acc_t sum = 0;
        for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
            sum += (acc_t)input[c * len + i];
        }
        output[c] = (data_t)(sum / (acc_t)len);
    }
}

static void dense_q(
    const data_t *input,
    data_t *output,
    int in_dim,
    int out_dim,
    const int8_t *weight_q,
    scale_t weight_scale,
    const int8_t *bias_q,
    scale_t bias_scale
) {
    for (int o = 0; o < out_dim; ++o) {
        acc_t acc = 0;
        for (int i = 0; i < in_dim; ++i) {
#pragma HLS PIPELINE II=16
            acc += (acc_t)input[i] * (acc_t)weight_q[o * in_dim + i];
        }
        output[o] = (data_t)(acc * (acc_t)weight_scale + (acc_t)((data_t)bias_q[o] * bias_scale));
    }
}

void phase2_rpm_cnn(const float input[PHASE2_INPUT_LEN], float *output_rpm) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem depth=512
#pragma HLS INTERFACE m_axi port=output_rpm offset=slave bundle=gmem depth=1
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output_rpm bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t x0[PHASE2_INPUT_LEN];
    data_t conv1_out[8 * PHASE2_INPUT_LEN];
    data_t pool1_out[8 * (PHASE2_INPUT_LEN / 2)];
    data_t conv2_out[16 * (PHASE2_INPUT_LEN / 2)];
    data_t pool2_out[16 * (PHASE2_INPUT_LEN / 4)];
    data_t conv3_out[32 * (PHASE2_INPUT_LEN / 4)];
    data_t pooled[32];
    data_t fc1_out[16];
    data_t fc2_out[1];

#pragma HLS BIND_STORAGE variable=x0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=conv1_out type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=pool1_out type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=conv2_out type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=pool2_out type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=conv3_out type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=pooled complete
#pragma HLS ARRAY_PARTITION variable=fc1_out complete
#pragma HLS ARRAY_PARTITION variable=fc2_out complete

    normalize_input(input, x0);

    conv1d_same_q(
        x0, conv1_out, PHASE2_INPUT_LEN, 1, 8, 7, 3,
        conv1_weight_q, (scale_t)CONV1_WEIGHT_SCALE,
        conv1_bias_q, (scale_t)CONV1_BIAS_SCALE
    );
    for (int i = 0; i < 8 * PHASE2_INPUT_LEN; ++i) {
#pragma HLS PIPELINE II=1
        conv1_out[i] = relu(conv1_out[i]);
    }

    maxpool1d_2(conv1_out, pool1_out, 8, PHASE2_INPUT_LEN);
    conv1d_same_q(
        pool1_out, conv2_out, PHASE2_INPUT_LEN / 2, 8, 16, 5, 2,
        conv2_weight_q, (scale_t)CONV2_WEIGHT_SCALE,
        conv2_bias_q, (scale_t)CONV2_BIAS_SCALE
    );
    for (int i = 0; i < 16 * (PHASE2_INPUT_LEN / 2); ++i) {
#pragma HLS PIPELINE II=1
        conv2_out[i] = relu(conv2_out[i]);
    }

    maxpool1d_2(conv2_out, pool2_out, 16, PHASE2_INPUT_LEN / 2);
    conv1d_same_q(
        pool2_out, conv3_out, PHASE2_INPUT_LEN / 4, 16, 32, 3, 1,
        conv3_weight_q, (scale_t)CONV3_WEIGHT_SCALE,
        conv3_bias_q, (scale_t)CONV3_BIAS_SCALE
    );
    for (int i = 0; i < 32 * (PHASE2_INPUT_LEN / 4); ++i) {
#pragma HLS PIPELINE II=1
        conv3_out[i] = relu(conv3_out[i]);
    }

    global_mean_pool(conv3_out, pooled, 32, PHASE2_INPUT_LEN / 4);
    dense_q(
        pooled, fc1_out, 32, 16,
        fc1_weight_q, (scale_t)FC1_WEIGHT_SCALE,
        fc1_bias_q, (scale_t)FC1_BIAS_SCALE
    );
    for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
        fc1_out[i] = relu(fc1_out[i]);
    }
    dense_q(
        fc1_out, fc2_out, 16, 1,
        fc2_weight_q, (scale_t)FC2_WEIGHT_SCALE,
        fc2_bias_q, (scale_t)FC2_BIAS_SCALE
    );

    const out_t rpm = (out_t)fc2_out[0] * (out_t)PHASE2_TARGET_STD + (out_t)PHASE2_TARGET_MEAN;
    *output_rpm = (float)rpm;
}

}
