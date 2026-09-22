#include "platform.h"
#include "sleep.h"
#include "xil_cache.h"
#include "xgpio.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xphase2_rpm_cnn.h"
#include "xtime_l.h"
#include "phase3_test_vector.h"

#if defined(XPAR_XIICPS_0_DEVICE_ID)
#include "xiicps.h"
#include <string.h>
#define OLED_I2C_AVAILABLE 1
#define OLED_I2C_DEVICE_ID XPAR_XIICPS_0_DEVICE_ID
#else
#define OLED_I2C_AVAILABLE 0
#endif

#define GPIO_LED_BTN_DEVICE_ID XPAR_AXI_GPIO_0_DEVICE_ID
#define GPIO_SW_DEVICE_ID      XPAR_AXI_GPIO_1_DEVICE_ID

#define LED_CHANNEL 1U
#define BTN_CHANNEL 2U
#define SW_CHANNEL  1U

#define OLED_WIDTH 128U
#define OLED_PAGE_COUNT 8U
#define OLED_I2C_ADDR_0 0x3CU
#define OLED_I2C_ADDR_1 0x3DU
#define OLED_COLUMN_OFFSET 2U

#define CNN_DEVICE_ID XPAR_PHASE2_RPM_CNN_0_DEVICE_ID
#define CNN_INPUT_ADDR 0x01000000U
#define CNN_OUTPUT_ADDR 0x01001000U
#define CNN_INPUT_LEN TEST_INPUT_LEN

static XGpio gpio_led_btn;
static XGpio gpio_sw;
static XPhase2_rpm_cnn cnn;
static int cnn_ready;
static int oled_ready;

#if OLED_I2C_AVAILABLE
static XIicPs oled_i2c;
static u16 oled_i2c_addr;

/* A compact 5x7 font for the text used by the board demonstration. */
static const char oled_font_chars[] =
    " 0123456789:.-ABCDEFGHIJKLMNOPQRSTUVWXYZ";

static const u8 oled_font[][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00}, /* space */
    {0x3E, 0x51, 0x49, 0x45, 0x3E}, /* 0 */
    {0x00, 0x42, 0x7F, 0x40, 0x00}, /* 1 */
    {0x42, 0x61, 0x51, 0x49, 0x46}, /* 2 */
    {0x21, 0x41, 0x45, 0x4B, 0x31}, /* 3 */
    {0x18, 0x14, 0x12, 0x7F, 0x10}, /* 4 */
    {0x27, 0x45, 0x45, 0x45, 0x39}, /* 5 */
    {0x3C, 0x4A, 0x49, 0x49, 0x30}, /* 6 */
    {0x01, 0x71, 0x09, 0x05, 0x03}, /* 7 */
    {0x36, 0x49, 0x49, 0x49, 0x36}, /* 8 */
    {0x06, 0x49, 0x49, 0x29, 0x1E}, /* 9 */
    {0x00, 0x36, 0x36, 0x00, 0x00}, /* : */
    {0x00, 0x60, 0x60, 0x00, 0x00}, /* . */
    {0x08, 0x08, 0x08, 0x08, 0x08}, /* - */
    {0x7E, 0x11, 0x11, 0x11, 0x7E}, /* A */
    {0x7F, 0x49, 0x49, 0x49, 0x36}, /* B */
    {0x3E, 0x41, 0x41, 0x41, 0x22}, /* C */
    {0x7F, 0x41, 0x41, 0x22, 0x1C}, /* D */
    {0x7F, 0x49, 0x49, 0x49, 0x41}, /* E */
    {0x7F, 0x09, 0x09, 0x09, 0x01}, /* F */
    {0x3E, 0x41, 0x49, 0x49, 0x7A}, /* G */
    {0x7F, 0x08, 0x08, 0x08, 0x7F}, /* H */
    {0x00, 0x41, 0x7F, 0x41, 0x00}, /* I */
    {0x20, 0x40, 0x41, 0x3F, 0x01}, /* J */
    {0x7F, 0x08, 0x14, 0x22, 0x41}, /* K */
    {0x7F, 0x40, 0x40, 0x40, 0x40}, /* L */
    {0x7F, 0x02, 0x0C, 0x02, 0x7F}, /* M */
    {0x7F, 0x04, 0x08, 0x10, 0x7F}, /* N */
    {0x3E, 0x41, 0x41, 0x41, 0x3E}, /* O */
    {0x7F, 0x09, 0x09, 0x09, 0x06}, /* P */
    {0x3E, 0x41, 0x51, 0x21, 0x5E}, /* Q */
    {0x7F, 0x09, 0x19, 0x29, 0x46}, /* R */
    {0x46, 0x49, 0x49, 0x49, 0x31}, /* S */
    {0x01, 0x01, 0x7F, 0x01, 0x01}, /* T */
    {0x3F, 0x40, 0x40, 0x40, 0x3F}, /* U */
    {0x1F, 0x20, 0x40, 0x20, 0x1F}, /* V */
    {0x3F, 0x40, 0x38, 0x40, 0x3F}, /* W */
    {0x63, 0x14, 0x08, 0x14, 0x63}, /* X */
    {0x07, 0x08, 0x70, 0x08, 0x07}, /* Y */
    {0x61, 0x51, 0x49, 0x45, 0x43}  /* Z */
};

static int oled_i2c_send(const u8 *data, u32 length)
{
    int status = XIicPs_MasterSendPolled(&oled_i2c, (u8 *)data, (s32)length,
                                         oled_i2c_addr);
    if (status != XST_SUCCESS) {
        /* Clear a NACK/aborted transfer before the next address probe. */
        XIicPs_Abort(&oled_i2c);
    }
    return status;
}

static int oled_probe_address(u16 address)
{
    /* A control byte with no command is harmless for SSD1306-compatible OLEDs. */
    const u8 probe = 0x00U;
    oled_i2c_addr = address;
    return oled_i2c_send(&probe, 1U);
}

static int oled_command(u8 command)
{
    const u8 packet[2] = {0x00U, command};
    return oled_i2c_send(packet, 2U);
}

static int oled_command2(u8 command, u8 value)
{
    const u8 packet[3] = {0x00U, command, value};
    return oled_i2c_send(packet, 3U);
}

static int oled_set_page(u8 page)
{
    if (oled_command((u8)(0xB0U | page)) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (oled_command((u8)(0x00U | (OLED_COLUMN_OFFSET & 0x0FU))) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    return oled_command((u8)(0x10U | ((OLED_COLUMN_OFFSET >> 4U) & 0x0FU)));
}

static int oled_write_page(const u8 *pixels)
{
    u8 packet[1U + OLED_WIDTH];

    packet[0] = 0x40U;
    memcpy(&packet[1], pixels, OLED_WIDTH);
    return oled_i2c_send(packet, sizeof(packet));
}

static int oled_clear(void)
{
    u8 blank[OLED_WIDTH];
    u8 page;

    memset(blank, 0, sizeof(blank));
    for (page = 0U; page < OLED_PAGE_COUNT; ++page) {
        if (oled_set_page(page) != XST_SUCCESS ||
            oled_write_page(blank) != XST_SUCCESS) {
            return XST_FAILURE;
        }
    }
    return XST_SUCCESS;
}

static const u8 *oled_glyph(char character)
{
    u32 index;

    for (index = 0U; index < sizeof(oled_font_chars) - 1U; ++index) {
        if (oled_font_chars[index] == character) {
            return oled_font[index];
        }
    }
    return oled_font[0];
}

static int oled_draw_string(u8 page, const char *text)
{
    u8 packet[1U + 21U * 6U];
    u32 count = 0U;

    if (oled_set_page(page) != XST_SUCCESS) {
        return XST_FAILURE;
    }

    packet[count++] = 0x40U;
    while (*text != '\0' && count + 6U <= sizeof(packet)) {
        const u8 *glyph = oled_glyph(*text++);
        u32 column;

        for (column = 0U; column < 5U; ++column) {
            packet[count++] = glyph[column];
        }
        packet[count++] = 0x00U;
    }
    return oled_i2c_send(packet, count);
}

static int oled_start(void)
{
    XIicPs_Config *config;
    u16 found_address = 0U;
    u16 address;
    int status;

    config = XIicPs_LookupConfig(OLED_I2C_DEVICE_ID);
    if (config == NULL) {
        xil_printf("OLED: I2C0 config not found\r\n");
        return XST_FAILURE;
    }

    status = XIicPs_CfgInitialize(&oled_i2c, config, config->BaseAddress);
    if (status != XST_SUCCESS) {
        xil_printf("OLED: I2C0 init failed: %d\r\n", status);
        return status;
    }

    XIicPs_SetOptions(&oled_i2c, XIICPS_7_BIT_ADDR_OPTION);
    XIicPs_SetSClk(&oled_i2c, 100000U);

    xil_printf("OLED: I2C bus busy before scan = %d\r\n",
               XIicPs_BusIsBusy(&oled_i2c));

    /* Try the two standard OLED addresses first, then scan the full 7-bit range. */
    if (oled_probe_address(OLED_I2C_ADDR_0) == XST_SUCCESS) {
        found_address = OLED_I2C_ADDR_0;
    } else if (oled_probe_address(OLED_I2C_ADDR_1) == XST_SUCCESS) {
        found_address = OLED_I2C_ADDR_1;
    } else {
        for (address = 0x08U; address <= 0x77U; ++address) {
            if (address == OLED_I2C_ADDR_0 || address == OLED_I2C_ADDR_1) {
                continue;
            }
            if (oled_probe_address(address) == XST_SUCCESS) {
                xil_printf("OLED: responding I2C address = 0x%02x\r\n", address);
                if (found_address == 0U) {
                    found_address = address;
                }
            }
        }
    }

    if (found_address == 0U) {
        xil_printf("OLED: no I2C device acknowledged\r\n");
        xil_printf("OLED: I2C bus busy after scan = %d\r\n",
                   XIicPs_BusIsBusy(&oled_i2c));
        return XST_FAILURE;
    }

    oled_i2c_addr = found_address;
    xil_printf("OLED: selected I2C address = 0x%02x\r\n", oled_i2c_addr);

    status = oled_command(0xAEU);
    if (status != XST_SUCCESS) {
        xil_printf("OLED: command transfer failed at selected address\r\n");
        return XST_FAILURE;
    }

    /* SSD1306-compatible 128x64 initialization. */
    if (oled_command2(0xD5U, 0x80U) != XST_SUCCESS ||
        oled_command2(0xA8U, 0x3FU) != XST_SUCCESS ||
        oled_command2(0xD3U, 0x00U) != XST_SUCCESS ||
        oled_command(0x40U) != XST_SUCCESS ||
        oled_command2(0x8DU, 0x14U) != XST_SUCCESS ||
        oled_command2(0x20U, 0x00U) != XST_SUCCESS ||
        oled_command(0xA1U) != XST_SUCCESS ||
        oled_command(0xC8U) != XST_SUCCESS ||
        oled_command2(0xDAU, 0x12U) != XST_SUCCESS ||
        oled_command2(0x81U, 0xCFU) != XST_SUCCESS ||
        oled_command2(0xD9U, 0xF1U) != XST_SUCCESS ||
        oled_command2(0xDBU, 0x40U) != XST_SUCCESS ||
        oled_command(0xA4U) != XST_SUCCESS ||
        oled_command(0xA6U) != XST_SUCCESS ||
        oled_command(0xAFU) != XST_SUCCESS ||
        oled_clear() != XST_SUCCESS) {
        xil_printf("OLED: initialization command failed\r\n");
        return XST_FAILURE;
    }

    return XST_SUCCESS;
}
#endif

static void oled_show_cnn_result(float rpm, u32 latency_us)
{
#if OLED_I2C_AVAILABLE
    char rpm_line[22] = "RPM:0000.0";
    char time_line[22] = "TIME:00000US";
    int whole = (int)rpm;
    int fraction = (int)((rpm - (float)whole) * 10.0f);
    u32 time_value = latency_us;

    if (whole < 0) {
        whole = 0;
    }
    if (whole > 9999) {
        whole = 9999;
    }
    if (fraction < 0) {
        fraction = 0;
    }
    if (fraction > 9) {
        fraction = 9;
    }
    if (time_value > 99999U) {
        time_value = 99999U;
    }

    rpm_line[4] = (char)('0' + (whole / 1000) % 10);
    rpm_line[5] = (char)('0' + (whole / 100) % 10);
    rpm_line[6] = (char)('0' + (whole / 10) % 10);
    rpm_line[7] = (char)('0' + whole % 10);
    rpm_line[9] = (char)('0' + fraction);

    time_line[5] = (char)('0' + (time_value / 10000U) % 10U);
    time_line[6] = (char)('0' + (time_value / 1000U) % 10U);
    time_line[7] = (char)('0' + (time_value / 100U) % 10U);
    time_line[8] = (char)('0' + (time_value / 10U) % 10U);
    time_line[9] = (char)('0' + time_value % 10U);

    if (!oled_ready) {
        return;
    }
    oled_draw_string(2U, "STATUS: DONE");
    oled_draw_string(4U, rpm_line);
    oled_draw_string(6U, time_line);
#else
    (void)rpm;
    (void)latency_us;
#endif
}

static int run_cnn_demo(float *rpm_out, u32 *latency_us_out,
                        u32 *raw_output_out)
{
    volatile float *input_buf = (volatile float *)CNN_INPUT_ADDR;
    volatile float *output_buf = (volatile float *)CNN_OUTPUT_ADDR;
    XTime start_time;
    XTime end_time;
    u64 cycles;
    int status;

    if (!cnn_ready) {
        return XST_FAILURE;
    }

    for (int i = 0; i < CNN_INPUT_LEN; ++i) {
        input_buf[i] = test_fft_input[i];
    }
    output_buf[0] = -1.0f;

    Xil_DCacheFlushRange((UINTPTR)CNN_INPUT_ADDR,
                         CNN_INPUT_LEN * sizeof(float));
    Xil_DCacheFlushRange((UINTPTR)CNN_OUTPUT_ADDR, sizeof(float));

    XPhase2_rpm_cnn_Set_input_r(&cnn, (u64)CNN_INPUT_ADDR);
    XPhase2_rpm_cnn_Set_output_rpm(&cnn, (u64)CNN_OUTPUT_ADDR);

    XTime_GetTime(&start_time);
    XPhase2_rpm_cnn_Start(&cnn);
    while (!XPhase2_rpm_cnn_IsDone(&cnn)) {
        /* Polling is sufficient for this first integrated demonstration. */
    }
    XTime_GetTime(&end_time);

    Xil_DCacheInvalidateRange((UINTPTR)CNN_OUTPUT_ADDR, sizeof(float));

    *rpm_out = output_buf[0];
    *raw_output_out = *((volatile u32 *)CNN_OUTPUT_ADDR);
    cycles = end_time - start_time;
    *latency_us_out = (u32)((cycles * 1000000ULL) / COUNTS_PER_SECOND);
    status = XST_SUCCESS;

    return status;
}

int main(void)
{
    int status;
    u32 switches;
    u32 buttons;
    u32 last_switches = 0xFFFFFFFFU;
    u32 last_buttons = 0xFFFFFFFFU;
    u32 demo_led_state = 0U;
    u32 demo_state = 0U; /* 0 idle, 1 running, 2 done, 3 error */

    init_platform();
    xil_printf("\r\n=== FPGA CNN edge demo ===\r\n");

    status = XGpio_Initialize(&gpio_led_btn, GPIO_LED_BTN_DEVICE_ID);
    if (status != XST_SUCCESS) {
        xil_printf("AXI GPIO 0 initialization failed: %d\r\n", status);
        cleanup_platform();
        return XST_FAILURE;
    }

    status = XGpio_Initialize(&gpio_sw, GPIO_SW_DEVICE_ID);
    if (status != XST_SUCCESS) {
        xil_printf("AXI GPIO 1 initialization failed: %d\r\n", status);
        cleanup_platform();
        return XST_FAILURE;
    }

    XGpio_SetDataDirection(&gpio_led_btn, LED_CHANNEL, 0x00U);
    XGpio_SetDataDirection(&gpio_led_btn, BTN_CHANNEL, 0x0FU);
    XGpio_SetDataDirection(&gpio_sw, SW_CHANNEL, 0xFFU);
    XGpio_DiscreteWrite(&gpio_led_btn, LED_CHANNEL, 0x00U);

#if OLED_I2C_AVAILABLE
    status = oled_start();
    if (status == XST_SUCCESS) {
        oled_ready = 1;
        xil_printf("OLED: initialized\r\n");
        oled_draw_string(0U, "FPGA CNN DEMO");
        oled_draw_string(2U, "STATUS: READY");
        oled_draw_string(4U, "SW:FF BTN:0");
    } else {
        xil_printf("OLED: initialization failed\r\n");
    }
#else
    xil_printf("OLED: PS I2C0 is not present in this BSP\r\n");
    xil_printf("Enable I2C0 on MIO50/MIO51, export XSA, and rebuild Platform.\r\n");
#endif

    status = XPhase2_rpm_cnn_Initialize(&cnn, CNN_DEVICE_ID);
    if (status != XST_SUCCESS) {
        xil_printf("CNN initialization failed: %d\r\n", status);
        XGpio_DiscreteWrite(&gpio_led_btn, LED_CHANNEL, 0x80U);
    } else {
        cnn_ready = 1;
        xil_printf("CNN IP initialized\r\n");
    }

    xil_printf("Demo ready: press BTN0 to run CNN\r\n");

    while (1) {
        char display_line[22];
        static u32 previous_buttons = 0U;

        switches = XGpio_DiscreteRead(&gpio_sw, SW_CHANNEL) & 0xFFU;
        buttons = XGpio_DiscreteRead(&gpio_led_btn, BTN_CHANNEL) & 0x0FU;

        /* Idle mirrors switches; inference states take ownership of LEDs. */
        if (demo_led_state == 0U) {
            XGpio_DiscreteWrite(&gpio_led_btn, LED_CHANNEL, switches);
        }

        /* BTN1 returns to idle switch-mirror mode and redraws the GPIO page. */
        if ((buttons & 0x02U) != 0U && (previous_buttons & 0x02U) == 0U) {
            demo_state = 0U;
            demo_led_state = 0U;
            XGpio_DiscreteWrite(&gpio_led_btn, LED_CHANNEL, switches);
#if OLED_I2C_AVAILABLE
            if (oled_ready) {
                oled_draw_string(2U, "STATUS: READY");
            }
#endif
        }

        /* BTN0 is the one-shot inference trigger; use an edge to avoid repeats. */
        if ((buttons & 0x01U) != 0U && (previous_buttons & 0x01U) == 0U) {
            float rpm = 0.0f;
            u32 latency_us = 0U;
            u32 raw_output = 0U;

            xil_printf("CNN demo trigger\r\n");
            demo_state = 1U;
            demo_led_state = 0x01U;
            XGpio_DiscreteWrite(&gpio_led_btn, LED_CHANNEL, 0x01U);
#if OLED_I2C_AVAILABLE
            if (oled_ready) {
                oled_draw_string(2U, "STATUS: RUNNING");
                oled_draw_string(4U, "RPM:----.-");
                oled_draw_string(6U, "TIME:-----US");
            }
#endif

            status = run_cnn_demo(&rpm, &latency_us, &raw_output);
            if (status == XST_SUCCESS) {
                xil_printf("CNN done\r\n");
                xil_printf("raw output bits = 0x%08x\r\n", raw_output);
                xil_printf("output_rpm = %d.%d\r\n",
                           (int)rpm,
                           (int)(((rpm - (float)((int)rpm)) * 10.0f)));
                xil_printf("cnn_latency_us = %d\r\n", latency_us);
                demo_state = 2U;
                demo_led_state = 0x02U;
                XGpio_DiscreteWrite(&gpio_led_btn, LED_CHANNEL, 0x02U);
                oled_show_cnn_result(rpm, latency_us);
            } else {
                xil_printf("CNN inference failed\r\n");
                demo_state = 3U;
                demo_led_state = 0x80U;
                XGpio_DiscreteWrite(&gpio_led_btn, LED_CHANNEL, 0x80U);
#if OLED_I2C_AVAILABLE
                if (oled_ready) {
                    oled_draw_string(2U, "STATUS: ERROR");
                }
#endif
            }
        }
        previous_buttons = buttons;

        if (switches != last_switches || buttons != last_buttons) {
            xil_printf("switch=0x%02x  button_raw=0x%01x\r\n",
                       switches, buttons);

        /* Do not let button-release logs overwrite the CNN result page. */
#if OLED_I2C_AVAILABLE
        if (demo_state == 0U &&
            (switches != last_switches || buttons != last_buttons)) {
            display_line[0] = 'S';
            display_line[1] = 'W';
            display_line[2] = ':';
            display_line[3] = "0123456789ABCDEF"[(switches >> 4U) & 0x0FU];
            display_line[4] = "0123456789ABCDEF"[switches & 0x0FU];
            display_line[5] = ' ';
            display_line[6] = 'B';
            display_line[7] = 'T';
            display_line[8] = 'N';
            display_line[9] = ':';
            display_line[10] = (char)('0' + (buttons & 0x0FU));
            display_line[11] = ' ';
            display_line[12] = 'R';
            display_line[13] = 'U';
            display_line[14] = 'N';
            display_line[15] = ' '; 
            display_line[16] = 'G';
            display_line[17] = 'P';
            display_line[18] = 'I';
            display_line[19] = 'O';
            display_line[20] = '\0';
            if (oled_ready) {
                oled_draw_string(4U, display_line);
            }
        }
#endif

            last_switches = switches;
            last_buttons = buttons;
        }

        usleep(100000U);
    }
}
