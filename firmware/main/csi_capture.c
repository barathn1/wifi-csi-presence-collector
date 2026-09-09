#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_wifi.h"

#include "csi_capture.h"
#include "stats.h"
#include "wire_format.h"

static const char *TAG = "csi_capture";

/* CSI events can arrive in tight bursts (observed: 8+ events within
 * ~1ms of each other, likely aggregated/retried frames) much faster
 * than the sender task can drain them one at a time -- a bigger queue
 * absorbs bursts without dropping, at the cost of ~90KB RAM at this
 * depth (fine on the S3's available internal RAM). */
#define CSI_QUEUE_LEN 128

/* One queue item = this header, followed immediately in the same
 * allocated slot by up to max_csi_len raw CSI bytes. FreeRTOS queue
 * items must be a fixed size, so the trailing CSI region's capacity
 * (s_max_csi_len) is fixed once at csi_capture_start() and every slot
 * (producer scratch buffer, queue storage, consumer buffer) is sized
 * identically: sizeof(csi_queue_header_t) + s_max_csi_len. */
typedef struct {
    uint32_t seq;
    wifi_csi_info_t info; /* .buf/.hdr/.payload are stale after this copy;
                              the consumer repoints .buf at its own copy
                              of the trailing CSI bytes before use. */
} csi_queue_header_t;

static QueueHandle_t s_queue = NULL;
static uint8_t *s_scratch = NULL; /* WiFi-task-exclusive assembly buffer */
static size_t s_item_size = 0;
static uint16_t s_max_csi_len = 0;
static uint32_t s_seq_counter = 0;

static bool s_bssid_filter_enabled = false;
static uint8_t s_bssid[6] = {0};

static csi_transport_send_fn_t s_send_fn = NULL;

static void csi_rx_callback(void *ctx, wifi_csi_info_t *info)
{
    if (info == NULL || info->buf == NULL) {
        return;
    }

    if (s_bssid_filter_enabled &&
        memcmp(info->mac, s_bssid, 6) != 0 &&
        memcmp(info->dmac, s_bssid, 6) != 0) {
        return; /* ambient traffic from an unrelated nearby network */
    }

    g_csi_stats.csi_cb_count++;

    csi_queue_header_t *hdr = (csi_queue_header_t *)s_scratch;
    hdr->seq = s_seq_counter++;
    hdr->info = *info;

    uint16_t copy_len = info->len < s_max_csi_len ? info->len : s_max_csi_len;
    hdr->info.len = copy_len;
    memcpy(s_scratch + sizeof(csi_queue_header_t), info->buf, copy_len);

    if (xQueueSend(s_queue, s_scratch, 0) == pdTRUE) {
        g_csi_stats.enqueued++;
    } else {
        g_csi_stats.drop_queue_full++;
    }
}

static void sender_task(void *arg)
{
    uint8_t *item = malloc(s_item_size);
    uint8_t *wire_buf = malloc((size_t)WIRE_HEADER_SIZE + s_max_csi_len);

    while (1) {
        if (xQueueReceive(s_queue, item, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        csi_queue_header_t *hdr = (csi_queue_header_t *)item;
        wifi_csi_info_t info = hdr->info;
        info.buf = (int8_t *)(item + sizeof(csi_queue_header_t));

        size_t frame_len = wire_pack_frame(wire_buf, (size_t)WIRE_HEADER_SIZE + s_max_csi_len,
                                            hdr->seq, &info, s_max_csi_len);
        if (frame_len == 0 || !s_send_fn(wire_buf, frame_len)) {
            g_csi_stats.drop_tx_fail++;
        }
    }
}

void csi_capture_start(const cfg_store_t *cfg, const uint8_t *bssid,
                        csi_transport_send_fn_t send_fn)
{
    s_max_csi_len = cfg->max_csi_len;
    s_item_size = sizeof(csi_queue_header_t) + s_max_csi_len;
    s_send_fn = send_fn;
    s_bssid_filter_enabled = cfg->bssidflt_en;
    if (bssid != NULL) {
        memcpy(s_bssid, bssid, 6);
    }

    s_scratch = malloc(s_item_size);
    s_queue = xQueueCreate(CSI_QUEUE_LEN, s_item_size);

    if (cfg->promisc_en) {
        wifi_promiscuous_filter_t filter = {.filter_mask = WIFI_PROMIS_FILTER_MASK_DATA};
        if (strcmp(cfg->promisc_flt, "mgmt") == 0) {
            filter.filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT;
        } else if (strcmp(cfg->promisc_flt, "all") == 0) {
            filter.filter_mask = WIFI_PROMIS_FILTER_MASK_ALL;
        }
        ESP_ERROR_CHECK(esp_wifi_set_promiscuous_filter(&filter));
        ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
    } else {
        ESP_LOGW(TAG, "promiscuous mode disabled -- CSI is likely too sparse to be useful, "
                      "and esp_wifi_set_csi() may itself require it");
    }

    wifi_csi_config_t csi_config = {
        .lltf_en = cfg->lltf_en,
        .htltf_en = cfg->htltf_en,
        .stbc_htltf2_en = cfg->stbc_en,
        .ltf_merge_en = cfg->ltfmrg_en,
        .channel_filter_en = cfg->chfilt_en,
        .manu_scale = cfg->manu_scale,
        .shift = cfg->shift,
        .dump_ack_en = cfg->dumpack_en,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&csi_rx_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    xTaskCreate(sender_task, "csi_sender", 4096, NULL, tskIDLE_PRIORITY + 2, NULL);

    ESP_LOGI(TAG, "CSI capture started (max_csi_len=%u, bssid_filter=%d)",
             s_max_csi_len, s_bssid_filter_enabled);
}
