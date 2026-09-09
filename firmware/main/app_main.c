#include <string.h>

#include "esp_log.h"
#include "nvs_flash.h"

#include "ap_hotspot.h"
#include "cfg_store.h"
#include "csi_capture.h"
#include "stats.h"
#include "transport_serial.h"
#include "transport_tcp.h"
#include "tx_sender.h"
#include "wifi_sta.h"

static const char *TAG = "app_main";

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());

    cfg_store_t cfg;
    if (!cfg_store_load(&cfg)) {
        ESP_LOGE(TAG, "halting: no valid config in NVS -- run scripts/push_config.sh "
                      "from the laptop, then reset the board");
        return;
    }

    if (strcmp(cfg.role, "hotspot") == 0) {
        /* AP mode -- hosts the network, never joins one as a station. */
        ap_hotspot_start(&cfg);
        ESP_LOGI(TAG, "running as HOTSPOT (ssid=%s, channel=%u)", cfg.ssid, cfg.ap_channel);
        return;
    }

    /* transmitter/receiver both join the hotspot (ESP-hosted or otherwise) as a station. */
    wifi_sta_init_and_connect(cfg.ssid, cfg.pass);

    if (strcmp(cfg.role, "transmitter") == 0) {
        tx_sender_start(&cfg);
        ESP_LOGI(TAG, "running as TRANSMITTER (tx_rate_hz=%u, pkt_size=%u)",
                 cfg.tx_rate_hz, cfg.tx_pkt_sz);
        return;
    }

    csi_transport_send_fn_t send_fn;
    if (strcmp(cfg.mode, "serial") == 0) {
        transport_serial_start(&cfg);
        send_fn = transport_serial_send;
    } else {
        transport_tcp_start(&cfg);
        send_fn = transport_tcp_send;
    }

    csi_capture_start(&cfg, wifi_sta_get_bssid(), send_fn);
    stats_start_periodic_report();

    ESP_LOGI(TAG, "running as RECEIVER (transport=%s)", cfg.mode);
}
