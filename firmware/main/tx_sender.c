#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <arpa/inet.h>
#include <sys/socket.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"

#include "tx_sender.h"
#include "wifi_sta.h"

static const char *TAG = "tx_sender";

/* Arbitrary, fixed port -- nothing needs to listen on it. The receiver
 * board just overhears this traffic on-air in promiscuous mode. */
#define TX_UDP_PORT 45871

static uint16_t s_tx_rate_hz;
static uint16_t s_tx_pkt_sz;

static void tx_task(void *arg)
{
    uint8_t *payload = calloc(1, s_tx_pkt_sz);

    esp_netif_ip_info_t ip_info;
    ESP_ERROR_CHECK(esp_netif_get_ip_info(wifi_sta_get_netif(), &ip_info));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TX_UDP_PORT);
    addr.sin_addr.s_addr = ip_info.gw.addr;

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "socket() failed: errno %d", errno);
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "sending %u B/packet at %u Hz to gateway " IPSTR,
             s_tx_pkt_sz, s_tx_rate_hz, IP2STR(&ip_info.gw));

    int64_t period_us = 1000000LL / s_tx_rate_hz;
    int64_t next = esp_timer_get_time();
    uint32_t sent = 0;

    while (1) {
        sendto(sock, payload, s_tx_pkt_sz, 0, (struct sockaddr *)&addr, sizeof(addr));
        sent++;

        if (sent % ((uint32_t)s_tx_rate_hz * 2) == 0) {
            ESP_LOGI(TAG, "#TXSTAT,sent=%u,upt=%lld",
                     (unsigned)sent, (long long)(esp_timer_get_time() / 1000));
        }

        next += period_us;
        int64_t now = esp_timer_get_time();
        int64_t delay_us = next - now;
        if (delay_us > 0) {
            vTaskDelay(pdMS_TO_TICKS(delay_us / 1000 > 0 ? delay_us / 1000 : 1));
        } else {
            next = now; /* fell behind -- resync instead of accumulating drift */
        }
    }
}

void tx_sender_start(const cfg_store_t *cfg)
{
    s_tx_rate_hz = cfg->tx_rate_hz;
    s_tx_pkt_sz = cfg->tx_pkt_sz;
    xTaskCreate(tx_task, "tx_sender", 4096, NULL, tskIDLE_PRIORITY + 2, NULL);
}
