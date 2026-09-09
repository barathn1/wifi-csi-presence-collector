#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"

#include "wifi_sta.h"

#define WIFI_CONNECTED_BIT BIT0

static const char *TAG = "wifi_sta";

static EventGroupHandle_t s_wifi_event_group;
static volatile bool s_connected = false;
static uint8_t s_bssid[6] = {0};
static int s_retry_count = 0;
static esp_netif_t *s_netif = NULL;

static void event_handler(void *arg, esp_event_base_t event_base,
                           int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_CONNECTED) {
        wifi_event_sta_connected_t *evt = (wifi_event_sta_connected_t *)event_data;
        memcpy(s_bssid, evt->bssid, sizeof(s_bssid));
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_event_sta_disconnected_t *evt = (wifi_event_sta_disconnected_t *)event_data;
        s_connected = false;
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        s_retry_count++;
        int delay_ms = s_retry_count > 10 ? 10000 : 500 * s_retry_count;
        /* Common reason codes worth knowing without a lookup table:
         * 201 NO_AP_FOUND (SSID not seen in scan -- wrong SSID, out of
         *     range, or AP is on a band/channel this radio can't see, e.g.
         *     5GHz-only or band-steering away from 2.4GHz);
         * 2 AUTH_EXPIRE, 15 4WAY_HANDSHAKE_TIMEOUT (usually wrong password);
         * 8 ASSOC_LEAVE (AP kicked us, e.g. max clients). */
        ESP_LOGW(TAG, "disconnected (reason=%d), retrying in %d ms (attempt %d)",
                 evt->reason, delay_ms, s_retry_count);
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "got ip: " IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_count = 0;
        s_connected = true;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

void wifi_sta_init_and_connect(const char *ssid, const char *password)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    s_netif = esp_netif_create_default_wifi_sta();

    wifi_init_config_t init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_cfg));

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL));

    wifi_config_t wifi_config = {0};
    strncpy((char *)wifi_config.sta.ssid, ssid, sizeof(wifi_config.sta.ssid) - 1);
    strncpy((char *)wifi_config.sta.password, password, sizeof(wifi_config.sta.password) - 1);
    wifi_config.sta.threshold.authmode = strlen(password) == 0 ? WIFI_AUTH_OPEN : WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    /* Required for CSI: avoid modem-sleep gaps. */
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    ESP_LOGI(TAG, "connecting to SSID '%s' ...", ssid);
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
}

bool wifi_sta_is_connected(void)
{
    return s_connected;
}

const uint8_t *wifi_sta_get_bssid(void)
{
    return s_bssid;
}

esp_netif_t *wifi_sta_get_netif(void)
{
    return s_netif;
}
