#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_wifi.h"

#include "ap_hotspot.h"

static const char *TAG = "ap_hotspot";

static void event_handler(void *arg, esp_event_base_t event_base,
                           int32_t event_id, void *event_data)
{
    if (event_base != WIFI_EVENT) {
        return;
    }
    if (event_id == WIFI_EVENT_AP_STACONNECTED) {
        wifi_event_ap_staconnected_t *evt = (wifi_event_ap_staconnected_t *)event_data;
        ESP_LOGI(TAG, "station " MACSTR " joined, aid=%d", MAC2STR(evt->mac), evt->aid);
    } else if (event_id == WIFI_EVENT_AP_STADISCONNECTED) {
        wifi_event_ap_stadisconnected_t *evt = (wifi_event_ap_stadisconnected_t *)event_data;
        ESP_LOGI(TAG, "station " MACSTR " left, reason=%d", MAC2STR(evt->mac), evt->reason);
    }
}

void ap_hotspot_start(const cfg_store_t *cfg)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_cfg));

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL));

    wifi_config_t wifi_config = {0};
    strncpy((char *)wifi_config.ap.ssid, cfg->ssid, sizeof(wifi_config.ap.ssid) - 1);
    wifi_config.ap.ssid_len = strlen(cfg->ssid);
    strncpy((char *)wifi_config.ap.password, cfg->pass, sizeof(wifi_config.ap.password) - 1);
    wifi_config.ap.channel = cfg->ap_channel;
    wifi_config.ap.max_connection = 4;
    wifi_config.ap.authmode = strlen(cfg->pass) == 0 ? WIFI_AUTH_OPEN : WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "hosting SSID '%s' on channel %d", cfg->ssid, cfg->ap_channel);
}
