#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "esp_netif.h"

/* Joins WiFi as a station with bounded auto-reconnect backoff (a
 * multi-minute collection session shouldn't die on one hotspot hiccup).
 * Blocks until the first connection succeeds. Also disables modem sleep
 * (esp_wifi_set_ps(WIFI_PS_NONE)) before connecting, required to avoid
 * gaps in CSI capture. */
void wifi_sta_init_and_connect(const char *ssid, const char *password);

bool wifi_sta_is_connected(void);

/* BSSID of the AP we're associated to, valid once connected. Used by the
 * promiscuous-mode CSI filter to reject frames from unrelated nearby
 * WiFi networks. */
const uint8_t *wifi_sta_get_bssid(void);

/* The STA netif, valid once connected. Used by tx_sender to resolve the
 * hotspot's gateway IP (esp_netif_get_ip_info) without hardcoding it. */
esp_netif_t *wifi_sta_get_netif(void);
