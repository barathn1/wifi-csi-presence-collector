#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "cfg_store.h"

/* Both transports implement this same signature so switching between
 * them is a config change, not a rebuild. Returns true if handed off
 * successfully. */
typedef bool (*csi_transport_send_fn_t)(const uint8_t *buf, size_t len);

/* Enables CSI capture (promiscuous mode + esp_wifi_set_csi_config/rx_cb)
 * per cfg, and starts the sender task that drains the internal queue,
 * wire-packs each sample (wire_format.h), and hands it to send_fn.
 *
 * Call after WiFi STA is connected -- bssid must be the AP's real BSSID,
 * used for the software filter that rejects CSI triggered by frames
 * from unrelated nearby WiFi networks (only active if cfg->bssidflt_en).
 *
 * Promiscuous mode is required, not merely a density optimization:
 * esp_wifi_set_csi() itself is documented to depend on it. */
void csi_capture_start(const cfg_store_t *cfg, const uint8_t *bssid,
                        csi_transport_send_fn_t send_fn);
