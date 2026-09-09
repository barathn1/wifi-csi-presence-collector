#pragma once

#include "cfg_store.h"

/* Runs the hotspot role: hosts a WiFi access point (cfg->ssid/pass, on
 * cfg->ap_channel) for the receiver board (and the laptop) to join. No
 * CSI work, no STA join -- this board only hosts the network. */
void ap_hotspot_start(const cfg_store_t *cfg);
