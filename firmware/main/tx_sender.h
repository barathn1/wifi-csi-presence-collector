#pragma once

#include "cfg_store.h"

/* Runs the transmitter role: sends a steady stream of UDP packets to the
 * hotspot's gateway (resolved via wifi_sta_get_netif(), never
 * hardcoded), at cfg->tx_rate_hz. This is the ONLY thing a transmitter
 * board does -- no CSI capture, no data transport to the laptop. Its
 * traffic exists purely for a separate receiver board to overhear and
 * extract CSI from. Call after wifi_sta_init_and_connect() succeeds. */
void tx_sender_start(const cfg_store_t *cfg);
