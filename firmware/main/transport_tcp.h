#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "cfg_store.h"

/* Connects as a TCP CLIENT to cfg->laptop_ip:cfg->tcp_port -- the laptop
 * is deliberately the server, so the main data path never needs to
 * discover the ESP32's own DHCP-assigned IP. Starts a background task
 * that reconnects with backoff on disconnect, so a multi-minute session
 * survives a laptop-side listener restart. Blocks until the first
 * connection succeeds. */
void transport_tcp_start(const cfg_store_t *cfg);

/* Matches csi_transport_send_fn_t. Fails fast (no long retry) if not
 * currently connected or if the send fails; the background task handles
 * reconnecting. */
bool transport_tcp_send(const uint8_t *buf, size_t len);
