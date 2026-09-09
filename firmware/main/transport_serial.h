#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "cfg_store.h"

/* Serial transport: writes framed CSI data directly to stdout (the
 * board's USB-Serial/JTAG virtual COM port -- the same link used for
 * flashing/console). No connection step needed; this is always
 * available since it's the same physical USB cable. */
void transport_serial_start(const cfg_store_t *cfg);

/* Matches csi_transport_send_fn_t. */
bool transport_serial_send(const uint8_t *buf, size_t len);
