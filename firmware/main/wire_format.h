#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_wifi.h"

#define WIRE_MAGIC 0xC51A
#define WIRE_VERSION 1
#define WIRE_HEADER_SIZE 40
#define WIRE_FLAG_FIRST_WORD_INVALID 0x01

/* Packs one CSI wire frame into `out` (caller-provided buffer of at
 * least WIRE_HEADER_SIZE + max_csi_len bytes). Returns the total frame
 * length, or 0 if out_cap is too small. CSI is truncated to max_csi_len
 * if longer (should not normally happen -- max_csi_len is sized to the
 * largest documented CSI length with headroom).
 *
 * Mirrors collector/wire.py's HEADER_FMT byte-for-byte -- keep both in
 * sync if either changes. Fields are memcpy'd individually rather than
 * through a C struct, matching the Python side, to avoid any compiler
 * struct-padding surprises; ESP32 is little-endian so no byte-swapping
 * is needed to match the little-endian wire format. */
size_t wire_pack_frame(uint8_t *out, size_t out_cap, uint32_t seq,
                        const wifi_csi_info_t *info, uint16_t max_csi_len);
