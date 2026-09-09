#pragma once

#include <stdbool.h>
#include <stdint.h>

/* Mirrors the NVS keys written by collector/nvs_gen.py into the
 * "cfgstore" partition (see scripts/push_config.sh). config/config.yaml
 * is the only human-edited source of truth -- this is just the runtime
 * view of it.
 *
 * `role` picks which part of this struct actually matters: "hotspot"
 * (AP mode, hosts ssid/pass on ap_channel, no CSI work) only needs
 * ssid/pass/ap_channel; "receiver" needs everything else; "transmitter"
 * (optional 3rd-board role) needs ssid/pass + tx_rate_hz/tx_pkt_sz. All
 * roles share the same firmware binary and struct -- only the pushed
 * NVS content differs. */
typedef struct {
    char role[16]; /* "hotspot" | "transmitter" | "receiver" */
    char ssid[33];
    char pass[65];

    /* hotspot-only */
    uint8_t ap_channel;

    /* transmitter-only (optional 3rd-board role) */
    uint16_t tx_rate_hz;
    uint16_t tx_pkt_sz;

    /* receiver-only */
    char laptop_ip[16];
    uint16_t tcp_port;
    char mode[8]; /* "tcp" | "serial" */
    uint16_t max_csi_len;

    bool lltf_en;
    bool htltf_en;
    bool stbc_en;
    bool ltfmrg_en;
    bool chfilt_en;
    bool manu_scale;
    uint8_t shift;
    bool dumpack_en;

    bool promisc_en;
    char promisc_flt[8]; /* "data" | "mgmt" | "all" */
    bool bssidflt_en;
} cfg_store_t;

/* Loads cfgstore from NVS into *out. Returns false if the partition is
 * blank/incomplete for the pushed role -- callers should halt WiFi
 * bring-up rather than fall back to a compiled-in default, which would
 * reintroduce a second source of truth. */
bool cfg_store_load(cfg_store_t *out);
