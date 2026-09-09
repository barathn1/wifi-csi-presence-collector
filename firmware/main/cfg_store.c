#include <string.h>

#include "esp_log.h"
#include "nvs.h"
#include "nvs_flash.h"

#include "cfg_store.h"

#define CFGSTORE_PARTITION "cfgstore"

static const char *TAG = "cfg_store";

static bool get_str(nvs_handle_t h, const char *key, char *out, size_t out_cap)
{
    size_t len = out_cap;
    esp_err_t err = nvs_get_str(h, key, out, &len);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "missing/invalid NVS key '%s': %s", key, esp_err_to_name(err));
        return false;
    }
    return true;
}

static bool get_u16(nvs_handle_t h, const char *key, uint16_t *out)
{
    esp_err_t err = nvs_get_u16(h, key, out);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "missing/invalid NVS key '%s': %s", key, esp_err_to_name(err));
        return false;
    }
    return true;
}

static bool get_u8(nvs_handle_t h, const char *key, uint8_t *out)
{
    esp_err_t err = nvs_get_u8(h, key, out);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "missing/invalid NVS key '%s': %s", key, esp_err_to_name(err));
        return false;
    }
    return true;
}

static bool get_bool(nvs_handle_t h, const char *key, bool *out)
{
    uint8_t v = 0;
    if (!get_u8(h, key, &v)) {
        return false;
    }
    *out = (v != 0);
    return true;
}

bool cfg_store_load(cfg_store_t *out)
{
    memset(out, 0, sizeof(*out));

    /* "cfgstore" is a separate named NVS partition (see
     * firmware/partitions.csv), not a namespace in the default "nvs"
     * partition -- it needs its own init call and
     * nvs_open_from_partition(), not plain nvs_open(). Kept isolated
     * from the system "nvs" partition so scripts/push_config.sh can
     * blast/rewrite it without touching WiFi PHY calibration data. */
    esp_err_t err = nvs_flash_init_partition(CFGSTORE_PARTITION);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "failed to init NVS partition '%s': %s", CFGSTORE_PARTITION, esp_err_to_name(err));
        return false;
    }

    nvs_handle_t h;
    err = nvs_open_from_partition(CFGSTORE_PARTITION, CFGSTORE_PARTITION, NVS_READONLY, &h);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "no config in NVS (%s) -- run scripts/push_config.sh from the laptop", esp_err_to_name(err));
        return false;
    }

    bool ok = true;
    ok &= get_str(h, "role", out->role, sizeof(out->role));
    ok &= get_str(h, "ssid", out->ssid, sizeof(out->ssid));
    ok &= get_str(h, "pass", out->pass, sizeof(out->pass));

    if (ok && strcmp(out->role, "hotspot") == 0) {
        ok &= get_u8(h, "ap_channel", &out->ap_channel);
    } else if (ok && strcmp(out->role, "transmitter") == 0) {
        ok &= get_u16(h, "tx_rate_hz", &out->tx_rate_hz);
        ok &= get_u16(h, "tx_pkt_sz", &out->tx_pkt_sz);
    } else if (ok) {
        /* receiver (default if role string is somehow unset) */
        ok &= get_str(h, "laptop_ip", out->laptop_ip, sizeof(out->laptop_ip));
        ok &= get_u16(h, "tcp_port", &out->tcp_port);
        ok &= get_str(h, "mode", out->mode, sizeof(out->mode));
        ok &= get_u16(h, "max_csi_len", &out->max_csi_len);

        ok &= get_bool(h, "lltf_en", &out->lltf_en);
        ok &= get_bool(h, "htltf_en", &out->htltf_en);
        ok &= get_bool(h, "stbc_en", &out->stbc_en);
        ok &= get_bool(h, "ltfmrg_en", &out->ltfmrg_en);
        ok &= get_bool(h, "chfilt_en", &out->chfilt_en);
        ok &= get_bool(h, "manu_scale", &out->manu_scale);
        ok &= get_u8(h, "shift", &out->shift);
        ok &= get_bool(h, "dumpack_en", &out->dumpack_en);

        ok &= get_bool(h, "promisc_en", &out->promisc_en);
        ok &= get_str(h, "promisc_flt", out->promisc_flt, sizeof(out->promisc_flt));
        ok &= get_bool(h, "bssidflt_en", &out->bssidflt_en);
    }

    nvs_close(h);

    if (!ok) {
        ESP_LOGE(TAG, "config in NVS is incomplete -- run scripts/push_config.sh from the laptop");
    }
    return ok;
}
