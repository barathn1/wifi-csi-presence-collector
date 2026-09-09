#include <string.h>

#include "esp_timer.h"

#include "wire_format.h"

size_t wire_pack_frame(uint8_t *out, size_t out_cap, uint32_t seq,
                        const wifi_csi_info_t *info, uint16_t max_csi_len)
{
    uint16_t csi_len = info->len;
    if (csi_len > max_csi_len) {
        csi_len = max_csi_len;
    }
    size_t total_len = (size_t)WIRE_HEADER_SIZE + csi_len;
    if (total_len > out_cap) {
        return 0;
    }

    uint8_t *p = out;

    uint16_t magic = WIRE_MAGIC;
    memcpy(p, &magic, sizeof(magic)); p += sizeof(magic);

    uint8_t version = WIRE_VERSION;
    memcpy(p, &version, sizeof(version)); p += sizeof(version);

    uint8_t flags = info->first_word_invalid ? WIRE_FLAG_FIRST_WORD_INVALID : 0;
    memcpy(p, &flags, sizeof(flags)); p += sizeof(flags);

    memcpy(p, &seq, sizeof(seq)); p += sizeof(seq);

    uint64_t device_time_us = (uint64_t)esp_timer_get_time();
    memcpy(p, &device_time_us, sizeof(device_time_us)); p += sizeof(device_time_us);

    int8_t rssi = (int8_t)info->rx_ctrl.rssi;
    memcpy(p, &rssi, sizeof(rssi)); p += sizeof(rssi);

    uint8_t channel_primary = (uint8_t)info->rx_ctrl.channel;
    memcpy(p, &channel_primary, sizeof(channel_primary)); p += sizeof(channel_primary);

    uint8_t channel_secondary = (uint8_t)info->rx_ctrl.secondary_channel;
    memcpy(p, &channel_secondary, sizeof(channel_secondary)); p += sizeof(channel_secondary);

    uint8_t sig_mode = (uint8_t)info->rx_ctrl.sig_mode;
    memcpy(p, &sig_mode, sizeof(sig_mode)); p += sizeof(sig_mode);

    uint8_t mcs = (uint8_t)info->rx_ctrl.mcs;
    memcpy(p, &mcs, sizeof(mcs)); p += sizeof(mcs);

    uint8_t cwb = (uint8_t)info->rx_ctrl.cwb;
    memcpy(p, &cwb, sizeof(cwb)); p += sizeof(cwb);

    uint8_t stbc = (uint8_t)info->rx_ctrl.stbc;
    memcpy(p, &stbc, sizeof(stbc)); p += sizeof(stbc);

#if CONFIG_IDF_TARGET_ESP32
    int8_t noise_floor = (int8_t)info->rx_ctrl.noise_floor;
#else
    /* rx_ctrl.noise_floor is a reserved bitfield on S2/S3/C3/C2 -- not
     * available on this target. */
    int8_t noise_floor = INT8_MIN;
#endif
    memcpy(p, &noise_floor, sizeof(noise_floor)); p += sizeof(noise_floor);

    memcpy(p, info->mac, 6); p += 6;
    memcpy(p, info->dmac, 6); p += 6;

    uint16_t wifi_rx_seq = info->rx_seq;
    memcpy(p, &wifi_rx_seq, sizeof(wifi_rx_seq)); p += sizeof(wifi_rx_seq);

    memcpy(p, &csi_len, sizeof(csi_len)); p += sizeof(csi_len);

    if (csi_len > 0) {
        memcpy(p, info->buf, csi_len);
    }

    return total_len;
}
