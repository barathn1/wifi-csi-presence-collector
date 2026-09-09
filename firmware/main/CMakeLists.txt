idf_component_register(
    SRCS "app_main.c" "wifi_sta.c" "csi_capture.c" "cfg_store.c"
         "transport_tcp.c" "transport_serial.c" "wire_format.c" "stats.c"
         "tx_sender.c" "ap_hotspot.c"
    INCLUDE_DIRS "."
)
