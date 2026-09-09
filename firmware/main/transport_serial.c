#include <stdio.h>

#include "transport_serial.h"

void transport_serial_start(const cfg_store_t *cfg)
{
    (void)cfg; /* no setup needed -- stdout is already the console/USB link */
}

bool transport_serial_send(const uint8_t *buf, size_t len)
{
    /* Single fwrite call so a whole frame lands atomically with respect
     * to other console writers (see stats.c for the same reasoning). */
    size_t written = fwrite(buf, 1, len, stdout);
    fflush(stdout);
    return written == len;
}
