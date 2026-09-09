#include <stdio.h>

#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "stats.h"

csi_stats_t g_csi_stats = {0};

static void stats_task(void *arg)
{
    char line[96];
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(2000));
        int n = snprintf(line, sizeof(line), "#STAT,cb=%u,enq=%u,dqf=%u,dtx=%u,upt=%lld\n",
                          (unsigned)g_csi_stats.csi_cb_count, (unsigned)g_csi_stats.enqueued,
                          (unsigned)g_csi_stats.drop_queue_full, (unsigned)g_csi_stats.drop_tx_fail,
                          (long long)(esp_timer_get_time() / 1000));
        /* Single fwrite call so the whole line lands atomically with
         * respect to other console writers (ESP-IDF's vfs console
         * driver serializes individual write() calls). */
        fwrite(line, 1, n, stdout);
        fflush(stdout);
    }
}

void stats_start_periodic_report(void)
{
    xTaskCreate(stats_task, "stats", 3072, NULL, tskIDLE_PRIORITY + 1, NULL);
}
