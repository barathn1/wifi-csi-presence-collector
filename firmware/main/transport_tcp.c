#include <errno.h>
#include <string.h>

#include <arpa/inet.h>
#include <sys/socket.h>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "esp_log.h"

#include "transport_tcp.h"

static const char *TAG = "transport_tcp";

static int s_sock = -1;
static SemaphoreHandle_t s_sock_mutex;
static char s_ip[16];
static uint16_t s_port;
static volatile bool s_connected = false;

static void connect_once(void)
{
    int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "socket() failed: errno %d", errno);
        vTaskDelay(pdMS_TO_TICKS(1000));
        return;
    }

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(s_port);
    inet_pton(AF_INET, s_ip, &addr.sin_addr);

    ESP_LOGI(TAG, "connecting to %s:%u ...", s_ip, s_port);
    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        ESP_LOGW(TAG, "connect failed: errno %d", errno);
        close(sock);
        vTaskDelay(pdMS_TO_TICKS(2000));
        return;
    }

    int flag = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    xSemaphoreTake(s_sock_mutex, portMAX_DELAY);
    s_sock = sock;
    s_connected = true;
    xSemaphoreGive(s_sock_mutex);

    ESP_LOGI(TAG, "connected");
}

static void connection_task(void *arg)
{
    while (1) {
        if (!s_connected) {
            connect_once();
        }
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

void transport_tcp_start(const cfg_store_t *cfg)
{
    strncpy(s_ip, cfg->laptop_ip, sizeof(s_ip) - 1);
    s_port = cfg->tcp_port;
    s_sock_mutex = xSemaphoreCreateMutex();

    xTaskCreate(connection_task, "tcp_conn", 4096, NULL, tskIDLE_PRIORITY + 1, NULL);

    while (!s_connected) {
        vTaskDelay(pdMS_TO_TICKS(200));
    }
}

bool transport_tcp_send(const uint8_t *buf, size_t len)
{
    if (!s_connected) {
        return false;
    }

    xSemaphoreTake(s_sock_mutex, portMAX_DELAY);
    int sock = s_sock;
    xSemaphoreGive(s_sock_mutex);

    if (sock < 0) {
        return false;
    }

    int sent = send(sock, buf, len, 0);
    if (sent < 0) {
        ESP_LOGW(TAG, "send failed: errno %d -- will reconnect", errno);
        xSemaphoreTake(s_sock_mutex, portMAX_DELAY);
        close(s_sock);
        s_sock = -1;
        s_connected = false;
        xSemaphoreGive(s_sock_mutex);
        return false;
    }
    return (size_t)sent == len;
}
