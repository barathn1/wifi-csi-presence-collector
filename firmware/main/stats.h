#pragma once

#include <stdint.h>

/* Ground-truth on-device counters -- read by the capacity test (via the
 * periodic "#STAT" line) to compute drop%, independent of whichever
 * transport is under test. */
typedef struct {
    uint32_t csi_cb_count;    /* CSI events generated (incremented unconditionally in the callback) */
    uint32_t enqueued;        /* successfully queued for the sender task */
    uint32_t drop_queue_full; /* dropped because the queue was full (sender too slow) */
    uint32_t drop_tx_fail;    /* dropped because the transport write failed/would-block */
} csi_stats_t;

extern csi_stats_t g_csi_stats;

/* Starts a background task that prints one
 * "#STAT,cb=..,enq=..,dqf=..,dtx=..,upt=.." line every ~2s via a single
 * fwrite() call, always available over the USB link regardless of which
 * transport carries CSI data (see firmware design notes). */
void stats_start_periodic_report(void);
