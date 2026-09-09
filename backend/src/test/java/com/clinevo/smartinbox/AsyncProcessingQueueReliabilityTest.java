package com.clinevo.smartinbox;

import com.clinevo.smartinbox.service.AsyncProcessingQueue;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class AsyncProcessingQueueReliabilityTest {

    @Test
    void enqueueRetryTracksRetriesAndDeadLettersAfterLimit() throws InterruptedException {
        AsyncProcessingQueue queue = new AsyncProcessingQueue();

        queue.enqueueRetry(101L, 2);
        queue.enqueueRetry(101L, 2);

        assertEquals(2, queue.currentRetryCount(101L));
        assertFalse(queue.isDeadLetter(101L));

        queue.enqueueRetry(101L, 2);

        assertTrue(queue.isDeadLetter(101L));
        assertEquals(3, queue.currentRetryCount(101L));
    }

    @Test
    void queueStillProcessesNewMessagesAfterRetryFailure() throws InterruptedException {
        AsyncProcessingQueue queue = new AsyncProcessingQueue();

        queue.enqueueRetry(202L, 2);
        assertEquals(202L, queue.dequeue());

        queue.enqueue(303L);
        assertEquals(303L, queue.dequeue());
    }
}
