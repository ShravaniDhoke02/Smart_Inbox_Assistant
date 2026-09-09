package com.clinevo.smartinbox.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;

@Component
@Slf4j
public class AsyncProcessingQueue {

    private final BlockingQueue<Long> queue = new LinkedBlockingQueue<>();
    private final Map<Long, Integer> retryCounts = new ConcurrentHashMap<>();
    private final Map<Long, Boolean> deadLetters = new ConcurrentHashMap<>();
    private final Set<Long> failedMessages = ConcurrentHashMap.newKeySet();

    public void enqueue(Long messageId) {
        retryCounts.remove(messageId);
        deadLetters.remove(messageId);
        failedMessages.remove(messageId);
        queue.offer(messageId);
        log.info("Message ID {} added to processing queue.", messageId);
    }

    public void enqueueRetry(Long messageId, int maxRetries) {
        int count = retryCounts.getOrDefault(messageId, 0) + 1;
        retryCounts.put(messageId, count);

        if (count > maxRetries) {
            deadLetters.put(messageId, true);
            failedMessages.add(messageId);
            log.warn("Message ID {} exceeded retry limit ({}). Sent to dead-letter queue.", messageId, maxRetries);
            return;
        }

        queue.offer(messageId);
        log.warn("Message ID {} re-queued for retry {} of {}.", messageId, count, maxRetries);
    }

    public Long dequeue() throws InterruptedException {
        return queue.take();
    }

    public int size() {
        return queue.size();
    }

    public int currentRetryCount(Long messageId) {
        return retryCounts.getOrDefault(messageId, 0);
    }

    public boolean isDeadLetter(Long messageId) {
        return Boolean.TRUE.equals(deadLetters.get(messageId));
    }

    public boolean isFailed(Long messageId) {
        return failedMessages.contains(messageId);
    }
}
