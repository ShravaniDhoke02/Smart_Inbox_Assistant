package com.clinevo.smartinbox.controller;

import com.clinevo.smartinbox.repository.MessageRepository;
import com.clinevo.smartinbox.entity.Message;
import com.clinevo.smartinbox.entity.Attachment;
import com.clinevo.smartinbox.service.AuditService;
import com.clinevo.smartinbox.service.AsyncProcessingQueue;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.beans.factory.annotation.Value;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/test-data")
@RequiredArgsConstructor
public class TestDataController {

    private final MessageRepository messageRepository;
    private final AuditService audit;
    private final AsyncProcessingQueue queue;
    private final ObjectMapper objectMapper;

    @Value("${demo.fixture-dir:}")
    private String fixtureDirectory;

    private static final List<String> DEMO_FILES = List.of(
            "01_digital_safety.pdf", "02_digital_safety_table.pdf", "03_digital_quality.pdf",
            "04_digital_info_request.pdf", "05_digital_irrelevant.pdf", "06_scan_handwritten_safety.pdf",
            "07_scan_handwritten_quality.pdf", "08_article_case_1.pdf", "09_article_case_2.pdf",
            "10_article_case_3.pdf", "11_article_case_4.pdf", "12_article_case_5.pdf",
            "13_spanish_case.pdf", "14_hindi_case.pdf", "15_mixed_safety_quality.pdf");

    @PostMapping("/seed")
    public ResponseEntity<?> seedSyntheticData() {
        Path fixtureDir = fixtureDirectory == null || fixtureDirectory.isBlank()
                ? Path.of(System.getProperty("user.dir")).getParent().resolve("sample-data")
                : Path.of(fixtureDirectory);
        List<String> missing = DEMO_FILES.stream().filter(file -> !Files.isRegularFile(fixtureDir.resolve(file))).toList();
        if (!missing.isEmpty()) {
            return ResponseEntity.badRequest().body("Generate fixtures first: ai-service/generate_synthetic_fixtures.py. Missing: " + String.join(", ", missing));
        }
        List<Long> created = new ArrayList<>();
        for (String filename : DEMO_FILES) {
            Message message = new Message();
            message.setSubject("Synthetic demo - " + filename.replace(".pdf", ""));
            message.setSender("demo.sender@synthetic.example");
            message.setBody("Synthetic training fixture. All people, products, and clinical details are fictional.");
            message.setReceivedDate(LocalDateTime.now());
            message.setStatus(Message.ProcessStatus.PENDING);
            message.setSyntheticDemo(true);
            Attachment attachment = new Attachment();
            attachment.setFilename(filename);
            attachment.setFileType("application/pdf");
            attachment.setStoragePath(fixtureDir.resolve(filename).toString());
            message.setAttachments(new ArrayList<>(List.of(attachment)));
            try {
                message.setAttachmentsJson(objectMapper.writeValueAsString(message.getAttachments()));
                message.setExtractedFactsJson("[]");
            } catch (Exception error) {
                throw new IllegalStateException("Could not persist synthetic document JSON", error);
            }
            Message saved = messageRepository.save(message);
            audit.log("MESSAGE", saved.getId(), "DEMO_DATA_CREATED", "SYSTEM", "Synthetic fixture queued: " + filename);
            queue.enqueue(saved.getId());
            created.add(saved.getId());
        }
        return ResponseEntity.status(201).body(java.util.Map.of("created", created.size(), "messageIds", created));
    }

    @DeleteMapping
    public ResponseEntity<String> clearTestData() {
        var recordsToDelete = new ArrayList<>(messageRepository.findBySyntheticDemoTrue());
        messageRepository.findBySender("User Upload").stream()
                .filter(message -> !recordsToDelete.contains(message))
                .forEach(recordsToDelete::add);

        long deleted = recordsToDelete.size();
        recordsToDelete.forEach(message -> {
            if (message.getAttachments() != null) {
                message.getAttachments().forEach(attachment -> {
                    if (attachment.getStoragePath() != null) {
                        try {
                            Files.deleteIfExists(Path.of(attachment.getStoragePath()));
                        } catch (Exception ignored) {
                            // Database cleanup should still complete if a temporary file is already gone.
                        }
                    }
                });
            }
            audit.log("MESSAGE", message.getId(), "DEMO_DATA_DELETED", "SYSTEM", "Synthetic or browser-uploaded test record removed");
        });
        messageRepository.deleteAll(recordsToDelete);
        return ResponseEntity.ok("Cleared " + deleted + " test record(s).");
    }

    @org.springframework.web.bind.annotation.GetMapping("/report")
    public ResponseEntity<?> syntheticTimingReport() {
        List<Map<String, Object>> records = messageRepository.findBySyntheticDemoTrue().stream().map(message -> {
            Map<String, Object> row = new java.util.LinkedHashMap<>();
            row.put("id", message.getId());
            row.put("subject", message.getSubject());
            row.put("status", String.valueOf(message.getStatus()));
            row.put("category", String.valueOf(message.getCategory()));
            row.put("processingDurationMs", message.getProcessingDurationMs() == null ? 0L : message.getProcessingDurationMs());
            row.put("attempts", message.getProcessingAttempts() == null ? 0 : message.getProcessingAttempts());
            row.put("progress", message.getStatus() == Message.ProcessStatus.REVIEW_REQUIRED
                    || message.getStatus() == Message.ProcessStatus.COMPLETED ? 100
                    : Math.max(0, Math.min(99, message.getProcessingProgress() == null ? 0 : message.getProcessingProgress())));
            row.put("stage", message.getProcessingStage() == null ? "Queued for processing" : message.getProcessingStage());
            return row;
        }).toList();
        return ResponseEntity.ok(Map.of("syntheticOnly", true, "documents", records, "count", records.size()));
    }

    @org.springframework.web.bind.annotation.GetMapping("/status")
    public ResponseEntity<?> syntheticBatchStatus(@RequestParam List<Long> ids) {
        List<Message> records = messageRepository.findAllById(ids);
        int total = ids.size();
        int terminal = (int) records.stream().filter(message -> isTerminal(message.getStatus())).count();
        int progress = total == 0 ? 100 : (int) Math.round(records.stream()
                .mapToInt(this::processingProgress)
                .average()
                .orElse(0));
        boolean complete = total > 0 && records.size() == total && terminal == total;
        Message active = records.stream().filter(message -> !isTerminal(message.getStatus())).findFirst().orElse(null);
        String stage = complete ? "Processing complete. Refreshing results..."
                : active == null ? "Waiting for processing status..." : processingStage(active);

        return ResponseEntity.ok(Map.of(
                "total", total,
                "completed", terminal,
                "progress", complete ? 100 : Math.min(99, progress),
                "stage", stage,
                "terminal", complete));
    }

    private int processingProgress(Message message) {
        if (isTerminal(message.getStatus())) return 100;
        return Math.max(0, Math.min(99, message.getProcessingProgress() == null ? 0 : message.getProcessingProgress()));
    }

    private String processingStage(Message message) {
        if (message.getStatus() == Message.ProcessStatus.FAILED) return "Processing failed";
        return message.getProcessingStage() == null || message.getProcessingStage().isBlank()
                ? "Queued for processing" : message.getProcessingStage();
    }

    private boolean isTerminal(Message.ProcessStatus status) {
        return status == Message.ProcessStatus.REVIEW_REQUIRED
                || status == Message.ProcessStatus.COMPLETED
                || status == Message.ProcessStatus.FAILED;
    }
}
