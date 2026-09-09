package com.clinevo.smartinbox.controller;

import com.clinevo.smartinbox.entity.Attachment;
import com.clinevo.smartinbox.entity.ExtractedFact;
import com.clinevo.smartinbox.entity.Message;
import com.clinevo.smartinbox.repository.MessageRepository;
import com.clinevo.smartinbox.entity.AuditLog;
import com.clinevo.smartinbox.service.AsyncProcessingQueue;
import com.clinevo.smartinbox.service.AuditService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.ContentDisposition;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.transaction.annotation.Transactional;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.util.List;
import java.util.ArrayList;
import java.util.UUID;
import java.util.Objects;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.Locale;

@RestController
@RequestMapping("/api/messages")
@RequiredArgsConstructor
public class MessageController {

    private final MessageRepository messageRepository;
    private final AuditService audit;
    private final AsyncProcessingQueue queue;
    private final ObjectMapper objectMapper;
    private final RestTemplate http = new RestTemplate();
    @Value("${ai.service.url:http://localhost:8000/analyze}")
    private String aiUrl;

    @GetMapping
public ResponseEntity<List<Message>> getPendingMessages() {

    List<Message> messages =
            messageRepository.findByStatusOrderByReceivedDateDesc(Message.ProcessStatus.REVIEW_REQUIRED);

    messages.forEach(message -> { restoreJson(message); enrichCaseDisplayData(message); });

    return ResponseEntity.ok(messages);
}

@GetMapping("/{id}/export")
@Transactional(readOnly = true)
public ResponseEntity<byte[]> exportMessage(@PathVariable Long id) {
    return messageRepository.findById(id)
            .map(message -> {
                restoreJson(message);
                enrichCaseDisplayData(message);
                deduplicateFacts(message);
                Map<String, Object> export = new LinkedHashMap<>();
                export.put("documentId", message.getId());
                export.put("documentReference", message.getSourcePdf());
                export.put("extractedFacts", message.getExtractedFacts() == null ? List.of() : message.getExtractedFacts().stream()
                        .map(fact -> {
                            Map<String, Object> data = new LinkedHashMap<>();
                            data.put("factGroup", fact.getFactGroup());
                            data.put("fieldName", fact.getFieldName());
                            data.put("fieldValue", fact.getFieldValue());
                            data.put("confidence", fact.getConfidence());
                            data.put("sourceReference", fact.getSourceReference());
                            return data;
                        })
                        .toList());

                try {
                    byte[] json = objectMapper.writerWithDefaultPrettyPrinter().writeValueAsBytes(export);
                    HttpHeaders headers = new HttpHeaders();
                    headers.setContentType(MediaType.APPLICATION_JSON);
                    headers.setContentDisposition(ContentDisposition.attachment()
                            .filename("message-" + id + "-extracted-data.json")
                            .build());
                    return ResponseEntity.ok().headers(headers).body(json);
                } catch (JsonProcessingException error) {
                    throw new IllegalStateException("Could not create JSON export", error);
                }
            })
            .orElse(ResponseEntity.notFound().build());
}

@GetMapping("/history")
public ResponseEntity<List<Message>> getHistory() {
    List<Message> messages =
            messageRepository.findAllByOrderByReceivedDateDesc();

    messages.forEach(message -> { restoreJson(message); enrichCaseDisplayData(message); });

    return ResponseEntity.ok(messages);
}

@GetMapping("/{id}")
public ResponseEntity<Message> getMessage(@PathVariable Long id) {
    return messageRepository.findById(id)
            .map(message -> {
                restoreJson(message);
                deduplicateFacts(message);
                enrichCaseDisplayData(message);
                return ResponseEntity.ok(message);
            })
            .orElse(ResponseEntity.notFound().build());
}
private void enrichCaseDisplayData(Message message) {

    // Generate a stable Case ID from the database ID
    if (message.getCaseId() == null || message.getCaseId().isBlank()) {
        message.setCaseId(String.format("CASE-%06d", message.getId()));
        messageRepository.save(message);
    }

    // Source PDF
    if (message.getAttachments() != null && !message.getAttachments().isEmpty()) {
        Attachment attachment = message.getAttachments().get(0);
        if (attachment != null && attachment.getFilename() != null) {
            message.setSourcePdf(attachment.getFilename());
        }
    }

    // Extract patient, product and reaction
    if (message.getExtractedFacts() != null) {
        for (ExtractedFact fact : message.getExtractedFacts()) {

            String group = fact.getFactGroup() == null
                    ? ""
                    : fact.getFactGroup().trim().toLowerCase(Locale.ROOT);

            String field = fact.getFieldName() == null
                    ? ""
                    : fact.getFieldName().trim().toLowerCase(Locale.ROOT);

            String value = fact.getFieldValue();

            if (value == null || value.isBlank()) {
                continue;
            }

            if (group.contains("patient")
                    && (field.contains("initial")
                    || field.contains("patient id")
                    || field.contains("patient name"))) {
                message.setCasePatient(value);
            }

            if (group.contains("product")
                    && (field.contains("product name")
                    || field.equals("product"))) {
                message.setCaseProduct(value);
            }

            if (group.contains("reaction")
                    && (field.contains("description")
                    || field.contains("reaction"))) {
                message.setCaseReaction(value);
            }
        }
    }

    // Safe fallbacks
    if (message.getSourcePdf() == null) {
        message.setSourcePdf("Not stated");
    }

    if (message.getCasePatient() == null) {
        message.setCasePatient("Not stated");
    }

    if (message.getCaseProduct() == null) {
        message.setCaseProduct("Not stated");
    }

    if (message.getCaseReaction() == null) {
        message.setCaseReaction("Not stated");
    }
}

    private Message deduplicateFacts(Message message) {
        if (message.getExtractedFacts() == null) return message;
        Map<String, ExtractedFact> uniqueFacts = new LinkedHashMap<>();
        for (ExtractedFact fact : message.getExtractedFacts()) {
                String key = canonicalFactKey(fact);
            ExtractedFact existing = uniqueFacts.get(key);
            if (existing == null || isLessUseful(existing, fact)) uniqueFacts.put(key, fact);
        }
        message.setExtractedFacts(new ArrayList<>(uniqueFacts.values()));
        return message;
    }

    private String canonicalFactKey(ExtractedFact fact) {
        String group = String.valueOf(fact.getFactGroup()).trim().toLowerCase(Locale.ROOT);
        String field = String.valueOf(fact.getFieldName()).trim().toLowerCase(Locale.ROOT);
        if (group.equals("patient information")) group = "patient";
        if (field.equals("patient initials") || field.equals("patient id") || field.equals("patient name")) {
            field = "initials";
        } else if (field.startsWith("patient ")) {
            field = field.substring("patient ".length());
        }
        return group + "\u0000" + field;
    }

    private boolean isLessUseful(ExtractedFact existing, ExtractedFact candidate) {
        boolean existingFallback = "Not stated".equalsIgnoreCase(String.valueOf(existing.getFieldValue()).trim());
        boolean candidateFallback = "Not stated".equalsIgnoreCase(String.valueOf(candidate.getFieldValue()).trim());
        return existingFallback && !candidateFallback
                || !existingFallback && !candidateFallback
                && candidate.getConfidence() != null
                && (existing.getConfidence() == null || candidate.getConfidence() > existing.getConfidence());
    }

    @GetMapping("/{id}/processing-status")
    public ResponseEntity<Map<String, Object>> getProcessingStatus(@PathVariable Long id) {
        return messageRepository.findById(id)
                .map(message -> ResponseEntity.ok(Map.<String, Object>of(
                        "id", message.getId(),
                        "status", message.getStatus().name(),
                        "progress", processingProgress(message),
                        "stage", processingStage(message),
                        "terminal", isTerminal(message.getStatus()))))
                .orElse(ResponseEntity.notFound().build());
    }

    private int processingProgress(Message message) {
        if (isTerminal(message.getStatus())) return 100;
        return Math.max(0, Math.min(99, message.getProcessingProgress() == null ? 0 : message.getProcessingProgress()));
    }

    private String processingStage(Message message) {
        if (message.getStatus() == Message.ProcessStatus.REVIEW_REQUIRED) return "Analysis complete — ready for review";
        if (message.getStatus() == Message.ProcessStatus.COMPLETED) return "Review complete";
        if (message.getStatus() == Message.ProcessStatus.FAILED) return "Processing failed";
        return message.getProcessingStage() == null || message.getProcessingStage().isBlank()
                ? "Queued for processing" : message.getProcessingStage();
    }

    private boolean isTerminal(Message.ProcessStatus status) {
        return status == Message.ProcessStatus.REVIEW_REQUIRED
                || status == Message.ProcessStatus.COMPLETED
                || status == Message.ProcessStatus.FAILED;
    }

    @GetMapping("/{id}/audit")
    public ResponseEntity<List<AuditLog>> getAuditTrail(@PathVariable Long id) {
        return messageRepository.findById(id).map(message -> { restoreJson(message); return ResponseEntity.ok(auditLogs(message)); }).orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/{id}/attachments/{attachmentId}/content")
    public ResponseEntity<Resource> viewAttachment(@PathVariable Long id, @PathVariable Long attachmentId) {
        Message message = messageRepository.findById(id).orElse(null);
        if (message == null) return ResponseEntity.notFound().build();
        restoreJson(message);
        Attachment attachment = message.getAttachments() == null ? null : message.getAttachments().stream().filter(item -> attachmentId.equals(item.getId())).findFirst().orElse(null);
        if (attachment == null
                || attachment.getStoragePath() == null || attachment.getStoragePath().isBlank()) {
            return ResponseEntity.notFound().build();
        }

        Path file = Path.of(attachment.getStoragePath()).normalize();
        if (!Files.isRegularFile(file)) {
            return ResponseEntity.notFound().build();
        }

        MediaType contentType = MediaType.APPLICATION_PDF;
        try {
            String detectedType = Files.probeContentType(file);
            if (detectedType != null) {
                contentType = MediaType.parseMediaType(detectedType);
            }
        } catch (IOException ignored) {
            // The stored PDF remains safe to serve with the default content type.
        }

        return ResponseEntity.ok()
                .contentType(contentType)
                .header(HttpHeaders.CONTENT_DISPOSITION, ContentDisposition.inline()
                        .filename(attachment.getFilename())
                        .build()
                        .toString())
                .body(new FileSystemResource(file));
    }

    @PutMapping("/{id}")
    public ResponseEntity<Message> updateMessage(@PathVariable Long id, @RequestBody Message updatedMessage) {
        return messageRepository.findById(id)
                .map(existingMessage -> {
                    restoreJson(existingMessage);
                    boolean classificationChanged = !Objects.equals(existingMessage.getCategory(), updatedMessage.getCategory())
                            || !Objects.equals(existingMessage.getConfidenceScore(), updatedMessage.getConfidenceScore());
                    boolean summaryChanged = !Objects.equals(existingMessage.getAiSummary(), updatedMessage.getAiSummary());
                    // Reviewer approves or modifies
                    existingMessage.setCategory(updatedMessage.getCategory());
                    existingMessage.setConfidenceScore(updatedMessage.getConfidenceScore());
                    existingMessage.setAiSummary(updatedMessage.getAiSummary());
                    
                    // In a real app we'd update facts individually, here we replace them or assume they are managed
                    if (updatedMessage.getExtractedFacts() != null) {
                        if (existingMessage.getExtractedFacts() == null) {
                            existingMessage.setExtractedFacts(new ArrayList<>());
                        }
                        existingMessage.getExtractedFacts().clear();
                        Map<String, ExtractedFact> uniqueFacts = new LinkedHashMap<>();
                        updatedMessage.getExtractedFacts().forEach(updatedFact -> {
                            ExtractedFact fact = new ExtractedFact();
                            fact.setMessageId(existingMessage.getId());
                            fact.setFactGroup(updatedFact.getFactGroup());
                            fact.setFieldName(updatedFact.getFieldName());
                            fact.setFieldValue(updatedFact.getFieldValue());
                            fact.setConfidence(updatedFact.getConfidence());
                            fact.setSourceReference(updatedFact.getSourceReference());
                                String key = canonicalFactKey(fact);
                            uniqueFacts.put(key, fact);
                        });
                        existingMessage.getExtractedFacts().addAll(uniqueFacts.values());
                        persistJson(existingMessage);
                        audit.log("MESSAGE", id, "FACTS_REVIEWED", "REVIEWER", "Reviewer saved " + uniqueFacts.size() + " extracted facts with their source references.");
                    }
                    
                    existingMessage.setStatus(Message.ProcessStatus.COMPLETED); // Mark as reviewed
                    existingMessage.setProcessingProgress(100);
                    existingMessage.setProcessingStage("Review complete");
                    Message saved = messageRepository.save(existingMessage);
                    audit.log("MESSAGE", id, "REVIEWED", "REVIEWER", "Reviewer approved or edited classification and extracted facts");
                    if (classificationChanged) audit.log("MESSAGE", id, "CLASSIFICATION_OVERRIDDEN", "REVIEWER", "Reviewer changed classification or confidence before approval.");
                    if (summaryChanged) audit.log("MESSAGE", id, "SUMMARY_EDITED", "REVIEWER", "Reviewer edited the AI-generated summary before approval.");
                    return ResponseEntity.ok(saved);
                })
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping("/translate")
    public ResponseEntity<?> translate(@RequestBody java.util.Map<String, String> request) {
        String text = request.getOrDefault("text", "");
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        return ResponseEntity.ok(http.postForObject(
                aiUrl.replace("/analyze", "/translate"),
                new HttpEntity<>(java.util.Map.of("text", text), headers),
                Object.class));
    }

    @PostMapping("/upload")
    public ResponseEntity<?> uploadPdf(@RequestParam("file") MultipartFile file) {
        if (file == null || file.isEmpty()) {
            return ResponseEntity.badRequest().body("Please select a PDF file to upload.");
        }

        String originalFilename = StringUtils.cleanPath(file.getOriginalFilename() == null ? "uploaded-document.pdf" : file.getOriginalFilename());
        if (!originalFilename.toLowerCase().endsWith(".pdf")) {
            return ResponseEntity.badRequest().body("Only PDF files are supported.");
        }

        try {
            Path uploadDir = Paths.get(System.getProperty("java.io.tmpdir"), "smartinbox-uploads");
            Files.createDirectories(uploadDir);

            String storedName = UUID.randomUUID() + "_" + originalFilename;
            Path target = uploadDir.resolve(storedName);
            Files.copy(file.getInputStream(), target, StandardCopyOption.REPLACE_EXISTING);

            Message message = new Message();
            message.setSubject(originalFilename.replaceFirst("(?i)\\.pdf$", ""));
            message.setSender("User Upload");
            message.setBody("Uploaded PDF from the browser. Please review this document for healthcare inbox classification.");
            message.setReceivedDate(LocalDateTime.now());
            message.setStatus(Message.ProcessStatus.PENDING);
            message.setProcessingProgress(0);
            message.setProcessingStage("Queued for AI analysis");
            message.setAttachments(List.of());

            Attachment attachment = new Attachment();
            attachment.setFilename(originalFilename);
            attachment.setFileType(file.getContentType() != null ? file.getContentType() : "application/pdf");
            attachment.setDetectedPdfType("Uploaded PDF");
            attachment.setStoragePath(target.toString());
            message.setAttachments(List.of(attachment));
            Message saved = messageRepository.save(message);
            attachment.setId(saved.getId());
            attachment.setMessageId(saved.getId());
            persistJson(saved);
            messageRepository.save(saved);
            queue.enqueue(saved.getId());
            audit.log("MESSAGE", saved.getId(), "USER_UPLOAD", "USER", "PDF uploaded from front-end");

            return ResponseEntity.status(HttpStatus.CREATED).body(saved);
        } catch (IOException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Could not save uploaded PDF: " + e.getMessage());
        }
    }

    private void restoreJson(Message message) {
        try {
            if (message.getAttachmentsJson() != null) message.setAttachments(objectMapper.readValue(message.getAttachmentsJson(), objectMapper.getTypeFactory().constructCollectionType(List.class, Attachment.class)));
            if (message.getExtractedFactsJson() != null) message.setExtractedFacts(objectMapper.readValue(message.getExtractedFactsJson(), objectMapper.getTypeFactory().constructCollectionType(List.class, ExtractedFact.class)));
        } catch (IOException error) {
            throw new IllegalStateException("Could not read document JSON", error);
        }
    }

    private void persistJson(Message message) {
        try {
            message.setAttachmentsJson(objectMapper.writeValueAsString(message.getAttachments() == null ? List.of() : message.getAttachments()));
            message.setExtractedFactsJson(objectMapper.writeValueAsString(message.getExtractedFacts() == null ? List.of() : message.getExtractedFacts()));
        } catch (JsonProcessingException error) {
            throw new IllegalStateException("Could not write document JSON", error);
        }
    }

    private List<AuditLog> auditLogs(Message message) {
        try {
            if (message.getAuditLogsJson() == null) return new ArrayList<>();
            return objectMapper.readValue(message.getAuditLogsJson(), objectMapper.getTypeFactory().constructCollectionType(List.class, AuditLog.class));
        } catch (IOException error) {
            throw new IllegalStateException("Could not read audit JSON", error);
        }
    }
}
