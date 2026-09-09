package com.clinevo.smartinbox.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;
import jakarta.persistence.Transient;

import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Entity
@Table(name = "messages")
@Data
public class Message {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "case_id", unique = true, length = 32)
    private String caseId;

    @Transient
    private String sourcePdf;

    @Transient
    private String casePatient;

    @Transient
    private String caseProduct;

    @Transient
    private String caseReaction;

    private String subject;
    private String sender;

    @Column(name = "external_message_id", unique = true, length = 512)
    private String externalMessageId;
    
    @Column(name = "received_date")
    private LocalDateTime receivedDate;
    
    @Column(columnDefinition = "CLOB")
    private String body;
    
    @Enumerated(EnumType.STRING)
    private ProcessStatus status = ProcessStatus.PENDING;
    
    // Can be multiple comma-separated, e.g. "Safety Report, Quality Complaint"
    private String category;
    
    @Column(name = "confidence_score")
    private Double confidenceScore;
    
    @Column(name = "classification_reason", length = 1000)
    private String classificationReason;
    
    @Column(name = "ai_summary", length = 2000)
    private String aiSummary;

    @Column(length = 64)
    private String language;

    @Lob
    private String translatedText;

    @Column(length = 1000)
    private String pdfTypes;

    @Column(length = 2000)
    private String imageNotes;

    @Lob
    private String tables;

    /** JSON persisted so the reviewer sees the same AI evidence after a reload. */
    @Lob
    private String literatureScreening;

    @Lob
    private String ocrConfidence;

    /** JSON persisted so source/provenance information remains auditable. */
    @Lob
    private String sourceTrace;

    /** Explicit marker used to ensure the demo-data cleanup endpoint never deletes real inbox data. */
    @Column(name = "synthetic_demo")
    private Boolean syntheticDemo = false;
    
    @Column(name = "created_at")
    private LocalDateTime createdAt = LocalDateTime.now();
    
    @Column(name = "updated_at")
    private LocalDateTime updatedAt = LocalDateTime.now();

    @Column(name = "processing_duration_ms")
    private Long processingDurationMs;

    @Column(name = "processing_attempts")
    private Integer processingAttempts = 0;

    @Column(name = "last_processing_error", length = 1000)
    private String lastProcessingError;

    /** Persisted worker state so clients can display real, restart-safe processing progress. */
    @Column(name = "processing_progress")
    private Integer processingProgress = 0;

    @Column(name = "processing_stage", length = 255)
    private String processingStage = "Queued for processing";

    /** JSON document payloads keep this table self-contained. */
    @Lob
    @Column(name = "attachments_json")
    private String attachmentsJson;

    @Lob
    @Column(name = "extracted_facts_json")
    private String extractedFactsJson;

    @Lob
    @Column(name = "audit_logs_json")
    private String auditLogsJson;

    @Transient
    private List<Attachment> attachments;

    @Transient
    private List<ExtractedFact> extractedFacts;

    public enum ProcessStatus {
        PENDING, PROCESSING, REVIEW_REQUIRED, COMPLETED, FAILED
    }
}
