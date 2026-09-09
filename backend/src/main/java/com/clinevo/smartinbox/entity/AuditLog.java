package com.clinevo.smartinbox.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class AuditLog {

    private Long id;

    // e.g., "MESSAGE", "EXTRACTED_FACT"
    private String entityType;

    private Long entityId;

    // e.g., "CREATED", "UPDATED", "REVIEWED_APPROVED", "REVIEWED_MODIFIED"
    private String action;

    // e.g., "SYSTEM", "REVIEWER_1"
    private String performedBy;

    private LocalDateTime timestamp = LocalDateTime.now();

    private String details;
}
