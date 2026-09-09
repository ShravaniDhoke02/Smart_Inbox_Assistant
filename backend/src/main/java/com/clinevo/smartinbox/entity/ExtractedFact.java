package com.clinevo.smartinbox.entity;

import lombok.Data;

@Data
public class ExtractedFact {

    private Long id;

    private Long messageId;

    // Group/Category, e.g., 'Patient', 'Reporter', 'Product', 'Reaction', 'Severity'
    private String factGroup;

    // E.g., 'Age', 'Name', 'Dose'
    private String fieldName;

    private String fieldValue;

    private Double confidence;

    // Traceability: e.g., "Email Body, Paragraph 2" or "Attachment 1 (PDF), Page 3"
    private String sourceReference;
}
