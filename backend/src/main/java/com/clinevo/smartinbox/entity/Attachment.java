package com.clinevo.smartinbox.entity;

import lombok.Data;

@Data
public class Attachment {

    private Long id;

    private Long messageId;

    private String filename;
    private String fileType;
    
    // E.g., 'Normal', 'Scanned', 'Article', 'Non-English'
    private String detectedPdfType;

    // Path where the file is stored locally or in blob storage
    private String storagePath;
}
