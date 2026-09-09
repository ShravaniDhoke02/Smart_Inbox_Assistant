package com.clinevo.smartinbox.dto;

import lombok.Data;
import java.util.List;

@Data
public class AIResponse {
    private String category;
    private Double confidenceScore;
    private String classificationReason;
    private String aiSummary;
    private String language;
    private String translatedText;
    private List<String> pdfTypes;
    private List<String> imageNotes;
    private Object tables;
    private Object literatureScreening;
    private Object ocrConfidence;
    private Object sourceTrace;
    private List<ExtractedFactDto> extractedFacts;
    
    @Data
    public static class ExtractedFactDto {
        private String factGroup;
        private String fieldName;
        private String fieldValue;
        private Double confidence;
        private String sourceReference;
    }
}
