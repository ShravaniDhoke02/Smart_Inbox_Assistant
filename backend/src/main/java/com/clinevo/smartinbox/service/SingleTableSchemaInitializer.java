package com.clinevo.smartinbox.service;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

/** Removes the former child tables after the document JSON columns are available. */
@Component
@RequiredArgsConstructor
public class SingleTableSchemaInitializer {
    private final JdbcTemplate jdbc;

    @Value("${persistence.single-table-cleanup:true}")
    private boolean cleanup;

    @jakarta.annotation.PostConstruct
    void removeLegacyTables() {
        if (!cleanup) return;
        drop("attachments");
        drop("extracted_facts");
        drop("audit_logs");
    }

    private void drop(String table) {
        try {
            jdbc.execute("DROP TABLE " + table + " CASCADE CONSTRAINTS");
        } catch (Exception ignored) {
            // The table may already be absent or the database may not support this syntax.
        }
    }
}
