# Smart Inbox Assistant - Engineering Write-up

## Purpose

This is a synthetic-data-only prototype for triaging a shared pharmacovigilance mailbox. It automates the first pass: intake, document understanding, multi-label classification, evidence-linked fact extraction, and reviewer approval. It is not a clinical decision system and must not be used with real patient data.

## Architecture and choices

Spring Boot is the system of record. It polls an IMAP mailbox when enabled, persists messages, attachments, facts, status, timings, and audit events, and places messages on an in-process queue. A Flask service receives the queued payload and performs PDF extraction, OCR fallback, language detection/translation, classification, structured fact extraction, and optional Gemini analysis. Oracle is used in Docker; H2 supports local development. The reviewer experience is served with the backend and presents a PDF viewer, editable fact list, source evidence, approval action, and audit timeline.

The service boundary keeps document-model experimentation out of the workflow/database layer. The in-process queue is deliberately small for this assignment; a production implementation would replace it with a durable broker and dead-letter/retry policy.

## Prompting and extraction approach

The model instruction requires JSON-only output, the four allowed categories, confidence values, source references, multi-label classification, and `Not stated` rather than invention. The backend and Python fallback do not trust omission: Safety reports are normalized against a mandatory schema so every required Patient, Reporter, Product, Reaction, Severity, and Narrative field is returned. Missing values use `Not stated`, confidence `0.0`, and an explicit absence reference.

Facts are presented with source evidence. Direct email evidence uses a sentence reference and matched text. PDF evidence uses page markers preserved during extraction and a matched-text excerpt. This is sufficient for a prototype reviewer but would be replaced in production by stable character offsets/bounding boxes and source-document hashes.

## Synthetic evaluation corpus

`generate_synthetic_fixtures.py` deterministically creates 15 fictional PDFs: Safety, Quality, MI, irrelevant, mixed Safety + Quality, tables, scans, five article-like documents, and non-English cases. The UI's **Load Synthetic Demo** action queues the corpus and the report endpoint exposes per-document processing duration. Every seeded record has a `syntheticDemo` marker, which prevents the cleanup endpoint from deleting ordinary records.

## Known limitations and production changes

OCR quality varies significantly for handwriting, and the prototype must surface that uncertainty for human review. Keyword/regex fallbacks are intentionally conservative but are not clinically validated. Article case extraction and table handling are useful demonstrations, not a substitute for document-layout models. Image flags are review prompts rather than diagnostic image analysis.

Production work would add durable asynchronous processing, encrypted object storage, malware scanning, strict tenancy/RBAC, immutable audit records, model-output schema validation, data-retention policies, PHI-safe deployment, translation/data-processing agreements, human quality control, and formal validation with pharmacovigilance specialists.
