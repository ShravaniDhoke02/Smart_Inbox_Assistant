# Synthetic Test Dataset — Smart Inbox Assistant (Clinevo Assignment)

**All data in this folder is 100% fictional/synthetic.** No real patients, reporters, clinicians,
hospitals, or products are represented. Product names (Cardiozin, Fevrolix, Neurotab XR),
people, and institutions are invented for this exercise only. This satisfies the assignment's
"no real patient data — ever" rule (Section 3E / Section 6).

## Folder structure

```
testdata/
├── emails/                 15 .eml-style text files (headers + body)
├── pdfs/
│   ├── normal_digital/      5 PDFs — real text layer, tables, straightforward forms
│   ├── scanned_handwritten/ 2 PDFs — image-only pages (handwriting font + paper noise,
│   │                         no text layer) to force OCR / vision-model handling
│   ├── articles/             5 PDFs — two-column "published article" layout with
│   │                         Abstract / Case / Discussion / References sections
│   └── non_english/          2 PDFs — one Spanish, one French case report
└── ground_truth.json        Expected classification + key extracted fields per item,
                              for scoring your pipeline's output against
```

Total: 15 emails + 14 PDFs = 29 documents (comfortably over the "10-15 sample documents"
batch-processing requirement in Section 3E).

## Coverage checklist vs. assignment Section 6

| # | Requirement | Where |
|---|---|---|
| 1 | ≥10 emails, varying detail about a reaction | `emails/01`–`05`, `12` (6 reaction-focused, ranging from a fully detailed physician report to a vague, low-detail consumer message) |
| 2 | ≥5 normal digital PDF attachments | `pdfs/normal_digital/*.pdf` |
| 3 | ≥2 scanned/handwritten-style PDFs | `pdfs/scanned_handwritten/*.pdf` |
| 4 | ≥5 fictional "article" PDFs | `pdfs/articles/*.pdf` |
| 5 | ≥2 non-English PDFs, case-relevant | `pdfs/non_english/*.pdf` (Spanish, French) |
| 6 | ≥2 quality-complaint-only + ≥2 info-request-only | `emails/06,07` (PQC-only); `emails/08,09` (MI-only) |
| 7 | ≥1 clearly irrelevant example | `emails/10` (marketing), `emails/11` (internal admin) |

Bonus coverage for Section 4 (literature screening): `emails/13` forwards an article PDF
independent of a patient email, and `pdfs/articles/Journal_Article_TwoCase_Series.pdf` and
`Journal_Article_General_Review_NoCase.pdf` are specifically designed to test edge cases —
an article with **two distinct patient cases to split apart**, and an article with **no
identifiable patient case at all** (should be screened out as not reportable).

## Email-by-email intent (what each is testing)

| File | Intended category(ies) | Why it's a useful test case |
|---|---|---|
| 01_safety_report_detailed | Safety Report (ICSR) | High-detail: named reporter, patient demographics, drug/dose/dates, reaction, hospitalization — a "clean" positive case |
| 02_safety_report_minimal | Safety Report (ICSR) | Low detail, no reporter role/country stated, informal tone — tests graceful "Not stated" handling |
| 03_safety_report_with_attachment | Safety Report (ICSR) | References a forwarded scanned handwritten attachment — tests email+PDF fusion |
| 04_safety_report_followup | Safety Report (ICSR) | Follow-up/update to case 01 — tests linking follow-up correspondence to an existing case |
| 05_safety_report_vague | Safety Report (ICSR), low confidence | Extremely vague (no drug name, no clear patient) — should get a **low confidence score**, not a confident guess |
| 06_quality_complaint_1 | Quality Complaint (PQC) only | Clear packaging defect, explicitly states no patient exposure/reaction |
| 07_quality_complaint_2 | Quality Complaint (PQC) only | Physical defect (cloudy syrup), explicitly not consumed / no reaction |
| 08_info_request_1 | Info Request (MI) only | Dosing/interaction question, explicitly "no adverse event to report" |
| 09_info_request_2 | Info Request (MI) only | Casual consumer dosing/food-interaction question |
| 10_marketing_irrelevant | Not Relevant | Conference marketing newsletter |
| 11_internal_admin_irrelevant | Not Relevant | Internal facilities email, not even about a product |
| 12_mixed_safety_and_quality | Safety Report **and** Quality Complaint (dual label) | Rash reaction *and* tablet discoloration mentioned — tests multi-label output |
| 13_literature_article_forward | Safety Report (pending article review) / literature screening | Forwards a case-report PDF for literature screening (Section 4 bonus) |
| 14_non_english_reaction_spanish | Safety Report (ICSR) | Spanish-language body + Spanish PDF attachment — tests language detection |
| 15_non_english_reaction_french | Safety Report (ICSR) | French-language body + French PDF attachment — tests language detection |

## PDF-by-PDF intent

**normal_digital/** — real selectable text, structured label/value forms, at least one with a
genuine table (lab results, dosing schedule) to test table extraction into rows/columns.

**scanned_handwritten/** — rendered as flattened images (no text layer) using a cursive font,
paper-grain noise, slight rotation, and ink-smudge artifacts, so a naive `pdftotext` will
return nothing and the pipeline is forced to use OCR/vision. Deliberately includes a couple of
ambiguous/hard-to-read words to justify a confidence score below 100%.

**articles/** — two real newspaper-style columns per page (built with a column-flowing layout,
not just narrow paragraphs), an Abstract, a References section (should be ignored when pulling
out the case), and:
- 3 articles with exactly one clear patient case each (Cardiozin, Neurotab, Fevrolix)
- 1 article that is a general review with **no individual patient case** (should be flagged
  as *not* a reportable case despite mentioning the products)
- 1 article containing **two separate patient cases** in one PDF (tests case-splitting)

**non_english/** — same ICSR structure as the normal_digital forms, but entirely in Spanish or
French, so the pipeline must detect the language and either translate or extract natively.

## Ground truth

See `ground_truth.json` for a machine-readable answer key (expected category labels and key
fields) you can diff your pipeline's output against when building your evaluation report.

## Note on "10 emails describing a reaction"

The assignment's Section 6 item 1 says the 10 reaction emails should have "varying levels of
detail" — items 01, 02, 04, 05, 12, 14, 15 (seven) are reaction-centric with deliberately
uneven detail; 03 and 13 also involve a reaction but are primarily testing attachment/forward
handling. If you want a strict 10, treat 01–05 + 12–15 (9) plus a duplicate variant of 02 with
different wording as your 10th — or simply note in your write-up that quality/info/irrelevant
emails were kept as a separate, explicitly-required set per Section 6 items 6–7 rather than
folded into the "10."
