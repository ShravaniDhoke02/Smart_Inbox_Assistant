#!/usr/bin/env python
import sys
import json
sys.path.insert(0, 'ai-service')
from main import analyze_document_payload
from pathlib import Path

# Test 1: Cardiozin case with table containing weight
pdf_path = Path("Clinevo_Synthetic_Test_Data/testdata/pdfs/normal_digital/Cardiozin_Case_Summary.pdf")
pdf_bytes = pdf_path.read_bytes()

# Create a payload similar to what the backend would send
payload = {
    "message_id": "Cardiozin_Case_Summary_Test",
    "subject": "Safety Report - Cardiozin Reaction",
    "sender": "test@example.com",
    "body": "Test case with patient reaction to Cardiozin",
    "attachments": [
        {
            "filename": str(pdf_path.name),
            "bytes": pdf_bytes,
            "content_type": "application/pdf"
        }
    ]
}

# Run the analysis
print("=" * 80)
print("ANALYZING: Cardiozin_Case_Summary.pdf")
print("=" * 80)
result = analyze_document_payload(payload)

# Extract and display key patient facts
print("\nEXTRACTED FACTS:")
print("-" * 80)
for fact in result.get("extractedFacts", []):
    if fact.get("factGroup") == "Patient":
        print(f"{fact.get('fieldName'):20} | Value: {fact.get('fieldValue'):20} | Confidence: {fact.get('confidence')}")

print("\n" + "=" * 80)
print("VERIFICATION RESULTS:")
print("=" * 80)

# Verify specific fields
facts_by_field = {(f["factGroup"], f["fieldName"]): f for f in result.get("extractedFacts", [])}

# Expected values from ground truth
expected = {
    ("Patient", "Age"): "62",
    ("Patient", "Sex"): "Female", 
    ("Patient", "Weight"): "68 kg",
    ("Patient", "Height"): "160 cm",
}

all_pass = True
for (group, field), expected_val in expected.items():
    actual = facts_by_field.get((group, field), {})
    actual_val = str(actual.get("fieldValue", "")).lower().strip()
    expected_val_lower = str(expected_val).lower().strip()
    
    status = "✓ PASS" if actual_val == expected_val_lower else "✗ FAIL"
    print(f"{status} | {field:15} | Expected: {expected_val:15} | Got: {actual.get('fieldValue', 'NOT FOUND')}")
    
    if actual_val != expected_val_lower:
        all_pass = False

print("\n" + "=" * 80)
if all_pass:
    print("ALL TESTS PASSED ✓")
else:
    print("SOME TESTS FAILED ✗")
print("=" * 80)
