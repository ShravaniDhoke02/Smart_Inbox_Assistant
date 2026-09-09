#!/usr/bin/env python
import sys
sys.path.insert(0, 'ai-service')
from main import extract_pdf_text, extract_facts
from pathlib import Path

# Test on actual PDF
pdf_path = Path("Clinevo_Synthetic_Test_Data/testdata/pdfs/normal_digital/Cardiozin_Case_Summary.pdf")
pdf_bytes = pdf_path.read_bytes()

# Extract text from PDF
print("=" * 80)
print("Extracting text from PDF...")
print("=" * 80)
attachment_text, pdf_type = extract_pdf_text(pdf_bytes, str(pdf_path.name))

# Print first 2000 chars of extracted text
print(f"\nPDF Type: {pdf_type}")
print(f"Extracted text (first 2000 chars):")
print("-" * 80)
print(attachment_text[:2000])
print("-" * 80)

# Now run extraction
print("\n" + "=" * 80)
print("Running extract_facts on PDF text...")
print("=" * 80)

facts = extract_facts(
    subject="Safety Report",
    body="",
    attachment_text=attachment_text,
    category_name="Safety Report (ICSR)"
)

# Display Patient facts
print("\nPATIENT FACTS:")
print("-" * 80)
for fact in facts:
    if fact.get("factGroup") == "Patient":
        print(f"{fact.get('fieldName'):20} | {fact.get('fieldValue'):20} | Conf: {fact.get('confidence')}")

# Verify specific fields
print("\n" + "=" * 80)
print("VERIFICATION:")
print("=" * 80)

facts_by_field = {(f["factGroup"], f["fieldName"]): f for f in facts}

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
