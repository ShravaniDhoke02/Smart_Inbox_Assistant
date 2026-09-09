#!/usr/bin/env python
import sys
sys.path.insert(0, 'ai-service')
from main import extract_facts

# Test case from failing test
pdf_text = "[Page 1] Patient (initials): J.T.\nAge: 68\nDrug: Trixamet"

print("=" * 80)
print("Testing: Patient (initials): J.T.")
print("=" * 80)
print(f"Input text:\n{pdf_text}\n")

facts = extract_facts(
    subject="uploaded scan",
    body="",
    attachment_text=pdf_text,
    category_name="Safety Report (ICSR)",
)

print("All extracted facts:")
print("-" * 80)
for fact in facts:
    print(f"{fact.get('factGroup'):12} | {fact.get('fieldName'):20} | {fact.get('fieldValue')}")

print("\n" + "=" * 80)
initials_facts = [f for f in facts if f["factGroup"] == "Patient" and f["fieldName"] == "Initials"]
print(f"Initials facts found: {len(initials_facts)}")
for fact in initials_facts:
    print(f"  Value: '{fact['fieldValue']}'")
    print(f"  Confidence: {fact['confidence']}")
    print(f"  Source: {fact['sourceReference']}")
