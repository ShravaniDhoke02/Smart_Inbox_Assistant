#!/usr/bin/env python
import sys
sys.path.insert(0, 'ai-service')
from main import extract_facts

# Test 1: PDF with label-value on separate lines
pdf_text_1 = '''Individual Case Safety Report - Case Summary Form
1. Patient Information
Age
62 years
Sex
Female
Weight / Height
68 kg / 160 cm
Relevant history
Hypertension, Type 2 Diabetes'''

# Test 2: PDF with label-value on same line
pdf_text_2 = '''Patient Information
Age: 34 years
Sex: Female
Weight: 60 kg
Height: Not stated
Relevant history: No relevant history stated'''

print("=" * 60)
print("TEST 1: Label and value on separate lines")
print("=" * 60)
facts_1 = extract_facts('Test Case', '', pdf_text_1, 'Safety Report (ICSR)')
for fact in facts_1:
    if fact.get('factGroup') == 'Patient':
        print(f"{fact.get('fieldName')}: {fact.get('fieldValue')}")

print("\n" + "=" * 60)
print("TEST 2: Label and value on same line")
print("=" * 60)
facts_2 = extract_facts('Test Case', '', pdf_text_2, 'Safety Report (ICSR)')
for fact in facts_2:
    if fact.get('factGroup') == 'Patient':
        print(f"{fact.get('fieldName')}: {fact.get('fieldValue')}")
