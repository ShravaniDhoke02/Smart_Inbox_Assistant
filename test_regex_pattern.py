#!/usr/bin/env python
import re

# The inline pattern for initials
pattern = r"(?is)\bpatient\s*\(\s*initials?\s*\)\s*[:\-]?\s*([A-Za-z](?:\s*[.\-]\s*[A-Za-z0-9]){1,8}\.?)"

test_texts = [
    "[Page 1] Patient (initials): J.T.\nAge: 68\nDrug: Trixamet",
    "Patient (initials): J.T.\nAge: 68",
    "Patient (initials): J.T.",
]

print("Testing regex pattern for initials extraction")
print("=" * 80)
print(f"Pattern: {pattern}\n")

for test_text in test_texts:
    print(f"Input: {repr(test_text)}")
    match = re.search(pattern, test_text)
    if match:
        print(f"  Match found!")
        print(f"  Group 0 (full match): {repr(match.group(0))}")
        print(f"  Group 1 (capture): {repr(match.group(1))}")
    else:
        print(f"  No match")
    print()
