"""Create only fictional documents used by the local demo and automated checks.

Run from this directory with: python generate_synthetic_fixtures.py
The output is intentionally git-friendly: PDFs and a JSON manifest live in ../sample-data/.
"""
import json
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "sample-data"

FIXTURES = [
    ("01_digital_safety.pdf", "digital", "Safety Report", "Patient: Morgan Hale\nAge: 45\nSex: male\nProduct: SynthAspirin\nDose: 500 mg\nReaction: nausea\nOutcome: recovered\nHospitalization: no"),
    ("02_digital_safety_table.pdf", "digital", "Safety Report table", "Dose | Outcome\n5 mg | nausea\n10 mg | rash\nProduct: DemoCure\nPatient age: 52"),
    ("03_digital_quality.pdf", "digital", "Quality Complaint", "Product: DemoCure\nBatch: SYN-1003\nBroken seal and damaged packaging. Photo mentioned."),
    ("04_digital_info_request.pdf", "digital", "Info Request", "What dose of DemoCure should be taken with food?\nNo adverse reaction and no defect reported."),
    ("05_digital_irrelevant.pdf", "digital", "Marketing", "Synthetic marketing newsletter for a fictional conference. No case information."),
    ("06_scan_handwritten_safety.pdf", "scan", "Scanned report", "Handwritten mock form\nPatient age 34\nDrug DemoCure\nReaction rash"),
    ("07_scan_handwritten_quality.pdf", "scan", "Scanned quality report", "Handwritten mock form\nLot SYN-2007\nDamaged carton"),
    ("08_article_case_1.pdf", "article", "Fictional article", "Abstract\nA 58-year-old patient developed rash after 10 mg Imaginar.\nCase report\nThe patient recovered.\nReferences"),
    ("09_article_case_2.pdf", "article", "Fictional article", "Abstract\nA 42-year-old patient developed nausea after DemoCure.\nCase report\nReferences"),
    ("10_article_case_3.pdf", "article", "Fictional article", "Abstract\nCase report: a 61-year-old patient developed dizziness.\nConclusion\nReferences"),
    ("11_article_case_4.pdf", "article", "Fictional article", "Journal abstract\nA fictional patient had vomiting after SynthAspirin.\nReferences"),
    ("12_article_case_5.pdf", "article", "Fictional article", "DOI: 10.0000/synthetic\nCase report\nA 39-year-old patient recovered after rash.\nReferences"),
    ("13_spanish_case.pdf", "digital", "Caso en espanol", "Paciente de 50 anos tuvo nausea despues de tomar DemoCure 5 mg. Reaccion: nausea."),
    ("14_hindi_case.pdf", "digital", "Hindi case", "Patient with DemoCure had rash. Synthetic Hindi-language case marker: हिंदी."),
    ("15_mixed_safety_quality.pdf", "digital", "Mixed report", "A patient developed nausea after DemoCure. Batch SYN-2015 has a broken seal."),
]


def create_text_pdf(path, title, text):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((50, 55), title, fontsize=16)
    y = 90
    for line in text.splitlines():
        page.insert_text((50, y), line, fontsize=11)
        y += 20
    document.save(path)


def create_scanned_pdf(path, title, text):
    image = Image.new("RGB", (1200, 1600), "#fffdf5")
    draw = ImageDraw.Draw(image)
    draw.text((80, 80), title, fill="black", spacing=10)
    draw.multiline_text((80, 170), text, fill="black", spacing=12)
    png_path = path.with_suffix(".png")
    image.save(png_path)
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_image(page.rect, filename=png_path)
    document.save(path)
    png_path.unlink()


def main():
    OUTPUT.mkdir(exist_ok=True)
    manifest = []
    for filename, kind, title, text in FIXTURES:
        path = OUTPUT / filename
        if kind == "scan":
            create_scanned_pdf(path, title, text)
        else:
            create_text_pdf(path, title, text)
        manifest.append({"filename": filename, "pdfType": kind, "synthetic": True})
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created {len(manifest)} synthetic PDFs in {OUTPUT}")


if __name__ == "__main__":
    main()
