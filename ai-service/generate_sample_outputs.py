"""Process every generated synthetic PDF and save one JSON result per document."""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as ai_main  # noqa: E402

SAMPLE_DATA = ROOT / "sample-data"
OUTPUT = ROOT / "sample-outputs" / "documents"


def main() -> None:
    # Batch artifacts must be reproducible without a cloud quota or API key.
    ai_main.GEMINI_API_KEY = ""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(SAMPLE_DATA.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found. Run generate_synthetic_fixtures.py first.")
    for pdf in pdfs:
        result = ai_main.analyze_document_payload({
            "message_id": pdf.stem,
            "subject": pdf.stem,
            "sender": "synthetic.sender@example.invalid",
            "body": "Synthetic fixture message for assignment validation.",
            "attachments": [{
                "filename": pdf.name,
                "base64_content": base64.b64encode(pdf.read_bytes()).decode("ascii"),
            }],
        })
        (OUTPUT / f"{pdf.stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {len(pdfs)} JSON outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
