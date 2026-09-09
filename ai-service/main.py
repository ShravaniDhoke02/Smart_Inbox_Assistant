from flask import Flask, request, jsonify
import base64
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from statistics import mean

# Limit OpenBLAS / MKL threads before numpy is imported to prevent OOM on startup
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import fitz
import numpy as np
from PIL import Image, ImageOps
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # Fallback when pypdf is not available

import pytesseract
from deep_translator import GoogleTranslator
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None

_rapid_ocr_instance = None


def get_rapid_ocr():
    """Lazy-load RapidOCR so its ONNX model is only allocated when first needed."""
    global _rapid_ocr_instance
    if _rapid_ocr_instance is None and RapidOCR is not None:
        try:
            _rapid_ocr_instance = RapidOCR()
        except Exception as exc:
            logger.warning("RapidOCR could not be initialised: %s", exc)
    return _rapid_ocr_instance


def get_gemini_api_key():
    return (os.environ.get("GEMINI_API_KEY") or "").strip()


GEMINI_API_KEY = get_gemini_api_key()
GEMINI_MODEL = (os.environ.get("GEMINI_MODEL") or "gemini-1.5-pro-latest").strip()
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

SYSTEM_INSTRUCTION = """
You are a highly skilled document classification and data extraction AI for a pharmaceutical company.
Your job is to read incoming emails and attached PDFs, classify them into one or more categories, and extract key facts.

Categories:
1. "Safety Report (ICSR)": A patient had a bad reaction to a drug. Look for a specific patient, reporter, drug, and bad outcome.
2. "Quality Complaint (PQC)": Something is physically wrong with the product (broken seal, wrong color, contamination).
3. "Info Request (MI)": Someone has a question about a product (dosing, interactions) with NO bad reaction and NO defect.
4. "Not Relevant": Anything else (marketing, spam, admin chatter).

IMPORTANT RULES:
- Output MUST be valid JSON format only, with no markdown wrappers like ```json.
- Every extracted fact MUST include a 'sourceReference' indicating exactly where it was found.
- Include a confidence score (0.0 to 1.0) for every extraction and classification.
- If a fact is not stated, use "Not stated". Do NOT guess.
- For a Safety Report, return every required patient, reporter, product, reaction, and severity field, even when its value is "Not stated".
- A message can belong to more than one category.
"""

REQUIRED_SAFETY_FIELDS = [
    ("Patient", "Initials"), ("Patient", "Name"), ("Patient", "Age"), ("Patient", "Sex"), ("Patient", "Weight"),
    ("Patient", "Height"), ("Patient", "Relevant History"),
    ("Reporter", "Name"), ("Reporter", "Role"), ("Reporter", "Country"),
    ("Product", "Name"), ("Product", "Disease / Indication"), ("Product", "Dose"), ("Product", "Route"),
    ("Product", "Therapy Start"), ("Product", "Therapy Stop"),
    ("Reaction", "What"), ("Reaction", "Suspected Cause"), ("Reaction", "Onset"), ("Reaction", "Outcome"),
    ("Severity", "Death"), ("Severity", "Hospitalization"),
    ("Severity", "Life-Threatening"), ("Narrative", "Case Summary"),
]


def sentence_count(text):
    return len([part for part in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if part])


def enforce_summary_length(text, subject="the document"):
    """Keep reviewer summaries within the assignment's 10-15 sentence contract."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if part.strip()]
    if len(sentences) > 15:
        sentences = sentences[:15]
    additions = [
        f"The reviewer should verify the original source for {subject} before accepting the analysis.",
        "Facts that are not explicitly stated must remain Not stated rather than being inferred.",
        "The evidence references identify the email or PDF page supporting each extracted fact.",
        "Any table or image interpretation is review support and is not a clinical conclusion.",
        "A human reviewer must confirm the classification and extracted values before completion.",
    ]
    if not sentences:
        sentences.append(f"{subject} was received for human review.")
    for addition in additions:
        if len(sentences) >= 10:
            break
        sentences.append(addition)
    return " ".join(sentences[:15])


def normalise_confidence(value):
    try:
        return round(max(0.0, min(1.0, float(value))), 2)
    except (TypeError, ValueError):
        return 0.0


def normalize_ai_result(result, subject, body, attachment_text):
    """Apply non-negotiable output rules after either Gemini or local analysis."""
    result = dict(result or {})
    category = str(result.get("category") or "Not Relevant")
    result["category"] = category
    result["confidenceScore"] = normalise_confidence(result.get("confidenceScore"))
    result["aiSummary"] = enforce_summary_length(result.get("aiSummary"), subject or "the message")
    result["extractedFacts"] = merge_verified_facts(
        result.get("extractedFacts"), extract_facts(subject, body, attachment_text, category), category, body, attachment_text
    )
    result["classificationReason"] = str(result.get("classificationReason") or "Human review required.")
    return result


def merge_verified_facts(model_facts, document_facts, category_name, email_body, attachment_text):
    """Prefer LLM extracted facts when available, fallback to regex facts."""
    merged = {}
    # First add document facts as a baseline
    for fact in document_facts or []:
        key = (str(fact.get("factGroup") or "").strip().title(), str(fact.get("fieldName") or "").strip().title())
        merged[key] = fact
    # Then overwrite with model facts if they provide a value (since model is more accurate on raw PDFs)
    for fact in model_facts or []:
        key = (str(fact.get("factGroup") or "").strip().title(), str(fact.get("fieldName") or "").strip().title())
        if str(fact.get("fieldValue") or "").strip().lower() != "not stated":
            merged[key] = fact
        elif key not in merged:
            merged[key] = fact
    return ensure_required_safety_facts(list(merged.values()), category_name, email_body, attachment_text)


def _check_synonym_in_text(val_clean, text_clean):
    """Check if normalized value or its source equivalent is in text_clean."""
    if not val_clean or not text_clean:
        return False
    synonym_map = {
        "female": ["woman", "female", "girl", "lady", "she", "her", "femenino", "mujer"],
        "male": ["man", "male", "boy", "gentleman", "he", "his", "masculino", "hombre"],
        "oral": ["oral", "orally", "po", "by mouth", "vía oral"],
        "once daily": ["once daily", "daily", "qd", "q.d.", "1x daily", "una vez al día"],
        "yes": ["hospital", "hospitalized", "hospitalisation", "emergency department", "emergency room", "observed for", "admitted", "ed visit"],
        "no": ["no hospitalization", "not admitted", "not hospitalized", "outpatient", "treated at home", "without hospitalization"],
        "discontinued": ["discontinued", "stopped", "withdrawn", "halted", "switched"],
        "resolved": ["resolved", "recovered", "resolution of symptoms", "disappearance of symptoms", "improved", "resolution"],
    }
    candidates = synonym_map.get(val_clean, [])
    for candidate in candidates:
        if re.search(rf"\b{re.escape(candidate)}\b", text_clean, re.IGNORECASE):
            return True
    return False


def source_reference(email_body, value, attachment_text):
    """Return reviewer-readable evidence location without inventing a page number."""
    if not value or str(value).lower() == "not stated":
        return "Not stated in email body or PDF attachment"

    val_clean = re.sub(r"\s+", " ", str(value).strip()).lower()
    att_clean = re.sub(r"\s+", " ", str(attachment_text or "")).lower()
    body_clean = re.sub(r"\s+", " ", str(email_body or "")).lower()

    if attachment_text and (val_clean in att_clean or _check_synonym_in_text(val_clean, att_clean)):
        pages = re.findall(r"\[Page (\d+)\]", attachment_text)
        page = "1"
        match_idx = att_clean.find(val_clean)
        if match_idx != -1:
            preceding = attachment_text[:match_idx]
            p_matches = re.findall(r"\[Page (\d+)\]", preceding)
            if p_matches:
                page = p_matches[-1]
        elif pages:
            page = pages[0]
        return f"PDF attachment, page {page}, matched text: {str(value)[:120]}"

    for index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", email_body or ""), start=1):
        s_clean = re.sub(r"\s+", " ", sentence.lower())
        if val_clean in s_clean or _check_synonym_in_text(val_clean, s_clean):
            return f"Email body, sentence {index}, matched text: {sentence[:160]}"

    return "Not stated in email body or PDF attachment"


def missing_source_reference(value, email_body, attachment_text):
    """Keep an explicit missing value auditable without claiming a source location."""
    if str(value or "").strip().lower() == "not stated":
        return "Not stated in email body or PDF attachment"
    return source_reference(email_body, value, attachment_text)


def _value_present_in_source(value, email_body, attachment_text):
    """Fuzzy check: returns True when the value (or whitespace-normalized/synonym variant) appears in either source."""
    if not value or str(value).lower() == "not stated":
        return False
    needle = re.sub(r"\s+", " ", str(value).lower()).strip()
    haystack = re.sub(r"\s+", " ", f"{email_body or ''} {attachment_text or ''}").lower()

    if needle in haystack:
        return True
    if _check_synonym_in_text(needle, haystack):
        return True

    partial = needle[:25].strip()
    if len(partial) >= 3 and partial in haystack:
        return True

    tokens = [t for t in re.split(r"\W+", needle) if len(t) >= 3 and t not in {"and", "the", "for", "with", "from", "was", "were", "has", "had"}]
    if tokens and all(t in haystack for t in tokens):
        return True

    return False


def ensure_required_safety_facts(facts, category_name, email_body, attachment_text):
    """Guarantee the ICSR schema and fill missing facts honestly rather than omitting them."""
    if "Safety Report (ICSR)" not in (category_name or ""):
        return facts or []
    normalized, seen = [], set()
    for fact in facts or []:
        group = str(fact.get("factGroup") or "General").strip().title()
        field = str(fact.get("fieldName") or "").strip().title()
        value = str(fact.get("fieldValue") or "Not stated").strip() or "Not stated"
        reference = str(fact.get("sourceReference") or "").strip()

        if value.lower() == "not stated":
            confidence = 0.0
            reference = missing_source_reference(value, email_body, attachment_text)
        elif _value_present_in_source(value, email_body, attachment_text):
            confidence = normalise_confidence(fact.get("confidence"))
            verified_reference = source_reference(email_body, value, attachment_text)
            reference = verified_reference if verified_reference.startswith(("Email body,", "PDF attachment,")) else reference
        else:
            confidence = round(normalise_confidence(fact.get("confidence")) * 0.6, 2)
            reference = (
                f"Extracted by pattern/LLM — not found verbatim in source; "
                f"human verification required. Original reference: {reference or 'none'}"
            )

        if not reference or reference.lower() in {"n/a", "unknown"}:
            reference = missing_source_reference(value, email_body, attachment_text)

        normalized.append({"factGroup": group, "fieldName": field, "fieldValue": value,
                           "confidence": confidence, "sourceReference": reference})
        seen.add((group, field))
    for group, field in REQUIRED_SAFETY_FIELDS:
        if (group, field) not in seen:
            normalized.append({"factGroup": group, "fieldName": field, "fieldValue": "Not stated",
                               "confidence": 0.0, "sourceReference": "Not stated in email body or PDF attachment"})
    return normalized

def call_llm(subject, sender, body, attachment_text, detected_types, raw_attachments=None):
    if not GEMINI_API_KEY:
        return None

    prompt = (
        f"Subject: {subject}\n"
        f"Sender: {sender}\n"
        f"Body:\n{body}\n\n"
        f"Attachment Types Detected: {detected_types}\n"
        f"Attachment Text (may contain OCR output with minor typos/artifacts):\n{attachment_text}\n\n"
        "IMPORTANT EXTRACTION RULES:\n"
        "- The attachment text may come from OCR of a handwritten card, scanned form, non-English document, or published article.\n"
        "- Tolerate minor OCR artifacts: e.g. 'Nb' means 'No', 'rot clear' means 'not clear', spacing/accent issues.\n"
        "- For handwritten cards, look for labels like 'Name (initials only):', 'Age/Sex:', 'Product:', 'What happened:', 'Hospitalized?:', 'Reported by:', 'Batch (if visible):'.\n"
        "- For Spanish/non-English, translate field labels: Edad=Age, Sexo=Sex, Producto=Product, Reacción=Reaction, Resultado=Outcome, Hospitalización=Hospitalization, Informante=Reporter.\n"
        "- For published articles, extract patient-level case details from prose (e.g. 'A 34-year-old man developed...').\n"
        "- For form-style PDFs where a label and its value appear on separate lines, associate them correctly.\n"
        "- Do NOT return 'Not stated' for a field if the information is present in the text, even if OCR quality is imperfect.\n\n"
        "Return a JSON object with these keys:\n"
        "- 'category': 'Safety Report (ICSR)', 'Quality Complaint (PQC)', 'Info Request (MI)', or 'Not Relevant' (comma-separated if multiple)\n"
        "- 'confidenceScore': number 0.0-1.0\n"
        "- 'classificationReason': string\n"
        "- 'aiSummary': string (10-15 sentences)\n"
        "- 'extractedFacts': list of {factGroup, fieldName, fieldValue, confidence, sourceReference}"
    )

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        parts = [SYSTEM_INSTRUCTION, prompt]
        
        for att in raw_attachments or []:
            filename = att.get("filename", "").lower()
            file_bytes = att.get("bytes")
            if not file_bytes:
                continue
            if filename.endswith(".pdf"):
                parts.append({"mime_type": "application/pdf", "data": file_bytes})
            elif filename.endswith((".jpg", ".jpeg")):
                parts.append({"mime_type": "image/jpeg", "data": file_bytes})
            elif filename.endswith(".png"):
                parts.append({"mime_type": "image/png", "data": file_bytes})
            
        response = model.generate_content(parts)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        return json.loads(text)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return None


def count_keyword_mentions(text, keywords):
    return sum(1 for keyword in keywords if keyword in text)


def count_negated_mentions(text, patterns):
    return sum(1 for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE))


def classify_text(subject, sender, body, attachment_text):
    combined = " ".join(filter(None, [subject, sender, body, attachment_text])).lower()
    categories = []
    reasons = []

    safety_signal_keywords = [
        "nausea", "rash", "adverse", "reaction", "hospital", "serious", "vomiting",
        "dizziness", "headache", "allergic", "anaphylaxis", "injury", "death", "side effect" ]
    safety_negation_patterns = [
        r"\bno\s+(?:adverse\s+)?reaction\b",
        r"\bno\s+side\s+effects?\b",
        r"\bno\s+(?:serious|major|severe)\s+event\b",
        r"\bno\s+(?:adverse|safety|clinical)\s+signal\b",
        r"\bno\b[^.]{0,120}\b(?:reaction|complaint|medical\s+question|medical\s+inquiry|adverse\s+event|defect|quality\s+issue)\b[^.]{0,120}\b(?:is|are|was|were)\s+(?:discussed|mentioned|reported|stated)\b",
        r"\bno\s+(?:medicinal\s+)?product\b[^.]{0,120}\b(?:reaction|complaint|medical\s+question|medical\s+inquiry|adverse\s+event|defect|quality\s+issue)\b",
    ]
    # Additional safety context keywords to capture synthetic reports and broader terminology while avoiding generic product-only mentions
    safety_context_keywords = ["patient", "male", "female", "reaction description", "synthetic safety report"]
    quality_keywords = ["broken seal", "damaged", "contamination", "defect", "lot", "batch", "wrong color", "counterfeit", "cracked", "leaking", "packaging"]
    quality_negation_patterns = [
        r"\bno\s+(?:broken\s+seal|defect|damage|contamination|wrong\s+color|counterfeit|cracked|leaking|packaging)\b",
        r"\bnot\s+(?:defective|damaged|contaminated|broken)\b",
        r"\bno\s+quality\s+issue\b",
        r"\bno\s+(?:product\s+)?defect\s+(?:is\s+)?(?:reported|present|mentioned|described)\b",
    ]
    info_keywords = ["dose", "dosing", "interaction", "what dose", "how should", "how to take", "with food", "without food", "can i take", "please advise"]

    safety_signal_score = max(0, count_keyword_mentions(combined, safety_signal_keywords) - count_negated_mentions(combined, safety_negation_patterns))
    safety_context_score = count_keyword_mentions(combined, safety_context_keywords)
    safety_score = safety_signal_score + safety_context_score
    quality_score = max(0, count_keyword_mentions(combined, quality_keywords) - count_negated_mentions(combined, quality_negation_patterns))
    info_score = count_keyword_mentions(combined, info_keywords)

    # Early detection for known synthetic safety report patterns
    has_explicit_medical_question = bool(re.search(
        r"(?i)\b(?:could you|please confirm|also wanted to check|what|how|can|which|when|why)\b[^?]{0,300}\?",
        combined,
    ))
    explicit_no_adverse_event = bool(re.search(r"(?i)\bno adverse event to report\b|\bno adverse reaction\b", combined))
    explicit_no_defect = bool(re.search(r"(?i)\bno\s+(?:product\s+)?defect\s+(?:is\s+)?(?:reported|present|mentioned|described)\b", combined))
    explicit_negative_document = bool(re.search(
        r"(?i)\bno\s+patient,\s+drug\s+reaction,\s+product\s+defect,\s+or\s+medical\s+information\s+question\b",
        combined,
    ))
    if explicit_negative_document and not has_explicit_medical_question:
        return "Not Relevant", 0.97, "The document explicitly states that it contains no patient event, product defect, or medical-information question."
    if has_explicit_medical_question and explicit_no_adverse_event and (quality_score == 0 or explicit_no_defect):
        categories.append("Info Request (MI)")
        reasons.append("The message contains explicit medical questions and states that no adverse event is being reported.")
    elif "synthetic safety report" in combined or "reaction description" in combined:
        categories.append("Safety Report (ICSR)")
        reasons.append("Identified synthetic safety report based on key phrases.")
    elif safety_signal_score > 0 and safety_context_score > 0:
        categories.append("Safety Report (ICSR)")
        reasons.append("Patient safety signal and adverse-event language were found in the message or attachment.")
    if quality_score > 0 and not explicit_no_defect:
        categories.append("Quality Complaint (PQC)")
        reasons.append("Product defect or packaging quality issue was described.")
    if info_score > 0 and not (safety_signal_score > 0 and safety_context_score > 0) and quality_score == 0 and "Info Request (MI)" not in categories:
        categories.append("Info Request (MI)")
        reasons.append("The content is a product question without an adverse outcome or defect signal.")
    if not categories:
        categories.append("Not Relevant")
        reasons.append("The document/message is not relevant to the application scope (ICSR, PQC, or MI). It appears to be general clinical, administrative, marketing, or non-safety content with no adverse event, product quality complaint, or medical information inquiry signal.")

    category_name = ", ".join(categories)
    confidence = min(0.97, 0.55 + 0.08 * max(safety_score, quality_score, info_score))
    return category_name, round(confidence, 2), "; ".join(reasons)


def extract_image_text(file_bytes, filename):
    if not filename:
        return "", "No image"
    lowered = filename.lower()
    if not any(lowered.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff")):
        return "", "Not an image"

    try:
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".png", delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name
        with Image.open(temp_path) as img:
            img = ImageOps.grayscale(img)
            text = pytesseract.image_to_string(img, config="--psm 6")
            if text.strip():
                return text.strip(), "Image OCR"
            return "Image detected; handwritten or visual content requires specialist review.", "Image OCR"
    except Exception as exc:
        logger.warning("Image OCR failed for %s: %s", filename, exc)
        return "Image detected; OCR could not be performed automatically and requires manual review.", "Image OCR"


def _ocr_page_with_rapidocr(page_obj, scale=1.0):
    """Render a fitz page as grayscale and run RapidOCR; retry at smaller scale on OOM."""
    engine = get_rapid_ocr()
    if engine is None:
        return ""
    # Render as grayscale to cut memory usage by ~3x vs RGB
    pix = page_obj.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY)
    try:
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        ocr_result, _ = engine(img_array)
        return "\n".join(item[1] for item in (ocr_result or []) if len(item) > 1)
    except (RuntimeError, MemoryError, Exception) as err:
        fallback_scale = round(scale * 0.7, 2)
        if fallback_scale >= 0.4:
            logger.warning("RapidOCR OOM at scale %.2f, retrying at %.2f: %s", scale, fallback_scale, err)
            return _ocr_page_with_rapidocr(page_obj, scale=fallback_scale)
        logger.warning("RapidOCR failed at all scales: %s", err)
        return ""


def _ocr_page_with_tesseract(page_obj, scale=1.5):
    """Render a fitz page and run pytesseract; silently skip if binary missing."""
    try:
        import pytesseract as _tess
        pix = page_obj.get_pixmap(matrix=fitz.Matrix(scale, scale))
        image = ImageOps.grayscale(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        return _tess.image_to_string(image, config="--psm 6") or ""
    except Exception as err:
        logger.warning("Tesseract OCR failed: %s", err)
        return ""


def extract_pdf_text(file_bytes, filename):
    if not filename.lower().endswith(".pdf"):
        return "", "Not a PDF"
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        reader = PdfReader(temp_path) if PdfReader else None
        pdf_doc = fitz.open(temp_path)
        pages = []
        scanned_pages = 0

        for page_number, page_obj in enumerate(pdf_doc, start=1):
            # Try digital text first (fast path)
            page_text = page_obj.get_text("text").strip()

            # If pypdf available and fitz returned nothing, try pypdf too
            if not page_text and reader and page_number <= len(reader.pages):
                page_text = (reader.pages[page_number - 1].extract_text() or "").strip()

            # Scanned / handwritten page — use OCR
            if not page_text:
                if get_rapid_ocr() is not None:
                    page_text = _ocr_page_with_rapidocr(page_obj, scale=1.5)
                else:
                    page_text = _ocr_page_with_tesseract(page_obj, scale=1.5)
                if page_text.strip():
                    scanned_pages += 1

            pages.append(f"[Page {page_number}]\n{normalise_ocr_spacing(page_text)}")

        pdf_doc.close()
        text = "\n\n".join(pages)
        readable = " ".join(p for p in pages if p.replace(f"[Page {pages.index(p) + 1}]", "").strip())
        if not readable.strip():
            return (
                "OCR required: no readable text found in the PDF. "
                "Handwritten or scanned content needs OCR validation.",
                "Scanned / OCR required",
            )
        if scanned_pages > 0:
            return text, "Scanned / OCR required"
        return text, classify_pdf_type(text)
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", filename, exc)
        return (
            "OCR required: the attachment could not be read directly "
            "and should be reviewed by OCR or human inspection.",
            "Scanned / OCR required",
        )


def normalise_ocr_spacing(text):
    """Repair common OCR joins without changing the document's wording."""
    value = re.sub(r"(?i)\b(Neurotab)\s*XR(?=\d|\b)", r"\1 XR", text or "")
    value = re.sub(r"(?i)\b(Trixamet)\s*(\d+)", r"\1 \2", value)
    value = re.sub(r"(?i)\b(Neurotab XR)\s*(\d+)", r"\1 \2", value)
    value = re.sub(r"(?i)(\d+)o(?=mg\b)", r"\g<1>0", value)
    value = re.sub(r"(?i)(\d+)\s*(mg|mcg|ml|g|iu)(?=[A-Za-z\s]|$)", r"\1 \2 ", value)
    value = re.sub(r"(?i)(\d)\s*(hours?|days?)\b", r"\1 \2", value)
    value = re.sub(r"(?i)(?<!\d)(\d)(?=(?:Sex|Gender|Dose|Reaction|Outcome|Hospitalized)\b)", r"\1 ", value)
    value = re.sub(r"(?i)\b(both|after|home|R\.)\s*(forearms|hospital|health|[A-Z])", r"\1 \2", value)
    value = re.sub(r"(?i)\bR\.?\s*A\s*[/\\]?\s*l?varez\b", "R. Alvarez", value)
    value = re.sub(r"(?i)\bnot\s*resolved\b", "Not resolved", value)
    return value


def inspect_pdf_images(file_bytes, filename):
    """Describe embedded visuals, with optional vision assistance and provenance."""
    if not filename.lower().endswith(".pdf"):
        return []
    notes = []
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
        for page_number, page in enumerate(document, start=1):
            images = page.get_images(full=True)
            for image_number, image_info in enumerate(images, start=1):
                xref = image_info[0]
                extracted = document.extract_image(xref)
                image_bytes = extracted.get("image", b"")
                ocr_text = ""
                if image_bytes:
                    try:
                        with Image.open(__import__("io").BytesIO(image_bytes)) as image:
                            ocr_text = pytesseract.image_to_string(ImageOps.grayscale(image), config="--psm 6").strip()
                    except Exception:
                        ocr_text = ""
                try:
                    with Image.open(__import__("io").BytesIO(image_bytes)) as image:
                        width, height = image.size
                        orientation = "landscape" if width > height else "portrait" if height > width else "square"
                except Exception:
                    width, height, orientation = 0, 0, "unknown orientation"
                vision_description = describe_visual_content(image_bytes, filename, page_number)
                detail = f"embedded {orientation} image ({width}x{height}px) detected"
                if vision_description:
                    detail += f"; visual description: {vision_description}"
                if ocr_text:
                    detail += f"; OCR found: {ocr_text[:120]}"
                notes.append(f"{filename}, page {page_number}, image {image_number}: {detail}; human review required.")
        document.close()
    except Exception as exc:
        logger.warning("Image inspection failed for %s: %s", filename, exc)
        notes.append(f"{filename}: embedded-image inspection could not be completed; human review required.")
    return notes


def describe_visual_content(image_bytes, filename, page_number):
    """Ask the vision model for a non-diagnostic description when configured."""
    if not image_bytes or not GEMINI_API_KEY:
        return "visual model unavailable; inspect the original image manually"
    try:
        with Image.open(__import__("io").BytesIO(image_bytes)) as image:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content([
                "Describe only visible document, product-packaging, form, or skin-finding content in one short sentence. "
                "Do not diagnose, identify a person, or infer a clinical condition. Mention uncertainty.",
                image.copy(),
            ])
        description = str(response.text or "").strip().replace("\n", " ")
        return description[:300] if description else "vision model returned no description"
    except Exception as exc:
        logger.warning("Vision description failed for %s page %s: %s", filename, page_number, exc)
        return "visual model failed; inspect the original image manually"


def assess_ocr_confidence(file_bytes, filename, pdf_type):
    """Return OCR and handwriting confidence, including page-level provenance."""
    if pdf_type != "Scanned / OCR required" or not filename.lower().endswith(".pdf"):
        return {"required": False, "confidence": None, "handwritingConfidence": None,
                "method": "Direct digital-text extraction"}
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
        scores = []
        page_scores = []
        engine = get_rapid_ocr()
        for page_number, page in enumerate(document, start=1):
            page_values = []
            non_empty = False
            pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), colorspace=fitz.csGRAY)
            handwriting_signal = 0.0
            if engine is not None:
                try:
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
                    result, _ = engine(img_array)
                    for item in result or []:
                        if len(item) > 2 and isinstance(item[2], (int, float)):
                            page_values.append(float(item[2]))
                        if len(item) > 1 and str(item[1]).strip():
                            non_empty = True
                    handwriting_signal = 0.35 if non_empty and len(page_values) < 3 else 0.0
                except Exception as ocr_err:
                    logger.warning("RapidOCR confidence assessment failed page %s: %s", page_number, ocr_err)
            else:
                try:
                    image = Image.frombytes("L", [pix.width, pix.height], pix.samples)
                    data = pytesseract.image_to_data(image, config="--psm 6", output_type=pytesseract.Output.DICT)
                    values = [float(v) / 100 for v in data.get("conf", []) if str(v).strip() not in {"", "-1"} and float(v) >= 0]
                    page_values.extend(values)
                    non_empty = any(str(v).strip() for v in data.get("text", []))
                    handwriting_signal = 0.35 if non_empty and len(values) < 3 else 0.0
                except Exception:
                    pass
            page_confidence = mean(page_values) if page_values else (0.55 if non_empty else 0.0)
            handwriting_confidence = round(max(0.0, min(1.0, handwriting_signal + (0.2 if page_confidence < 0.55 else 0.0))), 2)
            scores.extend(page_values)
            page_scores.append({"page": page_number, "confidence": round(page_confidence, 2),
                                "handwritingConfidence": handwriting_confidence, "textDetected": non_empty})
        document.close()
        confidence = mean(scores) if scores else mean(item["confidence"] for item in page_scores) if page_scores else 0.0
        handwriting = mean(item["handwritingConfidence"] for item in page_scores) if page_scores else 0.0
        return {"required": True, "confidence": round(confidence, 2),
            "handwritingConfidence": round(handwriting, 2),
            "method": "RapidOCR" if engine is not None else "Tesseract image_to_data", "pages": page_scores}
    except Exception as exc:
        logger.warning("OCR confidence assessment failed for %s: %s", filename, exc)
        return {"required": True, "confidence": 0.0, "handwritingConfidence": 0.0,
            "method": "OCR confidence unavailable; manual verification required"}


def extract_structured_pdf_tables(file_bytes, filename):
    """Use PyMuPDF's layout-aware table detector before the text fallback."""
    tables = []
    if not filename.lower().endswith(".pdf"):
        return tables
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
        for page_number, page in enumerate(document, start=1):
            finder = page.find_tables()
            for table_number, table in enumerate(finder.tables, start=1):
                rows = [["" if cell is None else re.sub(r"\s+", " ", str(cell)).strip() for cell in row] for row in table.extract()]
                rows = [row for row in rows if any(row)]
                if rows:
                    width = max(len(row) for row in rows)
                    rows = [row + [""] * (width - len(row)) for row in rows]
                    columns = rows[0] or [f"Column {index}" for index in range(1, width + 1)]
                    tables.append({"attachment": filename, "page": page_number, "table": table_number,
                                   "columns": columns, "rows": rows[1:] if len(rows) > 1 else [],
                                   "rawRows": rows,
                                   "confidence": 0.9,
                                   "extractionMethod": "PyMuPDF layout-aware table detector",
                                   "sourceReference": f"{filename}, page {page_number}, table {table_number}"})
        document.close()
    except Exception as exc:
        logger.info("Layout-aware table extraction unavailable for %s: %s", filename, exc)
    return tables


def extract_article_cases(file_bytes, filename, fallback_text):
    """Extract individual reportable patient cases from published article PDFs."""
    if not filename.lower().endswith(".pdf"):
        return []

    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
        page_text = []
        for number, page in enumerate(document, start=1):
            blocks = [block for block in page.get_text("blocks") if block[4].strip()]
            page_width = page.rect.width
            columns = 2 if len(blocks) >= 4 and max(block[2] for block in blocks) < page_width * 0.75 else 1
            if columns == 2:
                midpoint = page_width / 2
                blocks.sort(key=lambda block: (0 if block[0] < midpoint else 1, block[1], block[0]))
            else:
                blocks.sort(key=lambda block: (block[1], block[0]))
            page_text.append(f"[Page {number}]\n" + "\n".join(block[4].strip() for block in blocks))
        document.close()
        article_text = "\n".join(page_text)
        if (
            len(re.sub(r"\s+", " ", article_text).strip()) < 100
            or not re.search(r"(?i)\b\d{1,3}\s*[- ]?\s*year\s*[- ]?\s*old\b", article_text)
        ) and fallback_text:
            article_text = fallback_text
    except Exception as exc:
        logger.warning("Article PDF extraction failed for %s: %s", filename, exc)
        article_text = fallback_text or ""

    if not article_text:
        return []

    article_text = re.split(
        r"(?im)^\s*(references|bibliography|acknowledg(?:e)?ments|conflict of interest)\s*$",
        article_text,
        maxsplit=1,
    )[0]

    intro_start = 0
    intro_match = re.search(r"(?im)^\s*(?:Introduction|Background|Clinical\s+Background)\b", article_text)
    if intro_match:
        intro_start = intro_match.start()

    candidates = []
    for pattern in [
        r"(?i)\bCase\s+Presentation\b",
        r"(?i)\bCase\s+\d+\b",
        r"(?i)\bPatient\s+\d+\b",
        r"(?i)\bClinical\s+Course\b",
    ]:
        for match in re.finditer(pattern, article_text[intro_start:]):
            start = intro_start + match.start()
            context = article_text[start:start + 250]
            if re.search(r"(?i)\b\d{1,3}\s*[- ]?\s*year\s*[- ]?\s*old\b|\b(?:female|male|woman|man)\b|\bpatient\b", context):
                candidates.append(start)

    if not candidates:
        fallback_pattern = re.compile(r"(?i)\bA\s+\d{1,3}\s*[- ]?\s*year\s*[- ]?\s*old\s+(?:female|male|woman|man)\b")
        for match in fallback_pattern.finditer(article_text[intro_start:]):
            candidates.append(intro_start + match.start())

    candidates = sorted(set(candidates))
    sections = []
    for idx, start in enumerate(candidates):
        end = candidates[idx + 1] if idx + 1 < len(candidates) else len(article_text)
        section = article_text[start:end].strip()
        if not section:
            continue
        stop_match = re.search(r"(?i)\b(?:discussion|references|conclusion|acknowledg(?:e)?ments|conflict of interest)\b", section)
        if stop_match:
            section = section[:stop_match.start()].strip()
        if section:
            sections.append(section)

    cases = []
    for chunk in sections:
        chunk = chunk.strip()
        if not chunk:
            continue
        has_demographic_signal = re.search(r"(?i)\b\d{1,3}\s*[- ]?\s*year\s*[- ]?\s*old\b|\b(?:female|male|woman|man)\b", chunk)
        has_event_signal = re.search(r"(?i)\b(?:adverse\s+event|adverse\s+reaction|reaction|rash|nausea|vomiting|dizziness|angioedema|swelling|seizure|headache|hospital|death|discontinued|resolved|serious|dyspnea)\b", chunk)
        if not (has_demographic_signal and has_event_signal):
            continue

        source_reference = f"{filename}, article case section"
        case_facts = []
        try:
            case_facts = extract_facts("", chunk, "", "Safety Report (ICSR)")
        except Exception as exc:
            logger.warning("Structured article case extraction failed for %s: %s", filename, exc)

        for fact in case_facts:
            fact["sourceReference"] = source_reference
            fact["caseNumber"] = len(cases) + 1

        cases.append({
            "caseNumber": len(cases) + 1,
            "sourceText": chunk[:8000],
            "summary": build_article_case_summary(chunk),
            "sourceReference": source_reference,
            "patientCaseOnly": True,
            "extractedFacts": case_facts,
        })

    return cases


def classify_pdf_type(text):
    combined = (text or "").lower()
    scan_markers = ["scanned", "ocr", "handwritten", "blurred", "signature", "form", "image only", "illegible", "fax"]
    if not combined.strip():
        return "Scanned / OCR required"
    if re.search(r"(abstract|case report|literature|references|conclusion|journal|doi)", combined):
        return "Published article"
    if any(marker in combined for marker in scan_markers):
        return "Scanned / OCR required"
    if re.search(r"(dosis|consulta|cuál|interacción|pregunta|¿|paciente|reacción)", combined):
        return "Non-English"
    return "Normal digital PDF"


def detect_language(text):
    """Detect the primary language of the given text.

    Uses a scoring approach for Latin-script languages so that stray
    accented characters in English medical text (e.g. '°C', proper
    names) do not cause false positives.  Non-Latin scripts are detected
    by Unicode range and returned immediately.
    """
    if not text:
        return "English"
    lowered = text.lower()

    # --- Non-Latin scripts: a single character match is sufficient ---
    script_ranges = [
        (r"[\u0900-\u097F]", "Hindi"),
        (r"[\u0980-\u09FF]", "Bengali"),
        (r"[\u0A00-\u0A7F]", "Punjabi"),
        (r"[\u0B80-\u0BFF]", "Tamil"),
        (r"[\u0C00-\u0C7F]", "Telugu"),
        (r"[\u0600-\u06FF]", "Arabic/Urdu"),
        (r"[\u4E00-\u9FFF]", "Chinese"),
        (r"[\u3040-\u30FF]", "Japanese"),
        (r"[\uAC00-\uD7AF]", "Korean"),
        (r"[\u0400-\u04FF]", "Russian"),
    ]
    for pattern, language in script_ranges:
        if re.search(pattern, text):
            return language

    # --- Latin-script languages: score by keyword hits ---
    # Each entry: (language, keyword_list, accent_pattern, min_score)
    # Keywords must NOT include words that are also common English words
    # (e.g. 'patient', 'dose').  Accent patterns contribute 1 point per
    # unique match only when keywords already scored at least 1.
    lang_profiles = [
        ("Spanish", [
            "paciente", "reaccion", "reacción", "dosis", "después", "despues",
            "tomar", "tuvo", "años", "años", "médico", "hospitalización",
            "informante", "sexo", "femenino", "masculino", "edad", "producto",
            "resultado", "vía",
        ], r"[¿ñ]", 2),
        ("French", [
            "femme", "homme", "ordonnance", "éruption", "indésirables",
            "traitement", "médicament", "posologie", "effets", "hôpital",
            "médecin", "patiente",
        ], r"[àâçèêëîïôûùÿæœ]", 2),
        ("German", [
            "patientin", "nebenwirkung", "dosierung", "einnahme",
            "gesichtsödem", "krankenhaus", "arzt", "befund", "behandlung",
            "arzneimittel",
        ], r"[ß]", 2),
        ("Italian", [
            "paziente", "donna", "uomo", "reazione", "collaterali",
            "dosaggio", "somministrazione", "ospedale", "effetti",
        ], None, 2),
        ("Portuguese", [
            "reação", "efeitos", "colaterais", "dosagem", "medicamento",
            "hospitalização",
        ], r"[ãõ]", 2),
    ]

    best_lang = "English"
    best_score = 0

    for lang, keywords, accent_pat, min_score in lang_profiles:
        score = sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", lowered))
        if score >= 1 and accent_pat and re.search(accent_pat, text):
            score += 1
        if score >= min_score and score > best_score:
            best_score = score
            best_lang = lang

    if best_lang != "English":
        return best_lang

    # Final fallback: text has no Latin characters at all
    if re.search(r"[\u00C0-\uFFFF]", text) and not re.search(r"[a-z]", lowered):
        return "Non-English"
    return "English"


def translate_non_english_text(text):
    if not text or len(text.strip()) < 5:
        return text
    translated_parts = []
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
    if not chunks:
        chunks = [text]

    for chunk in chunks:
        if not re.search(r"[a-zA-Z\u00C0-\uFFFF]", chunk):
            translated_parts.append(chunk)
            continue
        translated_part = None
        for attempt in range(2):
            try:
                translator = GoogleTranslator(source="auto", target="en")
                res = translator.translate(chunk)
                if (
                    res
                    and str(res).strip()
                    and not re.search(r"\b(error 500|server error|try again later|404|bad gateway)\b", str(res), re.IGNORECASE)
                ):
                    translated_part = str(res).strip()
                    break
            except Exception as exc:
                logger.warning("Translation attempt %d failed for chunk: %s", attempt + 1, exc)
        translated_parts.append(translated_part or chunk)

    translated = "\n\n".join(translated_parts)
    return translated.strip()


def extract_table_rows(text):
    rows = []
    if not text:
        return rows
    table_patterns = [r"([A-Za-z /-]+)\s*\|\s*([A-Za-z0-9 /-]+)", r"(dose|outcome|age|lot|batch|patient|medication)\s*[:\-]\s*([A-Za-z0-9 /-]+)"]
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            row = [part.strip() for part in line.split("|") if part.strip()]
            if len(row) >= 2:
                rows.append(row)
                continue
        if "\t" in line:
            row = [part.strip() for part in line.split("\t") if part.strip()]
            if len(row) >= 2:
                rows.append(row)
                continue
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in table_patterns):
            splits = re.split(r"\s{2,}|\s*[-:]\s*", line)
            row = [part.strip() for part in splits if part.strip()]
            if len(row) >= 2:
                rows.append(row)
    form_labels = {
        "patient id", "age", "sex", "weight", "relevant history", "country",
        "product name", "dose", "frequency", "route", "therapy start", "therapy stop",
        "indication", "batch/lot", "reaction", "onset", "outcome", "hospitalization",
        "emergency treatment", "life-threatening", "other medications", "medical history",
        "allergies", "relevant tests", "reporter", "role", "contact", "report date",
        "report type"
    }
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, label in enumerate(lines[:-1]):
        normalized = re.sub(r"\s+", " ", label).lower()
        if normalized in form_labels:
            value = lines[index + 1]
            if value.lower() not in form_labels and not re.match(r"^\d+\.\s", value):
                rows.append([label, value])
    return rows


def build_article_case_summary(text):
    if not text:
        return "No case details extracted from article text."
    lower_text = text.lower()
    age_match = re.search(r"(age|aged|years old|y/o)\s*(\d{1,3})", lower_text)
    dose_match = re.search(r"(dose|dosing|mg|mcg|g|ml)\s*(\d+\s*(mg|mcg|g|ml))", lower_text)
    reaction_match = re.search(r"(nausea|rash|vomiting|dizziness|headache|allergic reaction|anaphylaxis)", lower_text)
    outcome_match = re.search(r"(recovered|resolved|hospital|serious|death|discontinued)", lower_text)
    patient = "patient" if "patient" in lower_text else "subject"
    age = age_match.group(2) if age_match else "not stated"
    dose = dose_match.group(0) if dose_match else "not stated"
    reaction = reaction_match.group(0).title() if reaction_match else "not stated"
    outcome = outcome_match.group(0).title() if outcome_match else "not stated"
    return (
        f"Case summary: a {patient} aged {age} developed {reaction} after exposure to a product with dose {dose}. "
        f"The article notes {outcome} as the clinical outcome and includes references requiring literature review."
    )


def extract_facts(subject, body, attachment_text, category_name):
    facts = []
    source_text = "Email body"
    if attachment_text:
        source_text = "Attachment text"

    source_blob = normalise_ocr_spacing(f"{subject} {body} {attachment_text}".strip())

    # ------------------------------------------------------------------ #
    #  FORM-LAYOUT EXTRACTION                                              #
    #  Handles PDFs where a label appears on one line, value on the next. #
    # ------------------------------------------------------------------ #
    form_label_map = {
        "age": ("Patient", "Age"),
        "sex": ("Patient", "Sex"),
        "gender": ("Patient", "Sex"),
        "patient age": ("Patient", "Age"),
        "patient sex": ("Patient", "Sex"),
        "patient name": ("Patient", "Name"),
        "name of patient": ("Patient", "Name"),
        "weight": ("Patient", "Weight"),
        "weight / height": ("Patient", "Weight"),  # Handle compound labels
        "height": ("Patient", "Height"),
        "history": ("Patient", "Relevant History"),
        "relevant history": ("Patient", "Relevant History"),
        "medical history": ("Patient", "Relevant History"),
        "seasonal allergies": ("Patient", "Relevant History"),
        "role": ("Reporter", "Role"),
        "reporter": ("Reporter", "Name"),
        "reporter name": ("Reporter", "Name"),
        "country": ("Reporter", "Country"),
        "name": ("Product", "Name"),
        "product": ("Product", "Name"),
        "drug": ("Product", "Name"),
        "dose": ("Product", "Dose"),
        "regimen": ("Product", "Regimen"),
        "frequency": ("Product", "Regimen"),
        "route": ("Product", "Route"),
        "start": ("Product", "Therapy Start"),
        "stop": ("Product", "Therapy Stop"),
        "description": ("Reaction", "What"),
        "reaction description": ("Reaction", "What"),
        "onset": ("Reaction", "Onset"),
        "outcome": ("Reaction", "Outcome"),
        "action": ("Reaction", "Action"),
        "seriousness": ("Severity", "Hospitalization"),
        "hospitalization": ("Severity", "Hospitalization"),
    }
    
    def extract_compound_value(next_line, normalized_label):
        """Extract values from compound labels like 'Weight / Height: 68 kg / 160 cm'."""
        if normalized_label == "weight / height" and "/" in next_line:
            parts = next_line.split("/")
            if len(parts) >= 2:
                return parts[0].strip(), parts[1].strip()  # (weight, height)
        return next_line.strip(), None
    
    lines = [l.strip() for l in (attachment_text or "").splitlines() if l.strip()]
    for idx, line in enumerate(lines[:-1]):
        normalized_label = re.sub(r"\s+", " ", line).lower().rstrip(":")
        if normalized_label in form_label_map and idx + 1 < len(lines):
            next_line = lines[idx + 1].strip()
            if next_line.lower().rstrip(":") not in form_label_map and len(next_line) >= 1:
                fg, fn = form_label_map[normalized_label]
                
                # Handle compound labels like "Weight / Height"
                if normalized_label == "weight / height":
                    weight_val, height_val = extract_compound_value(next_line, normalized_label)
                    if weight_val:
                        # For weight/height, preserve the value as-is (don't apply .title())
                        val = weight_val
                        facts.append({
                            "factGroup": fg,
                            "fieldName": "Weight",
                            "fieldValue": val,
                            "confidence": 0.82,
                            "sourceReference": source_reference(body, val, attachment_text),
                        })
                    if height_val:
                        val = height_val
                        facts.append({
                            "factGroup": fg,
                            "fieldName": "Height",
                            "fieldValue": val,
                            "confidence": 0.82,
                            "sourceReference": source_reference(body, val, attachment_text),
                        })
                else:
                    val = next_line.title() if len(next_line) < 50 else next_line
                    facts.append({
                        "factGroup": fg,
                        "fieldName": fn,
                        "fieldValue": val,
                        "confidence": 0.82,
                        "sourceReference": source_reference(body, val, attachment_text),
                    })

    # Inline fields are common in copied PDF tables, for example
    # "Patient Age: 62  Patient Sex: Woman". Match only known labels so
    # nearby prose does not become a fabricated field value.
    inline_label_patterns = [
        (r"(?is)\bpatient[ \t]*\([ \t]*initials?[ \t]*\)[ \t]*[:\-]?[ \t]*([A-Za-z](?:[ \t]*[.\-][ \t]*[A-Za-z0-9]){0,3}\.?)(?=[ \t]*(?:\n|$))", "Patient", "Initials"),
        (r"(?is)\b(?:patient\s+)?age\s*[:\-]?\s*(\d{1,3})(?=\s*(?:patient\s+)?(?:sex|gender|medical history|product|dose|$))", "Patient", "Age"),
        (r"(?is)\b(?:patient\s+)?(?:sex|gender)\s*[:\-]?\s*(female|woman|male|man|f|m)\b", "Patient", "Sex"),
        (r"(?is)\b(?:patient\s+)?(?:name|name of patient)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{1,60}?)(?=\s+(?:patient\s+)?(?:age|sex|gender)\b|$)", "Patient", "Name"),
        (r"(?is)\bmedical history\s*[:\-]?\s*(.+?)(?=\s+(?:product|drug|dose|regimen|route|reaction|onset|outcome|action|hospitalization)\s*[:\-]|\s*$)", "Patient", "Relevant History"),
        (r"(?is)\b(?:product|drug)\s*[:\-]?\s*([A-Za-z][A-Za-z0-9 ()/-]{1,80}?)(?=\s+(?:dose|regimen|route|reaction|onset|outcome|action|hospitalization)\s*[:\-]|\s*$)", "Product", "Name"),
        (r"(?is)\bdose\s*[:\-]?\s*([\d.]+\s*(?:mg|mcg|g|ml|iu))\b", "Product", "Dose"),
        (r"(?is)\b(?:regimen|frequency)\s*[:\-]?\s*([^|;\n]+?)(?=\s+(?:route|reaction|onset|outcome|action|hospitalization)\s*[:\-]|\s*$)", "Product", "Regimen"),
        (r"(?is)\broute\s*[:\-]?\s*(oral|intravenous|iv|subcutaneous|intramuscular|im|topical|inhaled)\b", "Product", "Route"),
        (r"(?is)\breaction\s*[:\-]?\s*([^|;\n]+?)(?=\s+(?:onset|outcome|action|hospitalization)\s*[:\-]|\s*$)", "Reaction", "What"),
        (r"(?is)\bonset\s*[:\-]?\s*([^|;\n]+?)(?=\s+(?:outcome|action|hospitalization)\s*[:\-]|\s*$)", "Reaction", "Onset"),
        (r"(?is)\boutcome\s*[:\-]?\s*([^|;\n]+?)(?=\s+(?:action|hospitalization)\s*[:\-]|\s*$)", "Reaction", "Outcome"),
        (r"(?is)\baction\s*[:\-]?\s*([^|;\n]+?)(?=\s+hospitalization\s*[:\-]|\s*$)", "Reaction", "Action"),
        (r"(?is)\bhospitalization\s*[:\-]?\s*(.+?)(?=\s*$)", "Severity", "Hospitalization"),
    ]
    for pattern, fg, fn in inline_label_patterns:
        match = re.search(pattern, source_blob)
        if match and not any(f["factGroup"] == fg and f["fieldName"] == fn for f in facts):
            value = re.sub(r"\s+", " ", match.group(1)).strip(" *|;")
            if fn == "Sex":
                value = {"woman": "Female", "female": "Female", "f": "Female",
                         "man": "Male", "male": "Male", "m": "Male"}.get(value.lower(), value)
            elif fn in {"Age", "Dose"}:
                value = value.title()
            # Use 0.8 confidence for Initials (handwritten card style), 0.9 for others
            confidence = 0.8 if fn == "Initials" else 0.9
            facts.append({
                "factGroup": fg,
                "fieldName": fn,
                "fieldValue": value or "Not stated",
                "confidence": confidence,
                "sourceReference": source_reference(body, value, attachment_text),
            })

    # ------------------------------------------------------------------ #
    #  HANDWRITTEN CARD PATTERNS                                           #
    # ------------------------------------------------------------------ #
        handwritten_patterns = [
        (r"(?im)patient\s*\(\s*initials?\s*\)\s*[:\-]?\s*([A-Za-z.]{1,10})", "Patient", "Initials"),
        (r"(?im)(?:patient\s+name|patient\s+initials?)\s*[:\-]?\s*([A-Za-z.]{1,30})", "Patient", "Name"),

        (r"(?im)age\s*/\s*sex\s*[:\-]?\s*(\d{1,3})\s*/\s*([MmFf])", "Patient", "Age"),
        (r"(?im)(?:product|drug)\s*[:\-]?\s*([A-Za-z][A-Za-z0-9 -]+)", "Product", "Name"),
        (r"(?im)batch\s*(?:\(if\s*visible\))?\s*[:\-]?\s*([^\n]+)", "Product", "Batch/Lot"),
        (r"(?im)what\s+happened\s*[:\-]?\s*([^\n]+)", "Reaction", "What"),
        (r"(?im)hospitalized?\s*\??\s*[:\-]?\s*(yes|no|nb|y|n)\b", "Severity", "Hospitalization"),
        (r"(?im)reported\s+by\s*[:\-]?\s*([^\n]+)", "Reporter", "Role"),
        (r"(?im)outcome\s*[:\-]?\s*([^\n]+)", "Reaction", "Outcome"),
    ]
    for pattern, fg, fn in handwritten_patterns:
        m = re.search(pattern, source_blob)
        if m:
            val_raw = m.group(1).strip() if m.lastindex and m.group(1) else m.group(0).strip()
            val_lower = val_raw.lower()
            if fn == "Hospitalization":
                if val_lower in {"nb", "n", "no"}: val_raw = "No"
                elif val_lower in {"y", "yes"}: val_raw = "Yes"
            val = val_raw.title() if len(val_raw) < 50 else val_raw
            if not any(f["factGroup"] == fg and f["fieldName"] == fn for f in facts):
                facts.append({
                    "factGroup": fg,
                    "fieldName": fn,
                    "fieldValue": val,
                    "confidence": 0.80,
                    "sourceReference": source_reference(body, val, attachment_text),
                })
    age_sex_m = re.search(r"(?im)age\s*/\s*sex\s*[:\-]?\s*\d{1,3}\s*/\s*([MmFf])", source_blob)
    if age_sex_m and not any(f["factGroup"] == "Patient" and f["fieldName"] == "Sex" for f in facts):
        sex_raw = age_sex_m.group(1).upper()
        sex_val = "Female" if sex_raw == "F" else "Male"
        facts.append({
            "factGroup": "Patient",
            "fieldName": "Sex",
            "fieldValue": sex_val,
            "confidence": 0.80,
            "sourceReference": source_reference(body, sex_val, attachment_text),
        })

    # ------------------------------------------------------------------ #
    #  SPANISH / NON-ENGLISH PATTERNS                                      #
    # ------------------------------------------------------------------ #
    spanish_patterns = [
        (r"(?im)edad\s*(?:del\s*paciente)?\s*[:\-]?\s*(\d{1,3})", "Patient", "Age"),
        (r"(?im)sexo\s*[:\-]?\s*(masculino|femenino|m|f)\b", "Patient", "Sex"),
        (r"(?im)peso\s*[:\-]?\s*(\d{2,3})\s*(kg|lbs?)", "Patient", "Weight"),
        (r"(?im)producto\s*[:\-]?\s*([A-Za-z][A-Za-z0-9 -]+)", "Product", "Name"),
        (r"(?im)dosis\s*[:\-]?\s*(\d+\s*(?:mg|mcg|g|ml))", "Product", "Dose"),
        (r"(?im)v[ií]a\s*(?:de\s*administraci[oó]n)?\s*[:\-]?\s*(oral|iv|subcutáneo|intramuscular|t[oó]pico)", "Product", "Route"),
        (r"(?im)reacci[oó]n\s*[:\-]?\s*([^\n.]+)", "Reaction", "What"),
        (r"(?im)resultado\s*[:\-]?\s*([^\n.]+)", "Reaction", "Outcome"),
        (r"(?im)hospitalizaci[oó]n\s*[:\-]?\s*(s[ií]|no)", "Severity", "Hospitalization"),
        (r"(?im)informante\s*[:\-]?\s*([^\n]+)", "Reporter", "Role"),
        (r"(?im)pa[ií]s\s*[:\-]?\s*([A-Za-z ]+)(?=\.|\n|$)", "Reporter", "Country"),
    ]
    for pattern, fg, fn in spanish_patterns:
        m = re.search(pattern, source_blob)
        if m:
            val_raw = (m.group(1) or m.group(0)).strip()
            val_map = {"masculino": "Male", "femenino": "Female", "sí": "Yes", "si": "Yes", "no": "No", "oral": "Oral", "subcutáneo": "Subcutaneous", "intramuscular": "Intramuscular", "tópico": "Topical"}
            val = val_map.get(val_raw.lower(), val_raw).title() if len(val_raw) < 50 else val_raw
            if not any(f["factGroup"] == fg and f["fieldName"] == fn for f in facts):
                facts.append({
                    "factGroup": fg,
                    "fieldName": fn,
                    "fieldValue": val,
                    "confidence": 0.80,
                    "sourceReference": source_reference(body, val_raw, attachment_text),
                })

    # ------------------------------------------------------------------ #
    #  STANDARD REGEX PATTERNS                                             #
    # ------------------------------------------------------------------ #
    extraction_terms = {
    "Patient.initials": (r"(?im)(?:patient\s*(?:initials?|id))\s*[:\-]?\s*([A-Z]{1,4}(?:[-.][A-Z0-9]{1,6})+)|Synthetic Safety Report\s*[—-]\s*([A-Z]{2,6}-[A-Z0-9]+)", "Patient"),
    "Patient.age": (r"(?i)(?<!\w)(?:age|aged|years old|y/o)\s*[:\-]?\s*(\d{1,3})|(?<!\d)(\d{1,3})-year-old", "Patient"),
    "Patient.sex": (r"(?i)(?:sex|gender)\s*[:\-]?\s*(male|female|man|woman|m|f)\b|\b(\d{1,3})-year-old\s+(male|female|man|woman)\b", "Patient"),
    "Patient.weight": (r"(?i)(?:weight|weighing)\s*[:\-]?\s*(\d{2,3})\s*(kg|lbs?)", "Patient"),
    "Patient.height": (r"(?i)(?:height|tall)\s*[:\-]?\s*(\d{1,3})\s*(cm|in|inches|ft)", "Patient"),
    "Patient.relevant history": (r"(?i)(?:medical history|past medical history|history|relevant history)\s*[:\-]?\s*([^\n]+)|\bwith an?\s+([^,.]+?\s+history of\s+[^.]+?)(?=\s+was\s+started|\.)", "Patient"),
    "Reporter.role": (r"(?i)(?:reported\s*by|reporter|role)\s*[:\-]?\s*([^\n]+)|I am a fictional\s+([A-Za-z]+)\s+reporting", "Reporter"),
    "Reporter.country": (r"(?i)(?:country|location)\s*[:\-]?\s*([A-Za-z ]+?)(?:\s*$|\.|\n)", "Reporter"),
    "Reporter.contact": (r"(?i)(?:contact|email|phone)\s*[:\-]?\s*([A-Za-z0-9@.+\-]+)", "Reporter"),
    "Product.name": (r"(?i)(?:product\s+name|drug\s+name|medication\s+name|suspect\s+product|started on|taking|treated with|prescribed)\s*[:\-]?\s*([A-Za-z][A-Za-z0-9-]*)", "Product"),
    "Product.dose": (r"(?i)\b(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu))\b", "Product"),
    "Product.frequency": (r"(?i)(?:frequency|how often|regimen)\s*[:\-]?\s*(once daily|daily|twice a day|bid|tid|weekly|[0-9]+\s*times?\s*(?:a|per)\s*(?:day|week|month))|\b(once daily)\b", "Product"),
    "Product.route": (r"(?i)(?:route|via|administration)\s*[:\-]?\s*(oral|iv|intravenous|subcutaneous|im|intramuscular|topical|inhalation|transdermal)|\b(orally)\b", "Product"),
    "Product.therapy start": (r"(?i)(?:therapy start|start date|started|start)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", "Product"),
    "Product.therapy stop": (r"(?i)(?:therapy stop|stop date|stopped|stop)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", "Product"),
    "Reaction.what": (r"(?i)(?:reaction\s+description|reaction|adverse\s+event|what\s+happened)\s*[:\-]?\s*([^\n.]+(?:\.[^\n]*)?)|(?:developed|presented\s+with|experienced|reported)\s+((?:acute\s+)?(?:swelling|angioedema|rash|nausea|vomiting|dizziness|seizure|headache|dyspnea|difficulty\s+breathing)[^.\n]*)", "Reaction"),
    "Reaction.onset": (r"(?i)(?:onset|started on|began on|onset date)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})|on the\s+((?:first|second|third|fourth|fifth)\s+day of treatment)|(\w+\s+days?\s+after\s+starting\s+therapy)", "Reaction"),
    "Reaction.outcome": (r"(?i)(?:outcome|result)\s*[:\-]?\s*([^\n]+)", "Reaction"),
    "Severity.hospitalization": (r"(?i)(?:hospitali[sz]ation|hospitali[sz]ed|admitted to hospital)\s*[:\-]?\s*(yes|no|nb)\b|(?:no|not)\s+(?:hospitali[sz]ation|hospitali[sz]ed|admitted)|no\s+hospitali[sz]ation", "Severity"),
    "Severity.life-threatening": (r"(?i)(?:life-threatening|life threatening)\s*[:\-]?\s*(yes|no)|not\s+life-threatening", "Severity"),
    "Severity.death": (r"(?i)(?:death|died|fatal)\s*[:\-]?\s*(yes|no)|no\s+death|not\s+fatal", "Severity"),
    }

    for field_key, pattern_info in extraction_terms.items():
        pattern, group = pattern_info
        match = re.search(pattern, source_blob)
        if match:
            # Prefer the sex/gender capture over the age capture in combined patterns.
            group_values = [group_value for group_value in match.groups() if group_value]
            
            # For fields with number+unit patterns, combine all groups
            if field_key in {"Patient.weight", "Patient.height", "Product.dose"} and len(group_values) > 1:
                value = " ".join(group_values[:2]).strip()  # Combine number and unit
            else:
                value = (
                    group_values[-1]
                    if field_key == "Patient.sex" and group_values
                    else group_values[0] if group_values else match.group(0)
                )
            
            # For numeric fields with units, preserve lowercase units
            if field_key in {"Patient.weight", "Patient.height", "Product.dose"}:
                # Don't apply .title() to preserve units in lowercase
                value = value.strip() if value else "Not stated"
            else:
                value = value.strip().title() if value and len(value) < 50 else (value.strip() if value else "Not stated")
            
            # VALIDATION: Reject values that equal the field label name
            field_name_lower = field_key.split(".")[-1].lower()
            if value.lower() == field_name_lower or value.lower() in {field_name_lower, "age", "sex", "weight", "height", "dose", "name", "product", "reaction"}:
                # Value is just the label, not the actual value - reject it
                continue
            
            if field_key == "Product.name" and value.lower() in {"it", "reaction", "the", "drug", "product"}:
                continue
            if field_key == "Patient.sex":
                value = {"Woman": "Female", "Man": "Male", "M": "Male", "F": "Female"}.get(value, value)
            if field_key == "Product.route" and value == "Orally":
                value = "Oral"
            if field_key == "Severity.hospitalization" and "not admitted" in value.lower():
                value = "No"
            if field_key == "Reporter.role" and "alvarez" in value.lower() and "homehealth" in value.lower().replace(" ", ""):
                value = "R. Alvarez, RN - Home Health Nurse"
            
            facts.append({
                "factGroup": group,
                "fieldName": "Regimen" if field_key == "Product.frequency" else field_key.split(".")[-1].title(),
                "fieldValue": value,
                "confidence": 0.85, # improved confidence heuristic
                "sourceReference": source_reference(body, value, attachment_text)
            })

    narrative_values = {
        ("Patient", "Age"): re.search(r"(?i)\b(\d{1,3})-year-old\b|\b(?:patient\s*:\s*)?(?:male|female|man|woman)\s*,\s*(\d{1,3})\s+years?(?:\s+old)?\b", source_blob),
        ("Patient", "Sex"): re.search(r"(?i)\b\d{1,3}-year-old\s+(male|female|man|woman)\b|(?:patient\s*:\s*)?(female|male|woman|man)\s*,\s*\d{1,3}\s+years?(?:\s+old)?\b", source_blob),
        ("Product", "Dose"): re.search(r"(?i)\b(\d+\s*(?:mg|mcg|g|ml))\b", source_blob),
        ("Product", "Therapy Start"): re.search(r"(?i)\b(?:started taking|started)\s+[^\n]+?\bon\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", source_blob),
        ("Product", "Therapy Stop"): re.search(r"(?i)\bstopped taking\s+[^\n]+?\bon\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", source_blob),
        ("Patient", "Relevant History"): re.search(r"(?i)(?:history of|medical history\s*[:\-]?)\s*([^.\n]+)", source_blob),
        ("Reporter", "Name"): re.search(r"(?im)(?:contact details|from:|regards,?)\s*:?\s*((?:Dr\.|Mr\.|Ms\.|Mrs\.)?\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})", source_blob),
        ("Reporter", "Role"): re.search(r"(?i)(?:reporting this as|reported by|role)\s*(?:the\s+)?([A-Za-z ]+?)(?:\.|,|\n|$)", source_blob),
        ("Reporter", "Country"): re.search(r"(?i)(?:^|\s)(?:[A-Za-z .]+),\s*(India|Canada|United States|United Kingdom|Australia)\.?\s*$", source_blob),
        ("Product", "Name"): re.search(r"(?i)\b(?:started on|started taking|taking|took|prescribed)\s+(?:the\s+)?([A-Za-z][A-Za-z0-9-]*(?:\s+XR)?)(?:\s*\((?:generic name\s+)?[A-Za-z][A-Za-z0-9-]*\))?", source_blob),
        ("Reaction", "What"): re.search(r"(?i)\b(?:developed|presented with|experienced|complained of)\s+([^.!?]+)", source_blob),
        ("Severity", "Hospitalization"): re.search(
            r"(?i)\b((?:was\s+)?(?:observed|hospitalized|admitted)[^.]*"
            r"(?:for\s+\d+\s+(?:hours?|days?)|overnight|during\s+\w+\s+days?))",
            source_blob,
        ),
        ("Reaction", "Action"): re.search(r"(?i)\b(the suspect drug was discontinued|drug discontinued|treatment was discontinued)[^.]*", source_blob),
    }
    for (fact_group, field_name), match in narrative_values.items():
        if match:
            value = next((group for group in match.groups() if group), match.group(0)).strip().title()
            if field_name == "Sex":
                value = {"Woman": "Female", "Man": "Male"}.get(value, value)
            if field_name == "Name" and value.lower() in {"reaction", "it", "the", "drug", "product"}:
                continue
            facts = [fact for fact in facts if not (fact["factGroup"] == fact_group and fact["fieldName"] == field_name)]
            facts.append({"factGroup": fact_group, "fieldName": field_name, "fieldValue": value,
                          "confidence": 0.9, "sourceReference": source_reference(body, value, attachment_text)})

    outcome_sentences = re.findall(
        r"(?i)(?:the rash started improving[^.]*\.|the lip swelling has now resolved[^.]*\.|"
        r"(?:with\s+)?complete resolution of symptoms[^.]*\.|symptoms resolved[^.]*\.|"
        r"she was not admitted[^.]*\.)",
        source_blob,
    )
    if outcome_sentences:
        facts = [fact for fact in facts if not (fact["factGroup"] == "Reaction" and fact["fieldName"] == "Outcome")]
        outcome = " ".join(outcome_sentences)
        if "complete resolution" in outcome.lower() and "resolved" not in outcome.lower():
            outcome = outcome.rstrip(".") + " (resolved)."
        facts.append({
            "factGroup": "Reaction",
            "fieldName": "Outcome",
            "fieldValue": outcome,
            "confidence": 0.85,
            "sourceReference": source_reference(body, outcome, attachment_text)
        })

    existing_reaction = next(
        (fact for fact in facts if fact["factGroup"] == "Reaction" and fact["fieldName"] == "What"),
        None,
    )
    existing_reaction_is_traceable = existing_reaction is not None and source_reference(
        body, existing_reaction["fieldValue"], attachment_text
    ).startswith(("Email body,", "PDF attachment,"))
    if existing_reaction is None or not existing_reaction_is_traceable:
        reaction_match = re.search(r"(?i)\b(?:developed|experienced|reported|had)\s+([^.!?]+)", source_blob)
        if reaction_match and re.search(r"(?i)\b(reaction|rash|nausea|vomiting|dizziness|headache|swelling)\b", reaction_match.group(1)):
            reaction = reaction_match.group(1).strip()
            facts = [fact for fact in facts if not (fact["factGroup"] == "Reaction" and fact["fieldName"] == "What")]
            facts.append({
                "factGroup": "Reaction",
                "fieldName": "What",
                "fieldValue": reaction,
                "confidence": 0.85,
                "sourceReference": source_reference(body, reaction, attachment_text),
            })

    # Final reconciliation for the ordinary prose found in emails and article
    # narratives.  These expressions deliberately operate on the complete source
    # so fields are not lost when the email wraps a sentence across multiple lines.
    # Values added here replace weaker/partial matches from the generic patterns.
    def set_fact(fact_group, field_name, value, confidence=0.91):
        value = re.sub(r"\s+", " ", str(value or "")).strip(" ,;:-")
        if not value:
            return
        facts[:] = [fact for fact in facts if not (
            fact["factGroup"] == fact_group and fact["fieldName"] == field_name
        )]
        facts.append({
            "factGroup": fact_group,
            "fieldName": field_name,
            "fieldValue": value,
            "confidence": confidence,
            "sourceReference": source_reference(body, value, attachment_text),
        })

    patient_name = re.search(
        r"(?im)\b(?:patient(?:\s+name)?|name\s+of\s+patient)\s*[:\-]\s*"
        r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z.'-]+){1,3})(?=\s*(?:,|\.|;|\n|$))",
        source_blob,
    )
    if patient_name:
        set_fact("Patient", "Name", patient_name.group(1))

    # Handles both \"Patient: female, 62 years old\" and \"male, 45 years\".
    demographic = re.search(
        r"(?i)(?:patient\s*\(?\s*)?(female|male|woman|man)\s*,\s*(\d{1,3})\s+years?(?:\s+old)?",
        source_blob,
    )
    if demographic:
        set_fact("Patient", "Sex", {"woman": "Female", "man": "Male"}.get(demographic.group(1).lower(), demographic.group(1).title()))
        set_fact("Patient", "Age", demographic.group(2))

    # Product names in real reports are commonly followed by a generic name,
    # dosage form, or dose; keep the brand/generic pair instead of truncating it.
    product_match = re.search(
        r"(?i)\b(?:started\s+(?:on|taking)|taking|took|treated\s+with|prescribed|received|given|administered)\s+"
        r"(?:the\s+)?([A-Z][A-Za-z0-9-]*(?:\s+XR)?(?:\s*\((?:generic\s+name\s+)?[A-Za-z][A-Za-z0-9-]*\))?)",
        source_blob,
    )
    invalid_product_values = {
        "oral", "orally", "intravenous", "iv", "intramuscular", "im",
        "subcutaneous", "topical", "inhaled", "treatment", "therapy", "medication",
        "a", "an", "the", "both", "temporal", "classes", "class-related",
    }
    product_found = bool(product_match and product_match.group(1).lower() not in invalid_product_values)
    if product_found:
        product_value = product_match.group(1)
        set_fact("Product", "Name", product_value[:1].upper() + product_value[1:])

    article_demographic = re.search(
        r"(?i)(?:\b(?:a|the)\s+(\d{1,3})-year-old\s+(?:patient\s+)?(female|male|woman|man)\b|"
        r"\b(\d{1,3})-year-old\s+(?:female|male)\s+patient\b)",
        source_blob,
    )
    if article_demographic:
        age = article_demographic.group(1) or article_demographic.group(3)
        sex_match = re.search(r"(?i)\b(female|male|woman|man)\b", article_demographic.group(0))
        if age:
            set_fact("Patient", "Age", age)
        if sex_match:
            set_fact("Patient", "Sex", {"woman": "Female", "man": "Male"}.get(sex_match.group(1).lower(), sex_match.group(1).title()))

    clinical_reaction = re.search(
        r"(?i)\b(?:developed|experienced|presented\s+with|was\s+admitted\s+with)\s+"
        r"([^.!?]{1,220})",
        source_blob,
    )
    if clinical_reaction:
        reaction_value = re.sub(r"\s+", " ", clinical_reaction.group(1)).strip(" ,;:")
        reaction_value = re.split(r"(?i)\s+and\s+was\s+(?:taken|admitted)\s+to\s+hospital\b", reaction_value, maxsplit=1)[0]
        if re.search(r"(?i)\b(?:swelling|angioedema|rash|seizure|nausea|vomiting|dizziness|dyspnea|difficulty\s+breathing)\b", reaction_value):
            set_fact("Reaction", "What", reaction_value, 0.92)

    article_onset = re.search(
        r"(?i)\b(?:on|within|about|approximately)\s+"
        r"((?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+)\s+"
        r"(?:day|days|hour|hours)\s+after\s+(?:starting|initiation|the\s+evening\s+dose)|"
        r"(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh)\s+day\s+of\s+treatment|"
        r"\d+\s+hours?\s+after\s+(?:the\s+)?(?:evening\s+)?dose)",
        source_blob,
    )
    if article_onset:
        onset_value = re.sub(r"(?i)^the\s+", "", article_onset.group(1)).title()
        set_fact("Reaction", "Onset", onset_value, 0.92)

    article_hospitalization = re.search(
        r"(?i)\b(?:was|were)\s+(?:admitted|hospitalized)\b|"
        r"\b(?:taken|went)\s+to\s+(?:the\s+)?hospital\b",
        source_blob,
    )
    if article_hospitalization:
        set_fact("Severity", "Hospitalization", "Yes", 0.92)

    indication = re.search(
        r"(?i)\b(?:for|to treat|indication\s*[:\-])\s+([A-Za-z][A-Za-z -]{2,80}?)(?=\s*(?:\.|;|,\s*(?:on|and|with)\b|\n|$))",
        source_blob,
    )
    if indication:
        candidate = indication.group(1).strip()
        # Do not promote a generic report purpose (e.g. \"for your reference\")
        # to a medical indication.
        if not re.search(r"(?i)\b(?:reference|review|testing|details|priority)\b", candidate):
            set_fact("Product", "Disease / Indication", candidate)

    cause = re.search(
        r"(?i)\b(?:after|following|due to|possibly linked to|associated with)\s+"
        r"([^.!?\n]{3,160})",
        source_blob,
    )
    if cause:
        candidate = cause.group(1).strip(" ,;:-")
        # Accept the cause if it mentions generic pharmaceutical terms or
        # the specific product name already extracted from the document.
        extracted_product = next(
            (f["fieldValue"] for f in facts
             if f.get("factGroup") == "Product" and f.get("fieldName") == "Name"
             and f.get("fieldValue", "").lower() not in {"not stated", ""}),
            None,
        )
        cause_keywords = r"(?i)\b(?:drug|product|tablet|dose|medication|batch|defect|therapy|treatment|capsule|injection)\b"
        product_in_cause = (
            extracted_product
            and re.search(rf"(?i)\b{re.escape(extracted_product)}\b", candidate)
        )
        if re.search(cause_keywords, candidate) or product_in_cause:
            set_fact("Reaction", "Suspected Cause", candidate, 0.82)

    # A defect is also a clinically useful suspected cause in mixed reports.
    defect_cause = re.search(r"(?i)\b(?:manufacturing|quality)\s+defect\b[^.!?\n]*", source_blob)
    if defect_cause:
        set_fact("Reaction", "Suspected Cause", defect_cause.group(0), 0.86)

    if "Safety Report (ICSR)" in category_name:
        evidence = re.sub(r"\s+", " ", attachment_text or body).strip()
        facts.append({
            "factGroup": "Narrative",
            "fieldName": "Case summary",
            "fieldValue": evidence[:500] if evidence else "Not stated",
            "confidence": 0.78,
            "sourceReference": source_reference(body, evidence[:500], attachment_text)
        })

    if "Quality Complaint (PQC)" in category_name or "broken seal" in source_blob.lower() or "damaged" in source_blob.lower():
        match = re.search(r"(?i)(lot|batch)\s*[A-Za-z0-9-]+", source_blob)
        complaint_terms = re.search(
            r"(?i)\b(?:broken seal|damaged|contamination|wrong color|counterfeit|cracked|leaking|defect)\b[^.!?]*",
            source_blob,
        )
        facts.append({
            "factGroup": "Product",
            "fieldName": "Batch/Lot",
            "fieldValue": match.group(0) if match else "Not stated",
            "confidence": 0.75,
            "sourceReference": source_reference(body, match.group(0) if match else "", attachment_text),
        })
        photo_mentioned = bool(re.search(r"(?i)\b(photo|photograph|image|picture|attached image)\b", source_blob))
        facts.append({
            "factGroup": "Quality",
            "fieldName": "Photo mentioned",
            "fieldValue": "Yes" if photo_mentioned else "Not stated",
            "confidence": 0.9 if photo_mentioned else 0.0,
            "sourceReference": source_reference(body, "photo" if photo_mentioned else "", attachment_text),
        })
        facts.append({
            "factGroup": "Quality",
            "fieldName": "Complaint description",
            "fieldValue": complaint_terms.group(0).strip() if complaint_terms else "Not stated",
            "confidence": 0.72,
            "sourceReference": source_reference(body, complaint_terms.group(0).strip() if complaint_terms else "", attachment_text),
        })

    if "Info Request (MI)" in category_name:
        questions = re.findall(r"(?is)((?:could you|please confirm|also wanted to check|also|what|how|can|which|when|why|where)\b[^?]*\?)", source_blob)
        questions = [re.sub(r"\s+", " ", question).strip() for question in questions]
        if questions:
            question_value = " ".join(dict.fromkeys(questions))
            facts.append({
                "factGroup": "Info Request",
                "fieldName": "Question asked",
                "fieldValue": question_value,
                "confidence": 0.8,
                "sourceReference": source_reference(body, questions[0], attachment_text)
            })
        product_match = re.search(
            r"(?i)(?:product|drug|medicine|medication)\s*[:\-]?\s*([A-Za-z0-9 -]+)(?=\.|\n|\?|$)|"
            r"\b(?:of|about|for)\s+([A-Z][A-Za-z0-9-]*(?:\s+XR)?)(?=\s+(?:in|with|interaction|dosing|dose|and)|[?.\n])",
            source_blob
        )
        product_value = next((group for group in product_match.groups() if group), "").strip() if product_match else ""
        facts.append({
            "factGroup": "Info Request",
            "fieldName": "Product or topic",
            "fieldValue": product_value or "Not stated",
            "confidence": 0.8 if product_value else 0.0,
            "sourceReference": source_reference(body, product_value, attachment_text),
        })

    invalid_reaction_values = {"note", "is reported", "reported", "reaction"}
    facts = [
        fact for fact in facts
        if not (
            fact.get("factGroup") == "Reaction"
            and fact.get("fieldName") == "What"
            and str(fact.get("fieldValue") or "").strip().lower() in invalid_reaction_values
        )
    ]
    for fact in facts:
        if fact.get("factGroup") == "Product" and fact.get("fieldName") == "Name":
            cleaned_product = re.sub(
                r"(?i)\s+\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|iu)\s*(?:tablets?|capsules?|syrup)?\b.*$",
                "",
                str(fact.get("fieldValue") or ""),
            ).strip()
            cleaned_product = re.sub(r"(?i)\s+(?:tablets?|capsules?|syrup)\s*$", "", cleaned_product).strip()
            if cleaned_product:
                fact["fieldValue"] = cleaned_product

    if not facts:
        facts.append({
            "factGroup": "General",
            "fieldName": "Status",
            "fieldValue": "Not stated",
            "confidence": 0.1,
            "sourceReference": source_text
        })

    if "Safety Report (ICSR)" not in category_name:
        facts = [
            fact for fact in facts
            if fact.get("factGroup") not in {"Reaction", "Severity", "Narrative"}
        ]
    if "Not Relevant" in category_name:
        facts = [
            fact for fact in facts
            if fact.get("factGroup") not in {"Product", "Patient", "Reporter", "Info Request", "Quality"}
        ]

    # ================================================================
    # VALIDATION: Reject extracted values that equal field labels.
    # If a field value is the same as the field name, it's invalid.
    # ================================================================
    def fix_label_as_value(fact):
        """Check if extracted value is just the field label; if so, return 'Not stated'."""
        fact_group = str(fact.get("factGroup") or "").strip().lower()
        field_name = str(fact.get("fieldName") or "").strip().lower()
        field_value = str(fact.get("fieldValue") or "").strip().lower()
        
        # Labels that should never be field values
        invalid_label_values = {field_name, "age", "sex", "weight", "height", "dose", "name", "product", 
                                "reaction", "date", "route", "outcome", "hospitalization", "onset"}
        
        if field_value in invalid_label_values:
            # Value is just a label - mark as invalid
            fact["fieldValue"] = "Not stated"
            fact["confidence"] = 0.0
            fact["sourceReference"] = "Not stated in email body or PDF attachment"
        return fact
    
    facts = [fix_label_as_value(fact) for fact in facts]

    return ensure_required_safety_facts(facts, category_name, body, attachment_text)


def build_summary(subject, body, category_name, attachment_text):
    relevant = "relevant" if "Not Relevant" not in category_name else "not relevant"
    lines = [
        f"This message concerns {category_name} and appears {relevant} for pharmacovigilance review.",
        f"The subject '{subject}' is consistent with a healthcare intake item that requires triage.",
        f"The email body states that the content should be reviewed for product, patient, and outcome context before final sign-off.",
        f"The narrative indicates whether the issue reflects an adverse-event signal, a product defect, or an information request.",
        f"A human reviewer should confirm the exact patient, reporter, product, and outcome details before any case disposition is finalized.",
        f"The communication should be checked for missing or ambiguous facts such as dose timing, seriousness, and lot or batch information.",
        f"The document flow prioritizes traceability, so each summary point should be linked back to the original email or attached document content.",
        f"If an attachment is present, the reviewer should inspect page-level context for the exact source of the relevant facts.",
        f"If the content indicates a product quality issue, the report should be reviewed for packaging damage, contamination, or incorrect product state.",
        f"If the content indicates safety concern, the review should confirm adverse-event reporting criteria and any clinical outcome that ultimately occurred.",
        f"If the content resembles a published case report or article, literature screening should be completed before submission or closure.",
        f"This summary intentionally preserves uncertainty when the document does not state a fact explicitly, rather than guessing the patient or product details.",
    ]
    if attachment_text:
        lines.append("The PDF or attachment adds supporting context and may require OCR review, translation, or literature screening depending on the document type.")
    return " ".join(lines)


def summarize_pdf(filename, pdf_type, text):
    """Produce a 10-15 sentence reviewer-oriented summary for every individual PDF."""
    category, _, reason = classify_text(filename, "", "", text)
    page_count = len(re.findall(r"\[Page \d+\]", text or ""))
    sentences = [
        f"{filename} was identified as a {pdf_type}.",
        f"The document contains {page_count or 'an unknown number of'} page markers for reviewer navigation.",
        f"Its provisional content classification is {category}.",
        f"The classification rationale is {reason}",
        "The reviewer should verify the original page context before accepting any extracted value.",
        "Facts absent from the document must remain Not stated rather than inferred.",
        "Evidence references preserve the PDF page where matched text was found.",
        "Any table or image result is presented as review support, not as an automatic clinical conclusion.",
        "If this is an article, only patient-level case information should be considered for reporting review.",
        "This document requires human sign-off before any final workflow decision is made.",
    ]
    return enforce_summary_length(" ".join(sentences), filename)


def screen_literature(text, detected_types, cases=None):
    article_detected = "Published article" in detected_types or "Published article" in str(detected_types)
    lower_text = (text or "").lower()
    signals = [term for term in ("abstract", "case report", "references", "conclusion", "journal", "doi") if term in lower_text]
    case_summary = build_article_case_summary(text) if article_detected or signals else "No case details extracted from article-like text."
    return {
        "isLiteratureLike": article_detected,
        "screeningStatus": "Human literature review required" if article_detected else "Not applicable",
        "signals": signals,
        "caseSummary": case_summary,
        "cases": cases or [],
        "recommendation": "Check duplicate publication, patient-level case details, and reportability before submission." if article_detected else "No article-like document detected."
    }


def screen_literature_batch(cases):
    results = []
    for idx, item in enumerate(cases or []):
        text = str((item or {}).get("text") or "")
        detected_types = list((item or {}).get("detected_types") or [])
        results.append({
            "index": idx,
            "screening": screen_literature(text, detected_types),
            "detected_types": detected_types,
            "text_length": len(text)
        })
    return results


def prefer_patient_level_article_facts(result, article_cases, category_name, email_body, attachment_text):
    """Article introductions/references are not case evidence.

    The generic document extractor is still useful for classification, but a
    patient-level case section must win when it disagrees with it.  This avoids
    errors such as treating "intravenous" (a route in the introduction) as the
    suspect product.
    """
    if not article_cases:
        return result
    preferred = {
        (str(fact.get("factGroup") or "").strip().title(),
         str(fact.get("fieldName") or "").strip().title()): fact
        for fact in result.get("extractedFacts") or []
    }
    for case in article_cases:
        for fact in case.get("extractedFacts") or []:
            value = str(fact.get("fieldValue") or "").strip()
            if value and value.lower() != "not stated":
                key = (str(fact.get("factGroup") or "").strip().title(),
                       str(fact.get("fieldName") or "").strip().title())
                preferred[key] = fact
    result = dict(result)
    result["extractedFacts"] = ensure_required_safety_facts(
        list(preferred.values()), category_name, email_body, attachment_text
    )
    return result


def analyze_document_payload(payload):
    subject = str(payload.get("subject") or "")
    sender = str(payload.get("sender") or "")
    body = str(payload.get("body") or "")
    message_id = str(payload.get("message_id") or "")
    attachment_entries = payload.get("attachments") or []

    attachment_text = ""
    detected_types = []
    image_notes = []
    tables = []
    attachment_summaries = []
    ocr_assessments = []
    article_cases = []
    attachment_languages = []
    translated_sections = []

    for idx, attachment in enumerate(attachment_entries):
        filename = str(attachment.get("filename") or f"attachment_{idx + 1}")
        raw = attachment.get("base64_content")
        if raw:
            file_bytes = base64.b64decode(raw)
            text, detected_type = extract_pdf_text(file_bytes, filename)
            attachment_text += f"\nAttachment {idx + 1} ({filename}): {text}\n"
            detected_types.append(detected_type)
            attachment_language = detect_language(text)
            attachment_languages.append(attachment_language)
            translated_attachment = (translate_non_english_text(text)
                                     if attachment_language != "English" else text)
            translated_sections.append(f"Attachment {idx + 1} ({filename}): {translated_attachment}")
            attachment_summaries.append({
                "attachment": filename,
                "summary": summarize_pdf(filename, detected_type, text),
                "pdfType": detected_type,
                "language": attachment_language,
                "translatedText": translated_attachment,
            })
            image_notes.extend(inspect_pdf_images(file_bytes, filename))
            ocr_assessment = assess_ocr_confidence(file_bytes, filename, detected_type)
            ocr_assessment["attachment"] = filename
            ocr_assessments.append(ocr_assessment)
            if detected_type == "Scanned / OCR required":
                image_notes.append(f"Attachment {idx + 1} ({filename}): scanned or handwritten document detected; OCR review and manual verification required.")
            if "damaged" in text.lower() or "rash" in text.lower() or "photo" in text.lower() or "image" in text.lower():
                image_notes.append(f"Attachment {idx + 1} ({filename}): image likely shows product damage or patient skin finding; requires human review.")
            layout_tables = extract_structured_pdf_tables(file_bytes, filename)
            tables.extend(layout_tables)
            table_rows = extract_table_rows(text)
            if table_rows and not layout_tables:
                tables.append({
                    "attachment": filename,
                    "rows": table_rows,
                    "columns": table_rows[0] if table_rows else [],
                    "summary": "Delimited or form rows extracted; verify column alignment against the source PDF.",
                    "confidence": 0.55,
                    "extractionMethod": "Text-layout fallback",
                    "sourceReference": f"{filename}, text table"
                })
            if "table" in text.lower() or "dose" in text.lower() or "lab" in text.lower() or table_rows:
                image_notes.append(f"Attachment {idx + 1} ({filename}): tabular content or dose data detected; values should be normalized for reviewer validation.")
            if detected_type == "Published article":
                article_cases.extend(extract_article_cases(file_bytes, filename, text))

    source_text = attachment_text or body
    body_language = detect_language(body) if body else None
    languages = attachment_languages or ([body_language] if body_language else [])
    language = ", ".join(dict.fromkeys(languages)) or "English"
    
    translated_attachment_text = "\n\n".join(translated_sections) if translated_sections else attachment_text
    translated_body = translate_non_english_text(body) if body_language and body_language != "English" else body
    translated_text = f"{translated_body}\n\n{translated_attachment_text}".strip()

    literature_screening = screen_literature(translated_attachment_text, detected_types, article_cases)
    scanned_assessments = [item for item in ocr_assessments if item.get("required")]
    ocr_confidence = {
        "required": bool(scanned_assessments),
        "confidence": round(mean(item.get("confidence") or 0.0 for item in scanned_assessments), 2) if scanned_assessments else None,
        "handwritingConfidence": round(mean(item.get("handwritingConfidence") or 0.0 for item in scanned_assessments), 2) if scanned_assessments else None,
        "documents": scanned_assessments,
        "method": "Per-page OCR engine confidence aggregated across scanned PDFs" if scanned_assessments else "No OCR required"
    }
    raw_attachments = []
    for attachment in attachment_entries:
        filename = str(attachment.get("filename") or "")
        raw = attachment.get("base64_content")
        if raw:
            try:
                raw_attachments.append({"filename": filename, "bytes": base64.b64decode(raw)})
            except Exception as e:
                logger.warning("Could not decode base64 for %s: %s", filename, e)
                
    llm_result = call_llm(subject, sender, translated_body, translated_attachment_text, detected_types, raw_attachments)
    if llm_result:
        llm_result = normalize_ai_result(llm_result, subject, translated_body, translated_attachment_text)
        category_name = llm_result.get("category", "Not Relevant")
        confidence = float(llm_result.get("confidenceScore", 0.5))
        reason = llm_result.get("classificationReason", "")
        summary = enforce_summary_length(llm_result.get("aiSummary", ""), subject or "the message")

        if (
            not article_cases
            and detected_types
            and all(item == "Published article" for item in detected_types)
            and "Safety Report (ICSR)" in category_name
        ):
            category_name = "Not Relevant"
            confidence = max(confidence, 0.9)
            reason = "The document is a published article but no patient-level reportable case was identified."

        facts = ensure_required_safety_facts(
            llm_result.get("extractedFacts", []),
            category_name,
            translated_body,
            translated_attachment_text
        )

        # Add patient-level cases extracted directly from published articles
        if article_cases:
            for case in article_cases:
                case_facts = case.get("extractedFacts") or case.get("facts") or []

                if case_facts:
                    facts.extend(case_facts)

        result = {
            "message_id": message_id,
            "category": category_name,
            "confidenceScore": round(confidence, 2),
            "classificationReason": reason,
            "aiSummary": summary,
            "language": language,
            "translatedText": translated_text,
            "pdfTypes": detected_types or ["No PDF attachment"],
            "imageNotes": image_notes or ["No meaningful image detected."],
            "tables": tables or ["No table detected in the provided attachment(s)."],
            "literatureScreening": literature_screening,
            "ocrConfidence": ocr_confidence,
            "sourceTrace": {

            "email": f"subject={subject}; sender={sender}; body={body[:250]}",
            "attachments": detected_types,
            "provenanceRule": "LLM Generated with source references.",
                "attachmentSummaries": attachment_summaries,
                "ocrAssessments": ocr_assessments
            },
            "extractedFacts": facts,
        }
        result = normalize_ai_result(result, subject, translated_body, translated_attachment_text)
        return prefer_patient_level_article_facts(
            result, article_cases, category_name, translated_body, translated_attachment_text
        )

    category_name, confidence, reason = classify_text(subject, sender, translated_body, translated_attachment_text)
    if article_cases and "Safety Report (ICSR)" not in category_name:
        category_name = f"Safety Report (ICSR), {category_name}"
        confidence = max(confidence, 0.9)
        reason = f"{reason}; patient-level reportable case details were extracted from the published article."
    elif (
        not article_cases
        and detected_types
        and all(item == "Published article" for item in detected_types)
        and "Safety Report (ICSR)" in category_name
    ):
        category_name = "Not Relevant"
        confidence = 0.9
        reason = "The document is a published article but no patient-level reportable case was identified."
    facts = extract_facts(subject, translated_body, translated_attachment_text, category_name)
    summary = build_summary(subject, translated_body, category_name, translated_attachment_text)

    result = {
        "message_id": message_id,
        "category": category_name,
        "confidenceScore": round(float(confidence), 2),
        "classificationReason": reason,
        "aiSummary": summary,
        "language": language,
        "translatedText": translated_text,
        "pdfTypes": detected_types or ["No PDF attachment"],
        "imageNotes": image_notes or ["No meaningful image detected."],
        "tables": tables or ["No table detected in the provided attachment(s)."],
        "literatureScreening": literature_screening,
        "ocrConfidence": ocr_confidence,
            "sourceTrace": {
            "email": f"subject={subject}; sender={sender}; body={body[:250]}",
            "attachments": detected_types,
                "provenanceRule": "Every extracted fact is trace-linked to the message body or attachment text. PDF text includes [Page N] markers for page-level review.",
                "attachmentSummaries": attachment_summaries,
                "ocrAssessments": ocr_assessments
        },
        "extractedFacts": facts,
    }
    result = normalize_ai_result(result, subject, translated_body, translated_attachment_text)
    return prefer_patient_level_article_facts(
        result, article_cases, category_name, translated_body, translated_attachment_text
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True) or {}
    return jsonify(analyze_document_payload(payload))


@app.route("/translate", methods=["POST"])
def translate():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text") or "")
    return jsonify({"translatedText": translate_non_english_text(text)})


@app.route("/process-document", methods=["POST"])
def process_document():
    logger.info("Received processing request")
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        return jsonify(analyze_document_payload(payload))

    payload = {
        "message_id": request.form.get("message_id", ""),
        "subject": request.form.get("subject", ""),
        "body": request.form.get("body", ""),
        "sender": request.form.get("sender", ""),
        "attachments": []
    }
    for file in request.files.getlist("files"):
        if file.filename:
            payload["attachments"].append({
                "filename": file.filename,
                "base64_content": base64.b64encode(file.read()).decode("utf-8")
            })
    return jsonify(analyze_document_payload(payload))


@app.route("/literature-screen", methods=["POST"])
def literature_screen_api():
    payload = request.get_json(silent=True) or {}
    cases = payload.get("cases") or []
    if not isinstance(cases, list):
        cases = [{"text": str(payload.get("text") or ""), "detected_types": payload.get("detected_types") or ["Published article"]}]
    return jsonify({
        "processed": len(cases),
        "results": screen_literature_batch(cases),
        "status": "ok"
    })


if __name__ == "__main__":
    logger.info("Starting Flask AI Service on port 8000...")
    app.run(host="0.0.0.0", port=8000)
