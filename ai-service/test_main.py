import os
import unittest

from main import (
    analyze_document_payload,
    build_summary,
    classify_text,
    classify_pdf_type,
    extract_table_rows,
    extract_facts,
    get_gemini_api_key,
    screen_literature,
    sentence_count,
    translate_non_english_text,
)


class ClassifierTests(unittest.TestCase):
    def test_question_is_not_safety_report(self):
        category, _, _ = classify_text(
            "Dose question",
            "doctor@example.com",
            "What dose should a patient take?",
            "",
        )
        self.assertEqual(category, "Info Request (MI)")

    def test_adverse_event_is_safety_report(self):
        category, _, _ = classify_text(
            "Adverse event",
            "doctor@example.com",
            "A patient developed nausea after taking Aspirin.",
            "",
        )
        self.assertEqual(category, "Safety Report (ICSR)")

    def test_safety_schema_includes_honest_missing_fields_and_provenance(self):
        facts = extract_facts(
            "Adverse event", "A patient developed nausea after Aspirin.", "", "Safety Report (ICSR)"
        )
        by_field = {(fact["factGroup"], fact["fieldName"]): fact for fact in facts}
        self.assertIn(("Patient", "Age"), by_field)
        self.assertEqual(by_field[("Patient", "Age")]["fieldValue"], "Not stated")
        self.assertEqual(by_field[("Patient", "Age")]["confidence"], 0.0)
        reaction = by_field[("Reaction", "What")]
        self.assertIn("Email body, sentence", reaction["sourceReference"])

    def test_safety_and_quality_are_multilabel(self):
        category, _, _ = classify_text(
            "Reaction plus defect",
            "pharmacy@example.com",
            "A patient developed nausea and the product has a broken seal.",
            "",
        )
        self.assertEqual(category, "Safety Report (ICSR), Quality Complaint (PQC)")

    def test_info_request_with_explicit_negative_signals_is_not_quality_complaint(self):
        category, _, _ = classify_text(
            "04_digital_info_request",
            "synthetic.sender@example.invalid",
            "Synthetic fixture message for assignment validation.",
            "[Page 1]\nInfo Request\nWhat dose of DemoCure should be taken with food?\nNo adverse reaction and no defect reported.\n",
        )
        self.assertEqual(category, "Info Request (MI)")

    def test_irrelevant_content_remains_not_relevant(self):
        category, _, _ = classify_text(
            "General admin note",
            "team@example.com",
            "Please review the new marketing flyer and update the calendar.",
            "",
        )
        self.assertEqual(category, "Not Relevant")

    def test_explicit_negation_of_medical_question_and_reaction_is_not_relevant(self):
        category, _, _ = classify_text(
            "08_normal_irrelevant",
            "synthetic.sender@example.invalid",
            "Synthetic fixture message for assignment validation.",
            "A fictional office-supply company advertises a discount on printer paper, notebooks, and pens. No medicinal product, reaction, complaint, or medical question is discussed. Synthetic test data only. No real patient/client information.",
        )
        self.assertEqual(category, "Not Relevant")

    def test_analysis_contract_contains_traceability_fields(self):
        result = analyze_document_payload({
            "message_id": "MSG-TEST",
            "subject": "Adverse event",
            "sender": "doctor@example.com",
            "body": "A patient developed nausea after taking Aspirin.",
            "attachments": [],
        })
        self.assertEqual(result["message_id"], "MSG-TEST")
        self.assertIn("sourceTrace", result)
        self.assertIn("extractedFacts", result)
        self.assertTrue(result["extractedFacts"])
        self.assertTrue(all(fact["sourceReference"] for fact in result["extractedFacts"]))
        self.assertIn("ocrConfidence", result)
        self.assertFalse(result["ocrConfidence"]["required"])
        self.assertGreaterEqual(sentence_count(result["aiSummary"]), 10)
        self.assertLessEqual(sentence_count(result["aiSummary"]), 15)

    def test_readable_short_pdf_is_not_marked_as_ocr(self):
        self.assertEqual(classify_pdf_type("Short readable PDF text"), "Normal digital PDF")

    def test_article_markers_take_precedence_over_ocr_wording(self):
        self.assertEqual(
            classify_pdf_type(
                "Review Article\nThe scanned discussion summarizes published studies and references."
            ),
            "Published article",
        )

    def test_article_screening_is_present(self):
        result = screen_literature("Abstract and references describe a published case report.", ["Published article"])
        self.assertTrue(result["isLiteratureLike"])
        self.assertEqual(result["screeningStatus"], "Human literature review required")

    def test_api_key_is_loaded_from_environment(self):
        previous_value = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "test-key-from-env"
        try:
            self.assertEqual(get_gemini_api_key(), "test-key-from-env")
        finally:
            if previous_value is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = previous_value

    def test_scan_and_translation_and_table_helpers_work(self):
        self.assertEqual(classify_pdf_type("This is a blurred OCR scan of a handwritten form"), "Scanned / OCR required")
        translated = translate_non_english_text("¿El paciente tuvo nausea y la dosis fue 5 mg?")
        self.assertIn("patient", translated.lower())
        self.assertIn("dose", translated.lower())
        rows = extract_table_rows("Dose | Outcome\n5 mg | nausea\n10 mg | rash")
        self.assertEqual(rows[0][0], "Dose")
        self.assertEqual(rows[1][1], "nausea")

    def test_ascii_spanish_is_detected(self):
        from main import detect_language
        self.assertEqual(detect_language("Paciente tuvo reaccion despues de tomar el producto."), "Spanish")

    def test_handwritten_ocr_labels_are_reconciled(self):
        facts = extract_facts(
            "uploaded scan",
            "",
            "[Page 1] Age:68\nSex:Male\nDrug:Trixamet 50mg tablets\nStarted:approx.3days ago\nReaction:Dizziness,mild skin rash on both forearms\nOutcome:Not resolved,patient advised to see doctor\nReortedyAlvarezNHomeHealthN",
            "Safety Report (ICSR)",
        )
        values = {(fact["factGroup"], fact["fieldName"]): fact["fieldValue"] for fact in facts}
        self.assertEqual(values[("Patient", "Age")], "68")
        self.assertEqual(values[("Patient", "Sex")], "Male")
        self.assertIn("Trixamet", values[("Product", "Name")])
        self.assertIn("Dizziness", values[("Reaction", "What")])
        self.assertIn("Not Resolved", values[("Reaction", "Outcome")])

    def test_handwritten_initials_keep_pdf_provenance(self):
        facts = extract_facts(
            "uploaded scan",
            "",
            "[Page 1] Patient (initials): J.T.\nAge: 68\nDrug: Trixamet",
            "Safety Report (ICSR)",
        )
        initials = next(fact for fact in facts if fact["factGroup"] == "Patient" and fact["fieldName"] == "Initials")
        self.assertEqual(initials["fieldValue"], "J.T.")
        self.assertEqual(initials["confidence"], 0.8)
        self.assertIn("PDF attachment, page 1", initials["sourceReference"])

    def test_inline_handwritten_initials_are_extracted(self):
        facts = extract_facts(
            "uploaded scan",
            "",
            "[Page 1]\nPatient (initials): J.T.\nAge: 68 Sex: Male",
            "Safety Report (ICSR)",
        )
        initials = next(fact for fact in facts if fact["factGroup"] == "Patient" and fact["fieldName"] == "Initials")
        self.assertEqual(initials["fieldValue"], "J.T.")
        self.assertIn("PDF attachment, page 1", initials["sourceReference"])

    def test_narrative_safety_report_extracts_required_values(self):
        text = (
            "The patient is a 46-year-old female. She started taking Nexovira 20 mg tablets once daily on 18 August 2026 "
            "for stomach-related symptoms. On the third day of treatment, she developed a red, itchy rash on "
            "her arms and upper chest. She also complained of mild swelling around her lips. The patient stopped "
            "taking Nexovira on 21 August 2026. The rash started improving over the next two days, and the lip "
            "swelling has now resolved. She was not admitted to the hospital and no emergency treatment was required. "
            "The patient's medical history and other medications are not available. I am a fictional physician reporting this event for testing purposes. Pune, India."
        )
        facts = extract_facts("Possible reaction after starting Nexovira tablets", "", text, "Safety Report (ICSR)")
        values = {(fact["factGroup"], fact["fieldName"]): fact["fieldValue"] for fact in facts}
        self.assertEqual(values[("Patient", "Age")], "46")
        self.assertEqual(values[("Patient", "Sex")], "Female")
        self.assertEqual(values[("Product", "Name")], "Nexovira")
        self.assertEqual(values[("Product", "Dose")], "20 Mg")
        self.assertEqual(values[("Product", "Therapy Start")], "18 August 2026")
        self.assertEqual(values[("Product", "Therapy Stop")], "21 August 2026")
        self.assertIn("red, itchy rash", values[("Reaction", "What")].lower())
        self.assertEqual(values[("Reaction", "Onset")], "Third Day Of Treatment")
        self.assertIn("resolved", values[("Reaction", "Outcome")].lower())
        self.assertEqual(values[("Severity", "Hospitalization")], "No")
        self.assertIn("not available", values[("Patient", "Relevant History")].lower())
        self.assertEqual(values[("Reporter", "Role")], "Physician")
        self.assertEqual(values[("Reporter", "Country")], "India")

    def test_article_case_extracts_prose_patient_details(self):
        text = (
            "A 62-year-old woman with an eight-year history of hypertension and a five-year history "
            "of type 2 diabetes mellitus was started on amlopril (Cardiozin) 10 mg orally once daily. "
            "Seven days after starting therapy, she presented with acute swelling of the face and lips "
            "accompanied by mild dyspnea. She was observed for 48 hours, with complete resolution of "
            "symptoms. The suspect drug was discontinued."
        )
        facts = extract_facts("Case report", "", text, "Safety Report (ICSR)")
        values = {(fact["factGroup"], fact["fieldName"]): fact["fieldValue"] for fact in facts}
        self.assertEqual(values[("Patient", "Age")], "62")
        self.assertEqual(values[("Patient", "Sex")], "Female")
        self.assertIn("hypertension", values[("Patient", "Relevant History")].lower())
        self.assertIn("diabetes", values[("Patient", "Relevant History")].lower())
        self.assertIn("Amlopril", values[("Product", "Name")])
        self.assertEqual(values[("Product", "Dose")], "10 Mg")
        self.assertEqual(values[("Product", "Regimen")], "Once Daily")
        self.assertEqual(values[("Product", "Route")], "Oral")
        self.assertIn("swelling", values[("Reaction", "What")].lower())
        self.assertIn("seven days", values[("Reaction", "Onset")].lower())
        self.assertIn("resolved", values[("Reaction", "Outcome")].lower())

    def test_article_background_cannot_become_product_or_reaction(self):
        text = (
            "Introduction: Drug-induced angioedema is most classically associated with ACE inhibitors. "
            "Case Presentation: A 62-year-old woman with an eight-year history of hypertension and a five-year history "
            "of type 2 diabetes mellitus was started on amlopril (Cardiozin) 10 mg orally once daily for blood pressure control. "
            "Seven days after starting therapy, she presented with acute swelling of the face and lips accompanied by mild dyspnea. "
            "She was treated with intravenous antihistamines and corticosteroids and observed for 48 hours, with complete resolution of symptoms."
        )
        values = {(f["factGroup"], f["fieldName"]): f["fieldValue"]
                  for f in extract_facts("article", "", text, "Safety Report (ICSR)")}
        self.assertEqual(values[("Patient", "Age")], "62")
        self.assertEqual(values[("Patient", "Sex")], "Female")
        self.assertEqual(values[("Product", "Name")], "Amlopril (Cardiozin)")
        self.assertNotEqual(values[("Product", "Name")].lower(), "intravenous")
        self.assertIn("swelling", values[("Reaction", "What")].lower())
        self.assertNotIn("most classically", values[("Reaction", "What")].lower())

    def test_detailed_and_mixed_fixture_fields_are_preserved(self):
        """Regression coverage for the fields reviewers need to see first."""
        detailed = (
            "Patient: female, 62 years old, weight approx. 68 kg, history of hypertension and type 2 diabetes. "
            "The patient was started on Cardiozin (generic name amlopril) 10mg once daily, oral, on 15-Jul-2026 "
            "for blood pressure management. On 22-Jul-2026 she developed facial and lip angioedema after starting the medication."
        )
        mixed = (
            "Reporting on behalf of a patient (male, 45 years) who took Fevrolix tablets from Batch FVX-0091 and "
            "developed a severe skin rash with blistering within 24 hours. This is possibly linked to a manufacturing/quality defect."
        )
        detailed_values = {(f["factGroup"], f["fieldName"]): f["fieldValue"]
                           for f in extract_facts("case", "", detailed, "Safety Report (ICSR)")}
        mixed_values = {(f["factGroup"], f["fieldName"]): f["fieldValue"]
                        for f in extract_facts("case", "", mixed, "Safety Report (ICSR), Quality Complaint (PQC)")}
        self.assertEqual(detailed_values[("Patient", "Sex")], "Female")
        self.assertEqual(detailed_values[("Patient", "Age")], "62")
        self.assertEqual(detailed_values[("Product", "Name")], "Cardiozin (generic name amlopril)")
        self.assertEqual(detailed_values[("Product", "Disease / Indication")], "blood pressure management")
        self.assertIn("medication", detailed_values[("Reaction", "Suspected Cause")].lower())
        self.assertEqual(mixed_values[("Patient", "Sex")], "Male")
        self.assertEqual(mixed_values[("Patient", "Age")], "45")
        self.assertEqual(mixed_values[("Product", "Name")], "Fevrolix")
        self.assertIn("defect", mixed_values[("Reaction", "Suspected Cause")].lower())

    def test_article_screening_extracts_case_details(self):
        result = screen_literature(
            "Abstract: A 52-year-old patient developed rash after dosing. Case report and references follow.",
            ["Published article"],
        )
        self.assertTrue(result["isLiteratureLike"])
        self.assertIn("case", result["caseSummary"].lower())

    def test_article_screening_does_not_return_whole_article_as_case(self):
        result = screen_literature(
            "Abstract: This is a published article. References and conclusion follow.",
            ["Published article"],
        )
        self.assertEqual(result["cases"], [])

    def test_summary_has_a_longer_reviewable_format(self):
        summary = build_summary(
            "Adverse event report",
            "A patient reported nausea and rash after aspirin.",
            "Safety Report (ICSR)",
            "A scanned PDF with patient age 52 and dose 5 mg.",
        )
        sentence_count = len([s for s in summary.split('.') if s.strip()])
        self.assertGreaterEqual(sentence_count, 10)

    def test_literature_screen_batch_endpoint_works(self):
        results = screen_literature(
            "Abstract: A 58-year-old female developed rash and headache after 10 mg Imaginar. References and case report details follow.",
            ["Published article"],
        )
        self.assertTrue(results["isLiteratureLike"])
        self.assertIn("case", results["caseSummary"].lower())


if __name__ == "__main__":
    unittest.main()
