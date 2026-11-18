"""
Part B - transcript -> structured SOAP note.

The LLM runs locally through Ollama (Llama 3.1 8B), so the transcript stays on the
machine too. Ollama's structured-output mode takes a JSON schema and constrains the
model's output to it, so the result is always valid JSON in our shape - no regex
parsing of free text.

Why such a detailed schema: my first version had one "history_of_present_illness"
list and one "physical_exam" list. The 8B model summarised aggressively and dropped
about a third of the facts (no-trauma, what relieves the pain, family history,
negative exam findings...). Ollama makes the model write every key in order, so
specific keys like "aggravating_factors" or "inspection" act as a checklist the
model has to walk through. Recall went up a lot without any extra LLM calls.
"""

import json

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"


def _list():
    return {"type": "array", "items": {"type": "string"}}


def _obj(*text_fields, lists=()):
    props = {f: {"type": "string"} for f in text_fields}
    props.update({f: _list() for f in lists})
    return {"type": "object", "properties": props, "required": list(props)}


# Lists of short statements keep each fact separate, which is what makes the S/O
# check in checks.py possible (it looks at one statement at a time).
SOAP_SCHEMA = {
    "type": "object",
    "properties": {
        "patient": _obj("name", "age", "sex", "date_of_service"),
        "subjective": {
            "type": "object",
            "properties": {
                "chief_complaint": {"type": "string"},
                "history_of_present_illness": _obj(lists=(
                    "onset_and_duration",
                    "location_and_radiation",
                    "quality",
                    "severity",
                    "course_over_time",
                    "aggravating_factors",
                    "relieving_factors",
                    "treatments_tried",
                    "associated_symptoms_and_pertinent_negatives",
                    "functional_impact",
                    "other_details",
                )),
                "past_medical_history": _list(),
                "past_surgical_history": _list(),
                "medications": _list(),
                "allergies": _list(),
                "immunizations": _list(),
                "family_history": _list(),
                "social_history": _list(),
                "review_of_systems": _list(),
            },
            "required": ["chief_complaint", "history_of_present_illness",
                         "past_medical_history", "past_surgical_history", "medications",
                         "allergies", "immunizations", "family_history",
                         "social_history", "review_of_systems"],
        },
        "objective": _obj(lists=(
            "vital_signs",
            "inspection",
            "palpation",
            "range_of_motion_and_strength",
            "special_tests_and_provocative_maneuvers",
            "labs_and_imaging_results",
        )),
        "assessment": _obj(lists=("diagnoses", "clinical_reasoning")),
        "plan": _obj(lists=(
            "investigations",
            "medications",
            "therapies_and_referrals",
            "activity_modification_and_self_care",
            "devices_and_supports",
            "patient_education_and_expected_course",
            "follow_up",
        )),
    },
    "required": ["patient", "subjective", "objective", "assessment", "plan"],
}

# The examples in the prompt are deliberately NOT taken from the sample recording,
# so the prompt isn't tuned to one case and can't "leak" an answer into the note.
SYSTEM_PROMPT = """You are a clinical documentation specialist. You convert the \
transcript of a clinical encounter (a clinician talking with a patient) into a SOAP note. \
The transcript has no speaker labels, so work out who is speaking from context: the \
clinician asks questions and gives instructions, the patient answers. You never add, \
guess or infer anything that was not said.

Section rules:
- SUBJECTIVE = what the PATIENT tells the clinician: the complaint and its history, what \
they have tried, past medical and surgical history, medications, allergies, \
immunizations, family history, social history (work, smoking, alcohol, drugs, travel, \
sick contacts) and the review of systems (symptoms asked about, present or denied).
  Past medical history = ongoing / chronic conditions, or that the patient says they \
are otherwise healthy. Surgeries and hospitalizations go in past surgical history.
- OBJECTIVE = findings from the clinician's examination: vital signs, inspection, \
palpation, range of motion, special / provocative tests, lab and imaging results. If the \
clinician directs the exam (for example asks the patient to press on an area or move a \
joint) the result is still an exam finding and belongs here - write it as a finding \
("Tenderness to palpation over the medial knee", "No pain with resisted hip flexion"), \
not as "patient reports". Anything the patient says while giving the history never goes \
here, even if it sounds like an exam finding.
- ASSESSMENT = the diagnosis exactly as the clinician named it, plus the reasoning the \
clinician gave for it.
- PLAN = everything the clinician said about management: investigations ordered or \
deliberately not ordered (with the reason given), medications, therapies, activity \
changes, devices, patient education, expected course and follow-up. Keep optional \
items optional ("may use ... if desired"). An expected recovery time is patient \
education, not a follow-up appointment.

Accuracy rules:
- Use only facts stated in the transcript. Keep medical terms, drug names and the \
clinician's wording for movements and tests exactly as spoken (do not change \
"extension" to "flexion" or vice versa).
- One fact per statement. Keep statements short but do not drop qualifiers (side, \
location, amount, how often, what it was triggered by).
- One statement per exam maneuver, including the position it was done in (e.g. \
"... with the knee fully bent"). Do not merge two maneuvers into one statement.
- In the plan, include every recommendation the clinician made, including self-care \
measures and optional items.
- Record negatives too: a symptom or finding the patient denies, or a test that was \
negative, is clinically important.
- Write numbers as digits, and only numbers that were actually said.
- If something is not mentioned (e.g. name, age, date, vital signs, labs) use \
"Not documented" for text fields and [] for lists. Never fill a field just because it \
exists.
- Do not include a diagnosis that the clinician did not state.
- Output only the JSON object."""


def build_user_prompt(transcript: str) -> str:
    # Ollama's docs recommend also showing the schema in the prompt; it helps the
    # model understand what each key is for, not only that it has to exist.
    return (
        "Create the SOAP note from this encounter transcript. Go through the transcript "
        "from start to finish and place every clinically relevant fact in the right "
        "field.\n\n"
        f"JSON schema to follow:\n{json.dumps(SOAP_SCHEMA)}\n\n"
        f"TRANSCRIPT:\n\"\"\"\n{transcript}\n\"\"\""
    )


def call_ollama(messages: list, schema: dict, model: str = MODEL) -> dict:
    """One chat call with a JSON schema. temperature 0 + fixed seed = repeatable output."""
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "format": schema,
            "stream": False,
            "options": {"temperature": 0, "seed": 42, "num_ctx": 8192},
        },
        timeout=1800,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["message"]["content"])


_EMPTY = {"not documented", "none mentioned", "not mentioned", "none", "no", "n/a", ""}


def _drop_placeholders(obj):
    """
    The model sometimes puts "Not documented" or a bare "no" INSIDE a list instead of
    leaving it empty. Strip those so an empty list really means "nothing said".
    ("None" is kept for allergies, where "no allergies" is a real answer.)
    """
    if isinstance(obj, dict):
        return {k: (v if k == "allergies" else _drop_placeholders(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [x for x in obj if x.strip().strip(".").lower() not in _EMPTY]
    return obj


def generate_soap(transcript: str, model: str = MODEL) -> dict:
    """
    One pass over the whole transcript.

    I also tried splitting the transcript into ~600-word chunks and merging the
    per-chunk notes. It recovered a few details but was clearly worse overall: a
    chunk that only covers the review of systems still has to fill every field, and
    the 8B model started filling them with plausible guesses - it copied "6/10" from
    the prompt example and invented "follow up in 2 weeks". In a clinical note an
    invented fact is far worse than a missed one, so single pass it is.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(transcript)},
    ]
    return _drop_placeholders(call_ollama(messages, SOAP_SCHEMA, model=model))
