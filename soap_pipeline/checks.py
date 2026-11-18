"""
Post-generation checks (the bonus part + a hallucination guard).

1. Subjective-in-Objective check
   Two signals look at every Objective statement:
     a) a rule check for patient-report wording ("reports", "complains of", "rates",
        "states", "denies", "feels" ...) - cheap, transparent, catches the obvious
        cases even without an LLM
     b) a narrow LLM classifier that sees each statement TOGETHER WITH the part of
        the transcript it came from, and decides how the fact came up: as the
        patient's answer to a history question (SUBJECTIVE) or as the result of an
        exam step the clinician performed or directed (OBJECTIVE).

   Why (b) needs the transcript: in this recording the patient says "I have full
   range of motion of both the elbow and hand" while giving their history. Out of
   context, "Full range of motion of elbow and hand" reads like an exam finding, and
   my first context-free classifier let it stay in Objective. Who said it, and when,
   is what makes it Subjective.

   Decision: move if the classifier says SUBJECTIVE. If only the wording rule fires
   and the classifier, having seen the transcript, says it's an exam result, the
   statement stays - but that is written to the audit log too. Without the LLM the
   rule alone decides.

2. Number grounding check
   Every number in the final note must also appear in the transcript. Doses, vitals
   and lab values are where a wrong digit actually hurts someone, so any number the
   model "invented" is reported.
"""

import re

from .soap import call_ollama

# phrases that only make sense when the patient is the source of the information
SUBJECTIVE_CUES = [
    r"\breports?\b", r"\breported\b", r"\bstates?\b", r"\bstated\b", r"\bsays\b",
    r"\bcomplain(s|ed|ing)?\b", r"\bc/o\b", r"\bdenies\b", r"\bdenied\b",
    r"\bfeel(s|ing)?\b", r"\bfelt\b", r"\bdescribes?\b", r"\brates?\b.*\b\d+\s*(/|out of)\s*10\b",
    r"\bper (the )?patient\b", r"\bpatient('s)? (notes|mentions|admits|endorses)\b",
    r"\bendorses\b", r"\bnotices\b",
]
_CUE_RE = re.compile("|".join(SUBJECTIVE_CUES), re.IGNORECASE)

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "reason": {"type": "string"},
                    "label": {"type": "string", "enum": ["SUBJECTIVE", "OBJECTIVE"]},
                },
                "required": ["index", "reason", "label"],
            },
        }
    },
    "required": ["labels"],
}

CLASSIFY_PROMPT = """You check the Objective section of a SOAP note. For each numbered \
statement you get the statement and the excerpt of the encounter transcript it most \
likely came from (no speaker labels: the clinician asks and instructs, the patient \
answers).

Decide how the fact came up in the conversation. The key question: right before the \
answer, does the clinician give an EXAMINATION instruction - inspection ("look at your \
... do you see / notice ..."), palpation (press, push on, palpate), movement or \
provocation tests (extend, flex, twist, lift, squeeze, "I'm going to get you to ...")?
- Yes -> OBJECTIVE. The answer is an exam finding, and a "no pain" / "no" answer is a \
NEGATIVE exam finding - still OBJECTIVE. Vital signs, lab and imaging results are \
OBJECTIVE too.
- No, the clinician asked a history question about symptoms, abilities, daily life or \
past history ("are you able to ...", "do you have any ...", "how bad is it") -> \
SUBJECTIVE, even if the statement sounds like an exam finding.

Examples (not from this encounter):
STATEMENT: No pain with resisted hip flexion
EXCERPT: ... okay now lift your leg up against my hand does that hurt no that's fine ...
-> reason: clinician instructs "lift your leg against my hand", answer is the result; \
label: OBJECTIVE
STATEMENT: Able to walk two blocks
EXCERPT: ... and how far can you walk before the pain starts maybe two blocks ...
-> reason: history question about daily function, no exam step; label: SUBJECTIVE

Give a short reason that quotes the instruction or question from the excerpt, then the \
label. Return one entry per statement, using the same index."""

_STOP = set("""a an the and or of to in on at for with without is are was were be been
it its this that there no not any as by from but if so than then up down your you
patient pt""".split())


def _objective_items(note: dict):
    """Yield (subsection, index, text) for every statement in the Objective section."""
    for sub, items in note["objective"].items():
        for i, text in enumerate(items):
            if text.strip().lower() != "not documented":
                yield sub, i, text


def _words(text: str) -> list:
    return re.findall(r"[a-z0-9']+", text.lower())


def find_excerpt(statement: str, transcript: str, window: int = 25,
                 before: int = 20, after: int = 15) -> str:
    """
    The transcript passage a statement most likely came from: slide a window over
    the transcript and pick the spot sharing the most content words with the
    statement, then add some context on both sides so the question is visible too.
    """
    words = transcript.split()
    i = _best_match(statement, words, window)
    if i is None:
        return ""
    return " ".join(words[max(0, i - before):i + window + after])


def _stem(word: str) -> str:
    # crude but enough here: "flexion"/"flex", "extension"/"extend", "palpation"/"palpate"
    return word[:4]


def _matches(statement: str, words: list, window: int = 25) -> list:
    """
    Start indexes of transcript windows that share the most content words with the
    statement (best score, or one below it), best first. Empty if nothing matches.
    """
    plain = [_stem(w.strip(".,?!;:\"").lower()) for w in words]
    keys = {_stem(w) for w in _words(statement) if w not in _STOP and len(w) > 2}
    if not keys or not words:
        return []
    scored = [(len(keys & set(plain[i:i + window])), i)
              for i in range(0, max(1, len(words) - window + 1), 3)]
    best = max(score for score, _ in scored)
    if best < max(1, (len(keys) + 1) // 2):  # nothing in the transcript really matches
        return []
    return [i for score, i in sorted(scored, key=lambda x: (-x[0], x[1])) if score >= best - 1]


def _best_match(statement: str, words: list, window: int = 25):
    hits = _matches(statement, words, window)
    return hits[0] if hits else None


# How a clinician starts an examination step, in a live encounter or in a dictation.
EXAM_INSTRUCTION = re.compile(
    r"\b(palpat\w*|push(ing)? on|press(ing)? on|get you to|going to (examine|check)|"
    r"look at (that|your|the)|extend(ing)? your|flex(ing)? your|bend(ing)? your|"
    r"straighten(ing)? your|twist(ing)? your|lift(ing)? your|squeez\w*|pronat\w*|"
    r"supinat\w*|adduct\w*|abduct\w*|on (exam|examination)|exam(ination)? (shows|reveals))\b",
    re.IGNORECASE,
)

# Objective subsections that come from measurements, not from an exam instruction
MEASURED = {"vital_signs", "labs_and_imaging_results"}


def exam_instruction_before(statement: str, transcript: str, look_back: int = 25,
                            window: int = 25):
    """
    Where did this statement come from? Returns (found, cue):
      found = False  -> no transcript passage matches the statement (can't tell)
      cue            -> the exam instruction just before that passage ("palpate",
                        "extend your" ...), or None if the passage has none
    Deterministic, so the check gives the same answer every run.
    """
    words = transcript.split()
    hits = _matches(statement, words, window)
    if not hits:
        return False, None
    # Short statements can match more than one place ("wrist" + "extension" appears in
    # the exam AND in the clinician's explanation of the diagnosis). Only call it
    # history if NONE of the close matches has an exam instruction before it.
    for i in hits:
        region = " ".join(words[max(0, i - look_back):i + window])
        m = EXAM_INSTRUCTION.search(region)
        if m:
            return True, m.group(0)
    return True, None


def rule_flags(note: dict) -> set:
    return {(sub, i) for sub, i, text in _objective_items(note) if _CUE_RE.search(text)}


def llm_labels(note: dict, transcript: str = "") -> dict:
    """{(subsection, index): (label, reason)} for every Objective statement."""
    items = list(_objective_items(note))
    if not items:
        return {}
    blocks = []
    for n, (_, _, text) in enumerate(items):
        excerpt = find_excerpt(text, transcript) if transcript else ""
        blocks.append(f"{n}. STATEMENT: {text}\n   TRANSCRIPT EXCERPT: "
                      f"{excerpt or '(no matching passage found)'}")
    result = call_ollama(
        [{"role": "system", "content": CLASSIFY_PROMPT},
         {"role": "user", "content": "\n\n".join(blocks)}],
        CLASSIFY_SCHEMA,
    )
    labels = {}
    for row in result.get("labels", []):
        n = row.get("index")
        if isinstance(n, int) and 0 <= n < len(items):
            sub, i, _ = items[n]
            labels[(sub, i)] = (row.get("label"), row.get("reason", ""))
    return labels


def llm_flags(note: dict, transcript: str = "") -> set:
    return {k for k, (label, _) in llm_labels(note, transcript).items() if label == "SUBJECTIVE"}


def fix_subjective_in_objective(note: dict, transcript: str = "", use_llm: bool = True) -> list:
    """Move patient-reported statements out of Objective. Returns an audit log."""
    by_rule = rule_flags(note)
    labels = llm_labels(note, transcript) if use_llm else {}

    to_move, log = {}, []
    for sub, i, text in list(_objective_items(note)):
        label, why = labels.get((sub, i), (None, ""))
        reasons = []
        if (sub, i) in by_rule:
            reasons.append("patient-report wording")
        if label == "SUBJECTIVE":
            reasons.append(f"LLM classifier: {why}")

        found, cue = (exam_instruction_before(text, transcript) if transcript
                      else (False, None))
        if found and sub not in MEASURED:
            # exam subsections: the transcript decides - an exam finding needs an exam step
            if cue is None:
                to_move[(sub, i)] = ["no examination step before it in the transcript, "
                                     "so it came from the history"] + reasons
            elif reasons:
                log.append({"kept": text, "in": f"objective.{sub}",
                            "reason": f"flagged ({' + '.join(reasons)}) but the transcript "
                                      f"shows an exam step right before it (\"{cue}\")"})
        elif label == "SUBJECTIVE" or ((sub, i) in by_rule and label != "OBJECTIVE"):
            to_move[(sub, i)] = reasons
        elif transcript and not found and sub not in MEASURED:
            # an exam finding that can't be traced to ANY passage of the transcript is
            # not a verified finding - it may be invented. Don't guess where it belongs,
            # make sure a human looks at it.
            log.append({"needs_review": text, "in": f"objective.{sub}",
                        "reason": "could not be traced to any passage in the transcript "
                                  "(possible hallucination) - left in place for review"})
        elif reasons:
            log.append({"kept": text, "in": f"objective.{sub}",
                        "reason": f"wording looked patient-reported, but the LLM "
                                  f"classifier says it is a measurement: {why}"})

    # delete from the end of each list so earlier indexes stay valid
    for (sub, i), reasons in sorted(to_move.items(), key=lambda x: (x[0][0], -x[0][1])):
        text = note["objective"][sub].pop(i)
        note["subjective"]["history_of_present_illness"]["other_details"].append(text)
        log.append({
            "moved": text,
            "from": f"objective.{sub}",
            "to": "subjective.history_of_present_illness.other_details",
            "reason": " + ".join(reasons),
        })
    return log


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# speech is transcribed as words ("about a four", "six months") but the note uses
# digits ("4/10"), so number words in the transcript are counted as numbers too
_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "once": 1, "twice": 2, "single": 1, "couple": 2,
}


def _all_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _all_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_strings(v)


def _numbers_in_transcript(transcript: str) -> set:
    nums = set(_NUM_RE.findall(transcript))
    for word in re.findall(r"[a-z]+", transcript.lower()):
        if word in _WORD_NUMS:
            nums.add(str(_WORD_NUMS[word]))
    return nums


def ungrounded_numbers(note: dict, transcript: str) -> list:
    """
    Numbers that appear in the note but nowhere in the transcript (as digits or words).
    "/10" pain-scale denominators are ignored because the clinician defines the scale
    in words ("one to ten, ten being the worst").
    """
    source = _numbers_in_transcript(transcript)
    missing = []
    for text in _all_strings(note):
        for num in _NUM_RE.findall(re.sub(r"/\s*10\b", "", text)):
            if num not in source:
                missing.append({"number": num, "in": text})
    return missing
