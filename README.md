# Medical Dictation → SOAP Note

Take a recording of a clinical encounter, transcribe it, and turn it into a structured SOAP note. Everything runs **locally**: no audio or text leaves the machine.

| Part | What | Tool |
|---|---|---|
| A | Speech to text, verbatim | **faster-whisper**, Whisper **large-v3** |
| B | Transcript → SOAP note (JSON + Markdown) | **Llama 3.1 8B** in **Ollama**, JSON-schema structured output |
| Bonus | Make sure nothing patient-reported sits in *Objective*; make sure no number is invented | rule check + transcript-grounded exam-step check + LLM classifier; number grounding |

The whole walkthrough, with outputs, is in **[`medical_dictation_to_soap.ipynb`](medical_dictation_to_soap.ipynb)**.

## Deliverables

| File | |
|---|---|
| [`medical_dictation_to_soap.ipynb`](medical_dictation_to_soap.ipynb) | Notebook that runs both parts end to end, with outputs |
| [`outputs/transcript.txt`](outputs/transcript.txt) | Raw transcript (Part A) |
| [`outputs/transcript_segments.json`](outputs/transcript_segments.json) | Same transcript with timestamps and per-segment confidence |
| [`outputs/soap_note.md`](outputs/soap_note.md) | Final SOAP note, readable (Part B) |
| [`outputs/soap_note.json`](outputs/soap_note.json) | Final SOAP note, structured (Part B) |
| [`outputs/validation_report.json`](outputs/validation_report.json) | What the logic checks found and changed (Bonus) |

## Setup

Tested on Windows 11, Python 3.12, a laptop with 8 GB RAM, a GTX 1650 (4 GB) and 16 CPU threads. It should work the same on Linux or macOS.

**1. Python packages**

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`av` is pinned to 14.2.0 on purpose. faster-whisper 1.2.1 calls `av.open(..., metadata_errors=...)`, an argument that PyAV 15+ removed, so newer PyAV crashes on the first file. No separate ffmpeg install is needed; PyAV ships its own.

**2. Ollama and the model**

Install Ollama from https://ollama.com/download, then:

```bash
ollama pull llama3.1:8b
```

Ollama runs as a background service on `http://localhost:11434`. If it isn't running, start it with `ollama serve`.

**3. Whisper model**

There's nothing to do here: faster-whisper downloads `large-v3` (about 3 GB) from Hugging Face on first use.

## How to run

**Notebook** (recommended, since it shows every step):

```bash
jupyter notebook medical_dictation_to_soap.ipynb
```

**Command line:**

```bash
python run_pipeline.py                          # data/sample_dictation.mp3 -> outputs/
python run_pipeline.py --audio path/to/file.mp3
python run_pipeline.py --skip-transcription     # reuse outputs/transcript.txt, only redo Part B
python run_pipeline.py --llm mistral            # any model you have in Ollama
```

**Runtime on my laptop:**
- Whisper large-v3 on CPU (int8, 16 threads): about 13 minutes for the 13.75-minute recording.
- The SOAP note plus checks: about 15–20 minutes, with the 8B model partly offloaded to a 4 GB GPU.

On a machine with a proper GPU, both take well under a minute.

GPU vs CPU is detected automatically. If CUDA, cuBLAS or cuDNN isn't usable, Whisper falls back to CPU int8. The code tests this with a 1-second decode, because a model can *load* on the GPU and still fail on the first real decode, which happened on my machine.

## Why these tools

**faster-whisper + large-v3 for Part A**
- large-v3 is the most accurate open Whisper checkpoint, and it handles clinical vocabulary well: "lateral epicondylitis", "pronate and supinate", "NSAIDs", "c-section".
- faster-whisper is the same model on CTranslate2. It's around 4x faster than the reference implementation and supports int8, which is what makes large-v3 practical on a CPU.
- It runs offline, which is what you want with patient audio.

**Llama 3.1 8B in Ollama for Part B**
- It's the best instruction-following model that still runs on this hardware.
- Ollama's **structured outputs** take a JSON schema and constrain generation to it, so the note is always valid JSON in the expected shape. There's no regex parsing of free text and no retry loop for broken JSON.
- Swapping in a bigger model is a single flag.

## What I tuned, and why

I compared every output to the transcript by hand. These changes all came from mistakes I actually saw.

**Transcription**
- **Whisper loops.** My first run used `temperature=0` and `condition_on_previous_text=True`, and large-v3 got stuck: *"so it's also a case of something called lateral epicondylitis"* came out **8 times in a row**, plus two smaller loops. Two changes fixed it:
  - Turning Whisper's temperature fallback back on. With a fixed `temperature=0`, a window that comes out repetitive (compression ratio > 2.4) is accepted instead of being decoded again.
  - Not conditioning each window on the previous one, so one bad window can't spread to the next.
- **A loop detector** (`repeated_phrases`) now flags any 3–15 word phrase repeated 3+ times in a row. It goes in the validation report, so this can't slip through silently again.
- **No `initial_prompt`.** A list of medical terms in the prompt can make Whisper "hear" words that were never said, which would break the verbatim requirement.

**SOAP note**
- **The recording is a conversation, not a monologue dictation.** There are no speaker labels, and the clinician guides a virtual exam ("press on the outside of your elbow", "extend your wrist"). The prompt explains how to tell the two speakers apart. It says the results of an exam step the clinician directs are **Objective**, even when the patient is the one saying "yes, that hurts".
- **A detailed schema as a checklist.** My first schema had one HPI list and one physical-exam list. The model summarised away about a third of the facts and turned *"should resolve in 6 months to 2 years"* into a *follow-up appointment*. Ollama makes the model write every key in order, so keys like `aggravating_factors`, `relieving_factors`, `inspection`, `special_tests_and_provocative_maneuvers`, `investigations` and `patient_education_and_expected_course` make it look for each item.
- **Chunking: tried and rejected.** Splitting the transcript into ~600-word chunks and merging recovered a few details, but each chunk still had to fill every field. The 8B model started making things up: it copied a pain score from the prompt's own example and invented "follow up in 2 weeks". An invented fact in a clinical note is worse than a missed one, so I went back to a single pass.
- **Prompt examples are not from this recording** (knee, hip...) and contain no numbers, so they can't leak into the note.
- **Repeatable output:** `temperature=0` and `seed=42`.

## The logic checks (bonus)

**Subjective information in Objective.** This is harder than it looks.
- The patient says *"I have full range of motion of both the elbow and hand"* while answering a history question. Out of context, "Full range of motion of elbow and hand" reads exactly like an exam finding. In my runs, the model kept putting it in Objective.
- What makes it Subjective is *who said it and when*, so the check looks at the transcript, not only at the note.

| Signal | How | Deterministic |
|---|---|---|
| Exam-step check | Finds the transcript passage the statement came from, then looks just before it for a clinician exam instruction ("look at your…", "palpate", "push on", "extend your…", "pronate", "I'm going to get you to…") | yes |
| Wording rule | Patient-report language: *reports, states, complains of, denies, feels, rates x/10, per patient…* | yes |
| LLM classifier | A second, narrow call that sees each statement **with its transcript excerpt** and decides whether it was a history answer or an exam result. Two contrasting few-shot examples (not from this case) | no |

**How the signals combine:**
- **Exam subsections** (inspection, palpation, range of motion, special tests): the exam-step check decides. A finding with no exam step before it came from the history, so it moves to Subjective. I made the deterministic check the decider because, in testing, the 8B classifier changed its answer for the same statement depending on what else was in the batch.
- **Vital signs and labs** (measurements, not instructions): the LLM and the wording rule decide.
- **Every move and every disagreement** goes into `validation_report.json` with the reason. When the signals disagree and the check *keeps* a statement, that's logged too, so a reviewer sees it.
- **An exam finding that can't be traced to any passage in the transcript** is not moved on a guess. It's flagged `needs_review` as a possible hallucination.

**Invented numbers.**
- Every number in the note has to appear in the transcript, either as digits or as words ("about a four" → 4).
- **Limitation:** it checks *that* a number was said, not *where*. In one run a made-up 6/10 passed because "six" appears in "six months". The next step would be checking each number together with its nearby words.

## Known limitations / what I'd do next

- **Speaker diarization** (e.g. pyannote) before the LLM step would make the S/O split much more reliable than inferring speakers from context.
- **An 8B model still leaves out some minor facts.** Examples: a few review-of-systems negatives, some negative range-of-motion tests, the amount of alcohol. Nothing in the final note is invented.
  - I also tried a second "what's missing from this draft?" pass. The 8B model just copied the draft back, so I dropped it.
  - The same code with a 70B model, or on a GPU server, is the realistic way to close that gap.
- **The exam-step check is conservative.** If a short history statement shares words with an exam passage, it is kept and the disagreement is logged rather than moved. I prefer a logged "please review" to silently corrupting a correct note.
- **Evaluation.** With the ground-truth transcript I'd measure WER (`jiwer`) after normalising case and punctuation, plus recall on a list of the medical terms. For the note, I'd use a field-level checklist reviewed by a clinician.
- **A clinician signs off.** This is a drafting tool. The confidence scores, loop detector and audit log are there so a human can review it quickly, not to replace that review.

## Project layout

```
medical_dictation_to_soap.ipynb   walkthrough notebook (run this)
run_pipeline.py                   same pipeline from the command line
soap_pipeline/
    transcribe.py                 Part A: faster-whisper, GPU/CPU fallback, loop detector
    soap.py                       Part B: schema, prompt, Ollama call
    checks.py                     Bonus: S/O check, number grounding
    render.py                     JSON -> Markdown
data/sample_dictation.mp3         the provided recording
outputs/                          transcript, SOAP note, validation report
```
