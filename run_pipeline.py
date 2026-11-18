"""
Run the whole thing from the command line:

    python run_pipeline.py                         # uses data/sample_dictation.mp3
    python run_pipeline.py --audio path/to/file.mp3
    python run_pipeline.py --skip-transcription    # reuse outputs/transcript.txt
"""

import argparse
from pathlib import Path

from soap_pipeline.checks import fix_subjective_in_objective, ungrounded_numbers
from soap_pipeline.render import save_note, to_markdown
from soap_pipeline.soap import MODEL, generate_soap
from soap_pipeline.transcribe import (load_model, low_confidence_segments,
                                      repeated_phrases, save_transcript, transcribe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="data/sample_dictation.mp3")
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--llm", default=MODEL)
    ap.add_argument("--skip-transcription", action="store_true")
    args = ap.parse_args()

    # Part A
    if args.skip_transcription:
        transcript = Path("outputs/transcript.txt").read_text(encoding="utf-8").strip()
    else:
        model = load_model(args.whisper_model)
        result = transcribe(args.audio, model)
        save_transcript(result)
        transcript = result["text"]
        unsure = low_confidence_segments(result)
        print(f"Transcribed {result['duration_sec']} s of audio, "
              f"{len(result['segments'])} segments, {len(unsure)} low-confidence")

    loops = repeated_phrases(transcript)
    if loops:
        print("WARNING - possible Whisper repetition loops, check the audio:", loops)

    # Part B
    note = generate_soap(transcript, model=args.llm)
    moves = fix_subjective_in_objective(note, transcript)
    report = {
        "llm": args.llm,
        "transcript_repetition_loops": loops,
        "subjective_in_objective_check": moves,
        "numbers_not_found_in_transcript": ungrounded_numbers(note, transcript),
    }
    save_note(note, report)

    print(to_markdown(note))
    print(f"Moved {sum('moved' in m for m in moves)} statement(s) from Objective to Subjective")
    print(f"Ungrounded numbers: {len(report['numbers_not_found_in_transcript'])}")


if __name__ == "__main__":
    main()
