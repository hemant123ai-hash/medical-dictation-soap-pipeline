"""
Part A - speech to text.

I went with faster-whisper (Whisper large-v3 converted to CTranslate2) because it
runs fully offline, so no patient audio ever leaves the machine, and on CPU it is
roughly 4x faster than the reference openai-whisper package for the same accuracy.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


def load_model(model_size: str = "large-v3") -> WhisperModel:
    """Load Whisper on the GPU if CUDA really works, otherwise fall back to CPU (int8)."""
    try:
        model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        # Loading can succeed even when cuBLAS/cuDNN are missing - the error only shows
        # up on the first decode. One second of silence is enough to find out early.
        list(model.transcribe(np.zeros(16000, dtype=np.float32), language="en")[0])
        print(f"Loaded {model_size} on GPU")
    except Exception as err:  # no CUDA / missing cuBLAS or cuDNN -> CPU is fine, just slower
        print(f"GPU not usable ({err.__class__.__name__}: {err}), using CPU")
        model = WhisperModel(model_size, device="cpu", compute_type="int8",
                             cpu_threads=os.cpu_count() or 4)
    return model


def transcribe(audio_path: str, model: WhisperModel) -> dict:
    """
    Transcribe the dictation verbatim.

    Settings chosen for accuracy rather than speed:
      - language fixed to English (skips auto-detect on the first 30 s)
      - beam_size=5, starting at temperature 0 (most likely wording). I keep Whisper's
        temperature fallback on purpose: when a window comes out repetitive
        (compression ratio > 2.4) or low-confidence, it is decoded again instead of
        being accepted. With a fixed temperature=0 that safety net is switched off.
      - condition_on_previous_text=False: large-v3 is known to get stuck in loops
        ("... lateral epicondylitis so it's also a case of ..." x8 in my first run)
        when every window is fed the previous window's text. Decoding each window on
        its own stops one bad window from poisoning the next.
      - VAD filter trims long silences so Whisper doesn't hallucinate text in pauses
      - no initial_prompt: a "helpful" vocabulary prompt can push the model to write
        words that were never said, which breaks the verbatim requirement
    """
    segments, info = model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
        vad_filter=True,
    )

    seg_list = []
    for seg in segments:  # generator - the actual decoding happens here
        seg_list.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            # low avg_logprob = the model was unsure; handy for a human to double check
            "avg_logprob": round(seg.avg_logprob, 3),
            "no_speech_prob": round(seg.no_speech_prob, 3),
        })

    full_text = " ".join(s["text"] for s in seg_list).strip()
    return {
        "audio_file": Path(audio_path).name,
        "duration_sec": round(info.duration, 2),
        "language": info.language,
        "text": full_text,
        "segments": seg_list,
    }


def low_confidence_segments(result: dict, threshold: float = -0.6) -> list:
    """Segments a reviewer should listen to again (avg_logprob below threshold)."""
    return [s for s in result["segments"] if s["avg_logprob"] < threshold]


def repeated_phrases(text: str, min_repeats: int = 3, max_len: int = 15) -> list:
    """
    Phrases (3-15 words) that repeat back-to-back 3+ times - the fingerprint of a
    Whisper loop. A real speaker repeats a word or two ("it's it's"), but the same
    6-word phrase three times in a row almost always means the decoder got stuck.
    """
    words = re.findall(r"[\w'-]+", text.lower())
    covered = set()  # word positions already inside a reported loop
    hits = []
    for n in range(3, max_len + 1):
        i = 0
        while i <= len(words) - n * min_repeats:
            gram = words[i:i + n]
            reps = 1
            while words[i + reps * n:i + (reps + 1) * n] == gram:
                reps += 1
            if reps >= min_repeats and i not in covered:
                hits.append({"phrase": " ".join(gram), "times": reps})
                covered.update(range(i, i + reps * n))
                i += reps * n
            else:
                i += 1
    return hits


def save_transcript(result: dict, out_dir: str = "outputs") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "transcript.txt").write_text(result["text"] + "\n", encoding="utf-8")
    (out / "transcript_segments.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    # python -m soap_pipeline.transcribe data/sample_dictation.mp3
    # Running Part A as its own process gives all of Whisper's memory back to the OS
    # when it exits - on an 8 GB machine that matters before the LLM loads.
    import sys
    import time

    t0 = time.time()
    res = transcribe(sys.argv[1] if len(sys.argv) > 1 else "data/sample_dictation.mp3",
                     load_model())
    save_transcript(res)
    print(f"Transcribed {res['duration_sec']} s of audio in {time.time() - t0:.0f} s, "
          f"{len(res['segments'])} segments")
