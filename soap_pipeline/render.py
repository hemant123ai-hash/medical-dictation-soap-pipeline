"""Turn the SOAP JSON into a readable Markdown note."""

import json
from pathlib import Path

SECTIONS = [
    ("subjective", "S - Subjective"),
    ("objective", "O - Objective"),
    ("assessment", "A - Assessment"),
    ("plan", "P - Plan"),
]


def _title(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _render(value, level: int) -> str:
    """Walk the note in schema order: dicts become headings, lists become bullets."""
    if isinstance(value, str):
        return f"{value}\n\n"
    if isinstance(value, list):
        return ("".join(f"- {x}\n" for x in value) if value else "- Not documented\n") + "\n"
    md = ""
    for key, sub in value.items():
        if isinstance(sub, str):
            md += f"**{_title(key)}:** {sub}\n\n"
        elif isinstance(sub, list):
            md += f"**{_title(key)}**\n{_render(sub, level + 1)}"
        else:
            md += f"{'#' * level} {_title(key)}\n\n{_render(sub, level + 1)}"
    return md


def to_markdown(note: dict) -> str:
    p = note["patient"]
    md = "# SOAP Note\n\n"
    md += (f"**Patient:** {p['name']}  \n**Age:** {p['age']}  \n**Sex:** {p['sex']}  \n"
           f"**Date of service:** {p['date_of_service']}\n\n")
    for key, heading in SECTIONS:
        md += f"## {heading}\n\n{_render(note[key], level=3)}"
    return md.rstrip() + "\n"


def save_note(note: dict, report: dict, out_dir: str = "outputs") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "soap_note.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    (out / "soap_note.md").write_text(to_markdown(note), encoding="utf-8")
    (out / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
