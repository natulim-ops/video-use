"""Plan an EDL from raw footage + creative brief using Gemini.

Given a folder of videos that have already been transcribed with
ElevenLabs Scribe (`transcribe_batch.py`) and packed
(`pack_transcripts.py`), this script asks Gemini to pick which clips,
in what order, and with what trim points to tell the story described
by the creative brief.

The output is an EDL in the format `render.py` expects. Each range's
`end` is post-processed with a volume scan to tighten cuts past Scribe's
last-word-end timestamp — Scribe frequently attributes trailing silence
(up to 1+ second) to the final word, which would show up as dead air.

Usage:

    python helpers/plan.py \\
        --videos-dir ./footage \\
        --brief '{"product":"Natulim FR","audience":"French women 25-45","tone":"authentic testimonial","duration_seconds":25}' \\
        --style styles/dtc-testimonial.yaml \\
        --output edit/edl.json

The brief can also be a path to a .json file. Pass `--no-tighten` to
skip the volume-scan post-processing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, Field


VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".m4v", ".avi"}

MODEL_ID = "gemini-2.5-flash-lite"

# Volume threshold (dB) — values below this are considered silence.
SILENCE_DB = -55.0
SCAN_WINDOW = 0.1  # seconds per volume probe window
SCAN_BACK = 1.5     # seconds to scan back from Scribe's claimed end
TIGHTEN_PAD = 0.03  # pad added after last detected speech frame


# --------------------------------------------------------------------------- #
# Response schema
# --------------------------------------------------------------------------- #


class EdlRange(BaseModel):
    source: str = Field(
        ...,
        description="Clip name (filename stem, e.g. 'IMG_5713'). MUST match a clip in the catalog.",
    )
    phrase_start: float = Field(
        ...,
        description=(
            "Start time in source file (seconds). Use the earliest phrase's "
            "start from the catalog — we will subtract a small pre-roll."
        ),
    )
    phrase_end: float = Field(
        ...,
        description=(
            "End time in source file (seconds). Use the latest phrase's end. "
            "We will volume-scan backwards to tighten past any trailing silence."
        ),
    )
    beat: str = Field(
        ...,
        description="One of: hook, problem, solution, social_proof, cta, other",
    )
    note: str = Field(
        ...,
        description="Short transcript excerpt (≤120 chars) for human sanity-check",
    )


class EdlPlan(BaseModel):
    ranges: list[EdlRange]


# --------------------------------------------------------------------------- #
# Load inputs
# --------------------------------------------------------------------------- #


def load_brief(brief_arg: str) -> dict:
    """Parse `brief_arg` as inline JSON or a path to a JSON file."""
    if brief_arg.lstrip().startswith("{"):
        return json.loads(brief_arg)
    candidate = Path(brief_arg)
    if candidate.exists() and candidate.is_file():
        return json.loads(candidate.read_text())
    sys.exit(f"brief is neither valid JSON nor an existing file: {brief_arg!r}")


def load_style(style_path: Path | None) -> dict | None:
    if style_path is None:
        return None
    if not style_path.exists():
        print(f"warning: style file not found: {style_path}")
        return None
    return yaml.safe_load(style_path.read_text())


def find_videos(videos_dir: Path) -> list[Path]:
    return sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix in VIDEO_EXTS
    )


def load_packed(edit_dir: Path) -> str:
    p = edit_dir / "takes_packed.md"
    if not p.exists():
        sys.exit(
            f"takes_packed.md not found at {p}. Run pack_transcripts.py first."
        )
    return p.read_text()


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #


def format_style_guidance(style: dict | None, duration_seconds: int) -> str:
    """Render the style YAML into prompt-friendly text."""
    if not style:
        return (
            "Default structure (modify as needed): "
            "hook → problem → solution → social_proof → cta"
        )
    lines = [f"Style: {style.get('name', 'custom')}"]
    if desc := style.get("description"):
        lines.append(desc.strip())
    lines.append("")
    lines.append("Segment guidance (durations are illustrative — scale to hit target):")
    for seg in style.get("structure", []) or []:
        name = seg.get("segment", "?")
        dur = seg.get("duration_seconds", "?")
        guide = (seg.get("guidance") or "").strip()
        lines.append(f"  - {name} (~{dur}s): {guide}")
    if pacing := style.get("pacing", {}).get("notes"):
        lines.append("")
        lines.append(f"Pacing: {pacing.strip()}")
    return "\n".join(lines)


def build_prompt(
    brief: dict,
    style: dict | None,
    packed: str,
    catalog_clips: list[str],
) -> str:
    duration = brief.get("duration_seconds", 30)
    lower = int(duration * 0.9)
    upper = int(duration * 1.1)

    # Render any extra brief fields (ending, language, cta, etc.) as bullets.
    core_fields = {"product", "audience", "tone", "duration_seconds", "style_ref"}
    extras = {k: v for k, v in brief.items() if k not in core_fields and v}
    extras_block = (
        "\n".join(f"- {k.capitalize()}: {v}" for k, v in extras.items())
        if extras else ""
    )

    return f"""You are a professional video editor assembling a DTC short-form ad from raw UGC footage.

## Creative Brief
- Product: {brief.get("product", "")}
- Audience: {brief.get("audience", "")}
- Tone: {brief.get("tone", "")}
- Target duration: {duration}s (acceptable range {lower}-{upper}s)
{extras_block}

## Style Guidance
{format_style_guidance(style, duration)}

## Footage Catalog
Each clip is listed with phrase-level transcripts from ElevenLabs Scribe.
Phrase ranges `[start-end]` are times IN THE SOURCE FILE.

Available clips: {", ".join(catalog_clips)}

{packed}

## Your Task
Produce 3-8 ranges that tell the story described by the brief. For each range:
- `source` must match one of the catalog clip names exactly (stem without extension)
- `phrase_start` / `phrase_end` should span the earliest-to-latest phrase
  you want to include from that clip. Use the EXACT numeric values from
  the catalog — do not round.
- `beat` is one of: hook, problem, solution, social_proof, cta, other.
- `note` is a short transcript excerpt for human sanity-check.

## Rules
1. Only use clips that exist in the catalog.
2. Each source clip appears AT MOST ONCE across all ranges. No duplicates.
3. Pick cuts at PHRASE boundaries — whole sentences when possible.
4. Order ranges to tell a coherent story — narrative arc beats the literal
   order of the source files.
5. Sum of phrase-durations should hit the target ±10% ({lower}-{upper}s).
   We will tighten each end automatically, so slight overshoot is fine.
6. Prefer clips with clearer speech and higher visual quality for the hook.
7. If the brief mentions a specific ending (e.g. "end with the product shot"),
   the LAST range MUST honor that even if it disrupts strict narrative order.
"""


# --------------------------------------------------------------------------- #
# Gemini call
# --------------------------------------------------------------------------- #


def _load_api_key() -> str:
    """Load GOOGLE_API_KEY from env or the project .env."""
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GOOGLE_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    sys.exit("GOOGLE_API_KEY not set. Add it to video-use/.env or export it.")


def generate_edl_from_gemini(prompt: str) -> EdlPlan:
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[prompt],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EdlPlan,
        ),
    )
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, EdlPlan):
        return parsed
    # Fallback: parse the raw text
    text = getattr(response, "text", None)
    if not text:
        sys.exit("Gemini returned no response text.")
    text = text.strip()
    # Strip markdown fences if present
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return EdlPlan.model_validate_json(text)


# --------------------------------------------------------------------------- #
# True-audio-end tightening (volume scan)
# --------------------------------------------------------------------------- #


def probe_volume(source: Path, start: float, duration: float) -> float | None:
    """Return mean_volume in dB for [start, start+duration], or None on failure."""
    try:
        out = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-nostats",
                "-ss", f"{start:.3f}",
                "-t", f"{duration:.3f}",
                "-i", str(source),
                "-af", "volumedetect",
                "-vn", "-f", "null", "-",
            ],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return None
    m = re.search(r"mean_volume:\s*(-?[0-9.]+)\s*dB", out.stderr)
    if not m:
        return None
    return float(m.group(1))


def tighten_end(source: Path, phrase_start: float, phrase_end: float) -> float:
    """Find the true audio end by scanning the whole phrase range.

    Returns last window whose mean volume is louder than SILENCE_DB,
    plus TIGHTEN_PAD. Scanning the whole range (not just the tail)
    handles cases where Scribe attributes a 2+ second silence to
    the final word or audio-event.
    """
    last_speech_end: float | None = None
    t = round(max(0.0, phrase_start), 3)
    while t < phrase_end:
        vol = probe_volume(source, t, SCAN_WINDOW)
        if vol is not None and vol > SILENCE_DB:
            last_speech_end = t + SCAN_WINDOW
        t = round(t + SCAN_WINDOW, 3)
    if last_speech_end is None:
        return phrase_end  # Nothing detected at all — keep original
    return min(phrase_end, round(last_speech_end + TIGHTEN_PAD, 3))


# --------------------------------------------------------------------------- #
# EDL writer
# --------------------------------------------------------------------------- #


def build_edl(
    plan: EdlPlan,
    videos_dir: Path,
    edit_dir: Path,
    clip_by_stem: dict[str, Path],
    start_pad: float = 0.05,
    tighten: bool = True,
) -> dict:
    sources: dict[str, str] = {}
    ranges: list[dict] = []
    for r in plan.ranges:
        if r.source not in clip_by_stem:
            sys.exit(
                f"Gemini picked source {r.source!r} but no matching clip found "
                f"in {videos_dir}. Available: {sorted(clip_by_stem.keys())}"
            )
        clip_path = clip_by_stem[r.source]
        # Record source path relative to edit_dir so the EDL is portable.
        try:
            rel = clip_path.resolve().relative_to(edit_dir.resolve())
            rel_path = str(rel)
        except ValueError:
            # Use .. escape — edit_dir is inside videos_dir typically
            rel_path = os.path.relpath(clip_path.resolve(), edit_dir.resolve())
        sources[r.source] = rel_path

        start = max(0.0, round(r.phrase_start - start_pad, 3))
        end = float(r.phrase_end)
        if tighten:
            tight = tighten_end(clip_path, r.phrase_start, r.phrase_end)
            if tight < end:
                print(
                    f"  tighten {r.source}: end {end:.3f} → {tight:.3f} "
                    f"(saved {end - tight:.3f}s)"
                )
                end = tight
        ranges.append({
            "source": r.source,
            "start": start,
            "end": end,
            "beat": r.beat,
            "note": r.note,
        })

    return {
        "sources": sources,
        "grade": "",
        "ranges": ranges,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate an EDL from a creative brief using Gemini."
    )
    ap.add_argument("--videos-dir", type=Path, required=True,
                    help="Directory containing source videos.")
    ap.add_argument("--brief", type=str, required=True,
                    help="Creative brief as inline JSON OR path to .json file.")
    ap.add_argument("--style", type=Path, default=None,
                    help="Optional style YAML (e.g. styles/dtc-testimonial.yaml).")
    ap.add_argument("--output", type=Path, default=None,
                    help="Path to write EDL (default: <videos-dir>/edit/edl.json).")
    ap.add_argument("--edit-dir", type=Path, default=None,
                    help="Override edit dir (default: <videos-dir>/edit).")
    ap.add_argument("--no-tighten", action="store_true",
                    help="Skip volume-scan end tightening.")
    args = ap.parse_args()

    videos_dir = args.videos_dir.resolve()
    if not videos_dir.is_dir():
        sys.exit(f"videos_dir not a directory: {videos_dir}")

    edit_dir = (args.edit_dir or (videos_dir / "edit")).resolve()
    if not edit_dir.is_dir():
        sys.exit(f"edit_dir does not exist: {edit_dir}. Run transcribe_batch + pack_transcripts first.")

    output = args.output.resolve() if args.output else edit_dir / "edl.json"

    brief = load_brief(args.brief)
    style = load_style(args.style)
    packed = load_packed(edit_dir)
    videos = find_videos(videos_dir)
    clip_by_stem = {v.stem: v for v in videos}
    catalog_clips = sorted(clip_by_stem.keys())

    if not catalog_clips:
        sys.exit(f"no video files found in {videos_dir}")

    prompt = build_prompt(brief, style, packed, catalog_clips)
    print(f"[plan] asking {MODEL_ID} for an EDL ({len(catalog_clips)} clips available)")
    plan = generate_edl_from_gemini(prompt)

    # Dedupe: if the model repeats a clip, keep the first occurrence only.
    seen: set[str] = set()
    deduped: list[EdlRange] = []
    for r in plan.ranges:
        if r.source in seen:
            print(f"  [dedup] dropping repeated {r.source} ({r.beat})")
            continue
        seen.add(r.source)
        deduped.append(r)
    plan.ranges = deduped

    print(f"[plan] {len(plan.ranges)} ranges after dedup:")
    for r in plan.ranges:
        print(f"  [{r.beat:<14}] {r.source}  {r.phrase_start:.2f}-{r.phrase_end:.2f}  {r.note[:70]}")

    if not args.no_tighten:
        print("[plan] tightening ends via volume scan…")
    edl = build_edl(
        plan, videos_dir, edit_dir, clip_by_stem, tighten=not args.no_tighten,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(edl, indent=2, ensure_ascii=False) + "\n")
    total_dur = sum(r["end"] - r["start"] for r in edl["ranges"])
    print(f"[plan] wrote {output} — {len(edl['ranges'])} ranges, total {total_dur:.2f}s")


if __name__ == "__main__":
    main()
