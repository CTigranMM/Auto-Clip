import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass

from faster_whisper import WhisperModel
from rich.console import Console

console = Console()

FILLER_WORDS = {"um", "uh", "uhm", "erm", "hmm"}
DEAD_AIR_THRESHOLD = 1.5


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class DeadAirSegment:
    start: float
    end: float


def transcribe(video_path: str) -> tuple[list[Word], list[DeadAirSegment]]:
    console.print("[cyan]Loading whisper model...[/cyan]")
    try:
        model = WhisperModel("base", device="cuda", compute_type="float16", num_workers=4)
        console.print("[green]Using GPU acceleration for transcription.[/green]")
    except Exception:
        console.print("[yellow]GPU not available, falling back to CPU...[/yellow]")
        cores = os.cpu_count() or 4
        model = WhisperModel(
            "base", 
            device="cpu", 
            compute_type="int8", 
            cpu_threads=1, 
            num_workers=cores
        )

    console.print("[cyan]Transcribing (word-level)...[/cyan]")
    segments, _ = model.transcribe(
        video_path, 
        beam_size=1,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    words: list[Word] = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append(Word(text=w.word.strip(), start=w.start, end=w.end))

    dead_air = _build_dead_air_map(words)
    console.print(
        f"[green]Transcription complete: {len(words)} words, "
        f"{len(dead_air)} dead air segments identified.[/green]"
    )
    return words, dead_air


def _build_dead_air_map(words: list[Word]) -> list[DeadAirSegment]:
    dead_air: list[DeadAirSegment] = []

    for i, word in enumerate(words):
        if word.text.lower() in FILLER_WORDS:
            dead_air.append(DeadAirSegment(start=word.start, end=word.end))

        if i < len(words) - 1:
            gap = words[i + 1].start - word.end
            if gap > DEAD_AIR_THRESHOLD:
                dead_air.append(DeadAirSegment(start=word.end, end=words[i + 1].start))

    return dead_air


def words_to_transcript_text(words: list[Word]) -> str:
    if not words:
        return ""
        
    lines = []
    current_line_words = []
    block_start = words[0].start
    
    for w in words:
        current_line_words.append(w.text)
        # Group into ~5 second blocks to prevent token bloat
        if w.end - block_start >= 5.0:
            text = " ".join(current_line_words)
            lines.append(f"[{block_start:.1f} - {w.end:.1f}] {text}")
            current_line_words = []
            block_start = w.end
            
    if current_line_words:
        text = " ".join(current_line_words)
        lines.append(f"[{block_start:.1f} - {words[-1].end:.1f}] {text}")
        
    return "\n".join(lines)
