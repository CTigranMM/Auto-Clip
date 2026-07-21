import json
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from core.transcriber import Word, words_to_transcript_text

console = Console()

MODEL_NAME = "llama3:8b"
SCORE_THRESHOLD = 12

SYSTEM_PROMPT = (
    "You are a viral TikTok editor. Read this transcript segment with timestamps. "
    "Identify multiple highly entertaining, stupid, violent, or emotional moments. "
    "Rate each moment's virality out of 20. Return your response STRICTLY as a JSON ARRAY of objects, "
    "where each object is: { \"s\": 1-20, \"r\": \"short description\", "
    "\"st\": <float seconds>, \"et\": <float seconds>, \"pt\": <float seconds> }. "
    "The 'pt' (punchline_time) must be the exact timestamp of the most impactful word or climax moment. "
    "CRITICAL: Clip length must be dynamically sized to fit the full context! "
    "Try to fit the context between 20-45 seconds. Make it shorter if the context is small, "
    "and make it longer (e.g. 60+ seconds) if they talk about it longer. "
    "Find as many clips as you can (aim for 2 to 5 per segment if possible). "
    "The st, et, and pt must be actual timestamp values from the transcript. "
    "If there are NO highly entertaining or viral moments in this segment, return an empty array: []"
)


@dataclass
class ClipCandidate:
    score: int
    reason: str
    start_time: float
    end_time: float
    punchline_time: float = 0.0


def _chunk_words(words: list[Word], target_duration: float = 75.0, overlap_duration: float = 30.0) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    if not words:
        return chunks
        
    total_duration = words[-1].end
    start_time = 0.0
    
    while start_time < total_duration:
        end_time = start_time + target_duration
        chunk = [w for w in words if start_time <= w.start < end_time]
        
        if chunk:
            chunks.append(chunk)
            
        start_time += (target_duration - overlap_duration)

    return chunks


def _parse_llm_json(raw: str) -> dict | None:
    # Attempt 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract JSON substring (Array or Object)
    match = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Attempt 3: fix smart quotes and trailing commas
            fixed = candidate.replace("\u201c", '"').replace("\u201d", '"')
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    return None


def _evaluate_single_chunk(i: int, chunk: list[Word]) -> tuple[int, list[ClipCandidate]]:
    # A viral clip needs at least 60 words to have any meaningful dialogue/context.
    # If the chunk has fewer than 60 words, it's mostly silence, music, or breathing.
    if len(chunk) < 60:
        return i, []
        
    transcript = words_to_transcript_text(chunk)
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            format="json",
            options={
                "temperature": 0.0,
                "num_ctx": 4096,
                "num_predict": 1024,
            },
        )
        raw = response["message"]["content"]
    except Exception as e:
        return i, []

    parsed = _parse_llm_json(raw)
    if parsed is None:
        return i, []

    if isinstance(parsed, dict) and "clips" in parsed:
        parsed = parsed["clips"]
    elif isinstance(parsed, dict):
        parsed = [parsed]
        
    if not isinstance(parsed, list):
        return i, []

    candidates = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
            
        score = item.get("s", item.get("score", 0))

        if not isinstance(score, (int, float)) or score == 0:
            continue

        try:
            start_time = float(item.get("st", item.get("start_time", chunk[0].start)))
        except (ValueError, TypeError):
            start_time = chunk[0].start
            
        try:
            end_time = float(item.get("et", item.get("end_time", chunk[-1].end)))
        except (ValueError, TypeError):
            end_time = chunk[-1].end

        # Remove the strict time restriction! We only enforce a bare minimum 15-second 
        # safety pad so the short-form video rendering doesn't look like a glitch.
        if end_time - start_time < 15.0:
            diff = 15.0 - (end_time - start_time)
            start_time = start_time - diff * 0.75
            end_time = end_time + diff * 0.25
            
        start_time = max(0.0, start_time)

        try:
            punch_time = float(item.get("pt", item.get("punchline_time", start_time)))
        except (ValueError, TypeError):
            punch_time = start_time
            
        candidate = ClipCandidate(
            score=int(score),
            reason=item.get("r", item.get("reason", "No description")),
            start_time=start_time,
            end_time=end_time,
            punchline_time=punch_time,
        )
        candidates.append(candidate)
        
    return i, candidates


def evaluate_chunks(
    words: list[Word],
) -> tuple[list[ClipCandidate], list[ClipCandidate]]:
    total_duration = words[-1].end if words else 0
    if total_duration > 3600:
        chunk_size = 300.0
    elif total_duration > 1200:
        chunk_size = 240.0
    else:
        chunk_size = 120.0

    chunks = _chunk_words(words, target_duration=chunk_size, overlap_duration=30.0)
    above: list[ClipCandidate] = []
    below: list[ClipCandidate] = []

    console.print(f"[cyan]Analyzing {len(chunks)} transcript chunks with {MODEL_NAME}...[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Evaluating...", total=len(chunks))
        
        # We MUST use max_workers=1 for local LLMs like Ollama. Hitting a local model with 
        # concurrent requests causes massive VRAM thrashing, drops the KV cache, and forces 
        # Out-Of-Memory CPU spilling which slows inference by 10x. Sequential processing is faster!
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {executor.submit(_evaluate_single_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
            
            for future in as_completed(futures):
                i = futures[future]
                _, chunk_candidates = future.result()
                progress.advance(task)
                
                for candidate in chunk_candidates:
                    if candidate.score >= SCORE_THRESHOLD:
                        above.append(candidate)
                    else:
                        below.append(candidate)

    # Sort results by start_time to maintain chronological order
    above.sort(key=lambda c: c.start_time)
    below.sort(key=lambda c: c.start_time)

    # Deduplicate overlapping clips from the sliding windows
    def deduplicate(clips: list[ClipCandidate]) -> list[ClipCandidate]:
        unique = []
        for c in clips:
            is_dup = False
            for u in unique:
                if abs(c.start_time - u.start_time) < 15.0:
                    # Keep the higher scoring one
                    if c.score > u.score:
                        u.score = c.score
                        u.reason = c.reason
                        u.end_time = c.end_time
                        u.punchline_time = c.punchline_time
                    is_dup = True
                    break
            if not is_dup:
                unique.append(c)
        return unique

    above = deduplicate(above)
    below = deduplicate(below)

    console.print(f"[bold green]Found {len(above)} clip(s) >= {SCORE_THRESHOLD}/20, "
                  f"{len(below)} below threshold[/bold green]")
    return above, below
