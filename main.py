import sys
from pathlib import Path

from InquirerPy import inquirer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.env_manager import ensure_directories, scan_videos
from core.transcriber import transcribe
from core.ai_evaluator import evaluate_chunks
from core.editor import render_clip

console = Console()


def print_banner():
    console.print(
        Panel.fit(
            "[bold yellow]AutoClip[/bold yellow] — AI-Powered Viral Clip Generator",
            border_style="bright_blue",
        )
    )


def main():
    print_banner()

    raw_dir, clips_dir = ensure_directories()

    videos = scan_videos(raw_dir)
    if not videos:
        console.print("[red]No .mp4 or .mkv files found in Raw/.[/red]")
        sys.exit(0)

    choices = [str(v.name) for v in videos]
    selected_name = inquirer.select(
        message="Select a video to process:",
        choices=choices,
    ).execute()

    selected_video = raw_dir / selected_name
    console.print(f"\n[bold]Processing:[/bold] {selected_video.name}\n")

    # Phase 2: Transcription
    words, dead_air = transcribe(str(selected_video))

    # Phase 3: AI evaluation
    above_candidates, below_candidates = evaluate_chunks(words)

    all_candidates = above_candidates + below_candidates
    if not all_candidates:
        console.print("[yellow]No clips found by the AI.[/yellow]")
        sys.exit(0)

    # Sort all candidates from best to worst
    all_candidates.sort(key=lambda c: c.score, reverse=True)

    console.print(f"\n[bold green]The AI found a total of {len(all_candidates)} potential clips![/bold green]")
    
    try:
        num_clips_str = inquirer.text(
            message=f"How many clips do you want to render? (Max {len(all_candidates)}):",
            default=str(min(5, len(all_candidates)))
        ).execute()
        num_clips = int(num_clips_str)
    except Exception:
        num_clips = min(5, len(all_candidates))
        
    num_clips = max(1, min(num_clips, len(all_candidates)))
    
    all_clips = all_candidates[:num_clips]

    table = Table(title=f"Top {num_clips} Viral Clips Selected")
    table.add_column("#", style="dim")
    table.add_column("Score", style="bold green")
    table.add_column("Segment", style="cyan")
    table.add_column("Reason")
    for i, c in enumerate(all_clips, 1):
        table.add_row(
            str(i),
            f"{c.score}/20",
            f"{c.start_time:.1f}s - {c.end_time:.1f}s",
            c.reason,
        )
    console.print(table)

    # Phase 4 & 5: Rendering
    video_name = selected_video.stem
    output_dir = clips_dir / video_name
    output_dir.mkdir(exist_ok=True)

    rendered = 0
    for i, candidate in enumerate(all_clips, 1):
        console.print(f"\n[bold]Rendering clip {i}/{len(all_clips)}...[/bold]")
        result = render_clip(
            video_path=str(selected_video),
            clip_start=candidate.start_time,
            clip_end=candidate.end_time,
            score=candidate.score,
            reason=candidate.reason,
            words=words,
            dead_air=dead_air,
            output_dir=output_dir,
            clip_index=i,
        )
        if result:
            rendered += 1

    console.print(
        f"\n[bold green]Done![/bold green] {rendered}/{len(all_clips)} clip(s) "
        f"saved to {output_dir}/"
    )


if __name__ == "__main__":
    main()
