import os
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm

console = Console()

SUPPORTED_EXTENSIONS = (".mp4", ".mkv")


def ensure_directories() -> tuple[Path, Path]:
    base = Path.cwd()
    raw_dir = base / "Raw"
    clips_dir = base / "Clips"

    missing = []
    if not raw_dir.exists():
        missing.append("Raw")
    if not clips_dir.exists():
        missing.append("Clips")

    if missing:
        console.print(f"[yellow]Missing folders: {', '.join(missing)}[/yellow]")
        if Confirm.ask("Create required folders?", default=True):
            raw_dir.mkdir(exist_ok=True)
            clips_dir.mkdir(exist_ok=True)
            console.print("[green]Folders created.[/green]")
        else:
            console.print("[red]Cannot proceed without required folders. Exiting.[/red]")
            sys.exit(1)

    return raw_dir, clips_dir


def scan_videos(raw_dir: Path) -> list[Path]:
    videos = [
        f
        for f in sorted(raw_dir.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return videos
