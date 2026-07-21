import os
from pathlib import Path
from contextlib import ExitStack

import numpy as np
from PIL import Image
import cv2
import math
from moviepy import (
    VideoFileClip,
    TextClip,
    concatenate_videoclips,
    CompositeVideoClip,
)
from rich.console import Console

from core.transcriber import Word, DeadAirSegment

console = Console()

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
HOOK_ZOOM = 1.10
PUNCH_ZOOM = 1.20
HOOK_ZOOM_DURATION = 3.0
PUNCH_ZOOM_DURATION = 0.5
MIN_DEAD_AIR = 0.5
SUBTITLE_FONT_SIZE = 60
SUBTITLE_STROKE_WIDTH = 4
WORDS_PER_GROUP = 2


def _find_font() -> str:
    candidates = [
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/consolab.ttf",
        "C:/Windows/Fonts/verdanab.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def _ease_in_out(t: float, duration: float) -> float:
    progress = min(t / duration, 1.0)
    return progress * progress * (3 - 2 * progress)


def _map_time(time_val: float, segments: list[tuple[float, float]]) -> float:
    mapped = 0.0
    for seg_start, seg_end in segments:
        if time_val > seg_end:
            mapped += (seg_end - seg_start)
        elif time_val >= seg_start:
            mapped += (time_val - seg_start)
            return mapped
    return mapped


def _apply_smart_vertical_crop_and_zooms(
    clip: VideoFileClip,
    cut_points: list[float],
    punch_time: float = -1.0,
) -> VideoFileClip:
    src_w, src_h = clip.size
    target_ratio = TARGET_WIDTH / TARGET_HEIGHT
    
    # Precalculate face centers for each segment
    centers = []
    seg_starts = [0.0] + cut_points
    seg_ends = cut_points + [clip.duration]
    
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
    except Exception:
        face_cascade = None
        
    for start_t, end_t in zip(seg_starts, seg_ends):
        mid_t = (start_t + end_t) / 2.0
        x_center = src_w // 2
        
        if face_cascade:
            try:
                frame = clip.get_frame(mid_t)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                if len(faces) > 0:
                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    x, y, w_face, h_face = largest_face
                    x_center = x + w_face // 2
            except Exception:
                pass
        centers.append(x_center)

    def dynamic_filter(get_frame, t):
        frame = get_frame(t)
        
        idx = 0
        for i, end_t in enumerate(seg_ends):
            if t <= end_t + 0.001:
                idx = i
                break
        
        target_center = centers[idx]
        
        hook_progress = _ease_in_out(t, HOOK_ZOOM_DURATION)
        hook_scale = 1.0 + (HOOK_ZOOM - 1.0) * hook_progress

        punch_scale = 1.0
        for cp in cut_points:
            time_since_cut = t - cp
            if 0 <= time_since_cut < PUNCH_ZOOM_DURATION:
                punch_progress = 1.0 - (time_since_cut / PUNCH_ZOOM_DURATION)
                punch_scale = max(
                    punch_scale,
                    1.0 + (PUNCH_ZOOM - 1.0) * punch_progress,
                )

        super_punch_scale = 1.0
        if punch_time >= 0:
            time_since_punch = t - punch_time
            if 0 <= time_since_punch < 1.0:
                punch_progress = 1.0 - (time_since_punch / 1.0)
                super_punch_scale = 1.0 + (1.5 - 1.0) * punch_progress

        if src_w / src_h > target_ratio:
            base_w = int(src_h * target_ratio)
            base_h = src_h
        else:
            base_w = src_w
            base_h = int(src_w / target_ratio)
            
        final_scale = max(hook_scale, punch_scale, super_punch_scale)
        # Dynamic camera wobble
        amp_x = 12.0
        amp_y = 8.0
        freq_x = 0.5
        freq_y = 0.3
        shift_x = int(amp_x * math.sin(t * freq_x * math.pi))
        shift_y = int(amp_y * math.cos(t * freq_y * math.pi))
        
        final_scale = max(final_scale, 1.05)
        
        crop_w = int(base_w / final_scale)
        crop_h = int(base_h / final_scale)
        
        if src_w / src_h > target_ratio:
            x1 = target_center - crop_w // 2 + shift_x
            y1 = (src_h - crop_h) // 2 + shift_y
        else:
            x1 = (src_w - crop_w) // 2 + shift_x
            y1 = (src_h - crop_h) // 2 + shift_y
            
        x1 = max(0, min(x1, src_w - crop_w))
        y1 = max(0, min(y1, src_h - crop_h))
        
        cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
        
        resized = cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
        return resized

    return clip.transform(dynamic_filter)


def _remove_dead_air(
    clip: VideoFileClip,
    clip_start: float,
    dead_air: list[DeadAirSegment],
    stack: ExitStack,
) -> tuple[VideoFileClip, list[float], list[tuple[float, float]]]:
    adjusted: list[tuple[float, float]] = []

    for da in dead_air:
        dur = da.end - da.start
        if dur < MIN_DEAD_AIR:
            continue
        adj_start = da.start - clip_start
        adj_end = da.end - clip_start
        adj_start = max(0.0, adj_start)
        adj_end = min(clip.duration, adj_end)
        if adj_end - adj_start >= MIN_DEAD_AIR:
            adjusted.append((adj_start, adj_end))

    adjusted.sort(key=lambda x: x[0])

    segments: list[tuple[float, float]] = []
    prev_end = 0.0
    for da_start, da_end in adjusted:
        if da_start > prev_end:
            segments.append((prev_end, da_start))
        prev_end = da_end
    if prev_end < clip.duration:
        segments.append((prev_end, clip.duration))

    if not segments:
        return clip, [], []

    sub_clips = []
    cut_points: list[float] = []
    cumulative = 0.0

    kept_segments: list[tuple[float, float]] = []
    for seg_start, seg_end in segments:
        duration = seg_end - seg_start
        if duration < 0.1:
            continue
        sub = clip.subclipped(seg_start, seg_end)
        stack.callback(sub.close)
        if hasattr(sub, "audio") and sub.audio:
            stack.callback(sub.audio.close)
            
        sub_clips.append(sub)
        kept_segments.append((seg_start, seg_end))
        
        if len(sub_clips) > 1:
            cut_points.append(cumulative)
            
        cumulative += duration

    if not sub_clips:
        return clip, [], []

    if len(sub_clips) == 1:
        return sub_clips[0], cut_points, kept_segments

    return concatenate_videoclips(sub_clips, method="chain"), cut_points, kept_segments


# _apply_zooms has been merged into _apply_smart_vertical_crop_and_zooms


def _create_subtitles(
    words: list[Word],
    clip_start: float,
    clip_end: float,
    segments: list[tuple[float, float]],
    font_path: str,
) -> list[TextClip]:
    groups: list[list[Word]] = []
    current_group: list[Word] = []

    for word in words:
        if word.end < clip_start or word.start > clip_end:
            continue

        mapped_start = _map_time(max(0.0, word.start - clip_start), segments)
        mapped_end = _map_time(max(0.0, word.end - clip_start), segments)

        if mapped_end - mapped_start < 0.01:
            continue

        # Create a new Word with mapped times just for subtitle processing
        mapped_word = Word(text=word.text, start=mapped_start, end=mapped_end)
        current_group.append(mapped_word)
        
        if len(current_group) >= 5:  # Changed to 5 for karaoke style
            groups.append(current_group)
            current_group = []
            
    if current_group:
        groups.append(current_group)

    subtitle_clips = []
    for group in groups:
        group_start = group[0].start
        group_end = group[-1].end
        
        for i, word in enumerate(group):
            # Append a newline to force ImageMagick to pad the bottom 
            # and prevent cutting off descenders (y, g, p) and bottom strokes.
            sub_phrase = " ".join(w.text for w in group[:i+1]) + "\n "
            start_t = word.start
            if i < len(group) - 1:
                end_t = group[i+1].start
            else:
                end_t = group_end
                
            duration = end_t - start_t
            if duration < 0.05:
                continue

            try:
                tc = TextClip(
                    text=sub_phrase,
                    font_size=SUBTITLE_FONT_SIZE,
                    color="yellow",
                    font=font_path,
                    stroke_color="black",
                    stroke_width=SUBTITLE_STROKE_WIDTH,
                    method="caption",
                    text_align="center",
                    size=(TARGET_WIDTH - 80, None),
                ).with_start(start_t).with_duration(duration)
                
                # We added a newline, which adds ~SUBTITLE_FONT_SIZE height to tc.size[1].
                # To keep the visual position the same, we reduce the bottom anchor offset.
                anchor_offset = 350 - SUBTITLE_FONT_SIZE
                tc = tc.with_position(("center", TARGET_HEIGHT - anchor_offset - tc.size[1]))
                subtitle_clips.append(tc)
            except Exception:
                pass

    return subtitle_clips


def render_clip(
    video_path: str,
    clip_start: float,
    clip_end: float,
    score: int,
    reason: str,
    words: list[Word],
    dead_air: list[DeadAirSegment],
    output_dir: Path,
    clip_index: int,
    punchline_time: float = 0.0,
) -> Path | None:
    video_name = Path(video_path).stem
    output_path = output_dir / f"{video_name}_clip{clip_index}.mp4"
    meta_path = output_dir / f"{video_name}_clip{clip_index}_meta.txt"

    try:
        with ExitStack() as stack:
            console.print(f"  [cyan]Loading video...[/cyan]")
            full_clip = VideoFileClip(video_path)
            stack.callback(full_clip.close)
            if hasattr(full_clip, "audio") and full_clip.audio:
                stack.callback(full_clip.audio.close)

            clip_start = max(0.0, clip_start)
            clip_end = min(full_clip.duration, clip_end)

            console.print(
                f"  [cyan]Extracting segment ({clip_start:.1f}s - {clip_end:.1f}s)...[/cyan]"
            )
            segment = full_clip.subclipped(clip_start, clip_end)
            stack.callback(segment.close)
            if hasattr(segment, "audio") and segment.audio:
                stack.callback(segment.audio.close)
                
            console.print(f"  [dim]Segment duration: {segment.duration:.1f}s[/dim]")

            console.print(f"  [cyan]Removing dead air...[/cyan]")
            paced_clip, cut_points, segments = _remove_dead_air(segment, clip_start, dead_air, stack)
            stack.callback(paced_clip.close)
            if hasattr(paced_clip, "audio") and paced_clip.audio:
                stack.callback(paced_clip.audio.close)
            console.print(
                f"  [dim]After dead air removal: {paced_clip.duration:.1f}s, "
                f"{len(cut_points)} cut points[/dim]"
            )

            mapped_punch_time = -1.0
            if punchline_time > clip_start:
                mapped_punch_time = _map_time(punchline_time - clip_start, segments)

            console.print(f"  [cyan]Applying smart vertical crop and dynamic camera...[/cyan]")
            zoomed_clip = _apply_smart_vertical_crop_and_zooms(paced_clip, cut_points, mapped_punch_time)
            stack.callback(zoomed_clip.close)

            font_path = _find_font()
            if font_path:
                console.print(f"  [cyan]Adding dynamic subtitles...[/cyan]")
                subtitle_clips = _create_subtitles(
                    words, clip_start, clip_end, segments, font_path
                )
                for tc in subtitle_clips:
                    stack.callback(tc.close)

                if subtitle_clips:
                    final = CompositeVideoClip(
                        [zoomed_clip] + subtitle_clips,
                        size=(TARGET_WIDTH, TARGET_HEIGHT),
                    ).with_duration(zoomed_clip.duration)
                    stack.callback(final.close)
                else:
                    final = zoomed_clip
            else:
                console.print("  [yellow]No bold font found, skipping subtitles.[/yellow]")
                final = zoomed_clip

            import os
            render_threads = max(4, os.cpu_count() or 4)
            console.print(f"  [cyan]Rendering to {output_path.name} ({final.duration:.1f}s) with {render_threads} threads...[/cyan]")
            final.write_videofile(
                str(output_path),
                preset="ultrafast",
                threads=render_threads,
                audio_codec="aac",
                logger=None,
            )

            meta_path.write_text(
                f"Score: {score}/20\nReason: {reason}\n"
                f"Source: {video_name}\n"
                f"Segment: {clip_start:.1f}s - {clip_end:.1f}s\n",
                encoding="utf-8",
            )
            console.print(f"  [green]Saved: {output_path.name} ({final.duration:.1f}s)[/green]")
            return output_path

    except Exception as e:
        console.print(f"  [red]Render error: {e}[/red]")
        return None
