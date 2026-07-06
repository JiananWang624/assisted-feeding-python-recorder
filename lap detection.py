from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import find_peaks


DEFAULT_TRIAL_DIR = Path(
    r"D:\Files\temp\data_collection_pipeline\python_recorder\data\trial_001_test"
)


@dataclass
class Detection:
    role: str
    region_start_s: float
    region_end_s: float
    sample_index: int
    time_in_wav_s: float
    host_time_s: float
    peak_amplitude: float
    nearest_rs_frame_index: int | None = None
    nearest_rs_host_time_s: float | None = None
    nearest_rs_dt_s: float | None = None
    nearest_rgb_file: str | None = None
    nearest_depth_file: str | None = None
    previous_rs_frame_index: int | None = None
    previous_rs_host_time_s: float | None = None
    previous_rs_dt_s: float | None = None
    previous_rgb_file: str | None = None
    next_rs_frame_index: int | None = None
    next_rs_host_time_s: float | None = None
    next_rs_dt_s: float | None = None
    next_rgb_file: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find first/last clap sync events and save waveform plots."
    )
    parser.add_argument("--trial_dir", type=Path, default=DEFAULT_TRIAL_DIR)
    parser.add_argument(
        "--short_audio_limit",
        type=float,
        default=10.0,
        help="If audio is no longer than this, search this initial window for first and last clap.",
    )
    parser.add_argument(
        "--edge_window",
        type=float,
        default=5.0,
        help="For longer audio, search the first and last N seconds.",
    )
    parser.add_argument("--noise_duration", type=float, default=1.0)
    parser.add_argument("--threshold_multiplier", type=float, default=8.0)
    parser.add_argument("--min_threshold", type=float, default=0.005)
    parser.add_argument("--min_distance", type=float, default=0.25)
    parser.add_argument("--plot_window", type=float, default=0.5)
    return parser.parse_args()


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float64), int(sr)


def estimate_threshold(
    audio: np.ndarray,
    sr: int,
    noise_duration_s: float,
    threshold_multiplier: float,
    min_threshold: float,
) -> tuple[float, float]:
    noise_samples = max(1, min(len(audio), int(noise_duration_s * sr)))
    noise = audio[:noise_samples]
    noise_level = float(np.std(noise))
    threshold = max(float(min_threshold), noise_level * float(threshold_multiplier))
    return noise_level, threshold


def sample_to_host_time_s(sample_index: int, blocks: pd.DataFrame, sr: int) -> float:
    starts = blocks["first_sample_index"].to_numpy(dtype=np.int64)
    frames = blocks["frames"].to_numpy(dtype=np.int64)
    callback_times = blocks["host_time_s"].to_numpy(dtype=np.float64)

    block_idx = int(np.searchsorted(starts, sample_index, side="right") - 1)
    block_idx = max(0, min(block_idx, len(blocks) - 1))

    first_sample_index = int(starts[block_idx])
    block_frames = int(frames[block_idx])
    callback_host_time_s = float(callback_times[block_idx])

    # sounddevice callback arrives after the block is available. Estimate the
    # first sample time by subtracting the block duration.
    block_start_host_time_s = callback_host_time_s - block_frames / sr
    return block_start_host_time_s + (sample_index - first_sample_index) / sr


def find_realsense_context(
    host_time_s: float, realsense_rows: pd.DataFrame | None
) -> dict:
    if realsense_rows is None or len(realsense_rows) == 0:
        return {}

    times = realsense_rows["host_time_s"].to_numpy(dtype=np.float64)
    nearest_idx = int(np.argmin(np.abs(times - host_time_s)))
    nearest = realsense_rows.iloc[nearest_idx]

    next_idx = int(np.searchsorted(times, host_time_s, side="left"))
    next_idx = min(next_idx, len(realsense_rows) - 1)
    prev_idx = max(0, next_idx - 1)
    previous = realsense_rows.iloc[prev_idx]
    next_row = realsense_rows.iloc[next_idx]

    return {
        "nearest_rs_frame_index": int(nearest["frame_index"]),
        "nearest_rs_host_time_s": float(nearest["host_time_s"]),
        "nearest_rs_dt_s": float(nearest["host_time_s"]) - host_time_s,
        "nearest_rgb_file": str(nearest["rgb_file"]),
        "nearest_depth_file": str(nearest["depth_file"]),
        "previous_rs_frame_index": int(previous["frame_index"]),
        "previous_rs_host_time_s": float(previous["host_time_s"]),
        "previous_rs_dt_s": float(previous["host_time_s"]) - host_time_s,
        "previous_rgb_file": str(previous["rgb_file"]),
        "next_rs_frame_index": int(next_row["frame_index"]),
        "next_rs_host_time_s": float(next_row["host_time_s"]),
        "next_rs_dt_s": float(next_row["host_time_s"]) - host_time_s,
        "next_rgb_file": str(next_row["rgb_file"]),
    }


def find_region_peaks(
    audio: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    threshold: float,
    min_distance_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    start_sample = max(0, int(start_s * sr))
    end_sample = min(len(audio), int(end_s * sr))
    if end_sample <= start_sample:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    segment = np.abs(audio[start_sample:end_sample])
    min_distance_samples = max(1, int(min_distance_s * sr))
    peaks, properties = find_peaks(
        segment,
        height=threshold,
        distance=min_distance_samples,
    )
    if len(peaks) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    peaks = peaks + start_sample
    heights = properties["peak_heights"]
    order = np.argsort(peaks)
    return peaks[order], heights[order]


def make_detection(
    role: str,
    sample_index: int,
    peak_amplitude: float,
    region_start_s: float,
    region_end_s: float,
    blocks: pd.DataFrame,
    realsense_rows: pd.DataFrame | None,
    sr: int,
) -> Detection:
    host_time_s = sample_to_host_time_s(sample_index, blocks, sr)
    rs_info = find_realsense_context(host_time_s, realsense_rows)
    return Detection(
        role=role,
        region_start_s=float(region_start_s),
        region_end_s=float(region_end_s),
        sample_index=int(sample_index),
        time_in_wav_s=float(sample_index / sr),
        host_time_s=float(host_time_s),
        peak_amplitude=float(peak_amplitude),
        **rs_info,
    )


def detect_sync_claps(
    audio: np.ndarray,
    sr: int,
    blocks: pd.DataFrame,
    realsense_rows: pd.DataFrame | None,
    short_audio_limit_s: float,
    edge_window_s: float,
    noise_duration_s: float,
    threshold_multiplier: float,
    min_threshold: float,
    min_distance_s: float,
) -> tuple[list[Detection], float, float, list[tuple[str, float, float, int]]]:
    duration_s = len(audio) / sr
    noise_level, threshold = estimate_threshold(
        audio, sr, noise_duration_s, threshold_multiplier, min_threshold
    )

    detections: list[Detection] = []
    regions: list[tuple[str, float, float, int]] = []

    if duration_s <= short_audio_limit_s:
        region_start_s = 0.0
        region_end_s = min(duration_s, short_audio_limit_s)
        peaks, heights = find_region_peaks(
            audio, sr, region_start_s, region_end_s, threshold, min_distance_s
        )
        regions.append(("short_audio_first_last_search", region_start_s, region_end_s, len(peaks)))
        if len(peaks) == 0:
            return detections, noise_level, threshold, regions

        first_peak = int(peaks[0])
        first_height = float(heights[0])
        last_peak = int(peaks[-1])
        last_height = float(heights[-1])

        if first_peak == last_peak:
            detections.append(
                make_detection(
                    "first_and_last_clap",
                    first_peak,
                    first_height,
                    region_start_s,
                    region_end_s,
                    blocks,
                    realsense_rows,
                    sr,
                )
            )
        else:
            detections.append(
                make_detection(
                    "first_clap",
                    first_peak,
                    first_height,
                    region_start_s,
                    region_end_s,
                    blocks,
                    realsense_rows,
                    sr,
                )
            )
            detections.append(
                make_detection(
                    "last_clap",
                    last_peak,
                    last_height,
                    region_start_s,
                    region_end_s,
                    blocks,
                    realsense_rows,
                    sr,
                )
            )
        return detections, noise_level, threshold, regions

    first_start_s = 0.0
    first_end_s = min(edge_window_s, duration_s)
    first_peaks, first_heights = find_region_peaks(
        audio, sr, first_start_s, first_end_s, threshold, min_distance_s
    )
    regions.append(("long_audio_first_5s_search", first_start_s, first_end_s, len(first_peaks)))
    if len(first_peaks) > 0:
        detections.append(
            make_detection(
                "first_clap",
                int(first_peaks[0]),
                float(first_heights[0]),
                first_start_s,
                first_end_s,
                blocks,
                realsense_rows,
                sr,
            )
        )

    last_start_s = max(0.0, duration_s - edge_window_s)
    last_end_s = duration_s
    last_peaks, last_heights = find_region_peaks(
        audio, sr, last_start_s, last_end_s, threshold, min_distance_s
    )
    regions.append(("long_audio_last_5s_search", last_start_s, last_end_s, len(last_peaks)))
    if len(last_peaks) > 0:
        detections.append(
            make_detection(
                "last_clap",
                int(last_peaks[-1]),
                float(last_heights[-1]),
                last_start_s,
                last_end_s,
                blocks,
                realsense_rows,
                sr,
            )
        )

    return detections, noise_level, threshold, regions


def fallback_path(path: Path) -> Path:
    for i in range(1, 100):
        candidate = path.with_name(f"{path.stem}_{i:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{path.stem}_new{path.suffix}")


def save_detection_csv(path: Path, detections: list[Detection]) -> Path:
    fieldnames = [
        "role",
        "region_start_s",
        "region_end_s",
        "sample_index",
        "time_in_wav_s",
        "host_time_s",
        "peak_amplitude",
        "nearest_rs_frame_index",
        "nearest_rs_host_time_s",
        "nearest_rs_dt_s",
        "nearest_rgb_file",
        "nearest_depth_file",
        "previous_rs_frame_index",
        "previous_rs_host_time_s",
        "previous_rs_dt_s",
        "previous_rgb_file",
        "next_rs_frame_index",
        "next_rs_host_time_s",
        "next_rs_dt_s",
        "next_rgb_file",
    ]
    try:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for detection in detections:
                writer.writerow(detection.__dict__)
        return path
    except PermissionError:
        fallback = fallback_path(path)
        with fallback.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for detection in detections:
                writer.writerow(detection.__dict__)
        return fallback


def save_region_csv(path: Path, regions: list[tuple[str, float, float, int]]) -> Path:
    try:
        f = path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        path = fallback_path(path)
        f = path.open("w", newline="", encoding="utf-8")

    with f:
        writer = csv.DictWriter(
            f, fieldnames=["region_name", "region_start_s", "region_end_s", "candidate_count"]
        )
        writer.writeheader()
        for name, start_s, end_s, count in regions:
            writer.writerow(
                {
                    "region_name": name,
                    "region_start_s": f"{start_s:.9f}",
                    "region_end_s": f"{end_s:.9f}",
                    "candidate_count": count,
                }
            )
    return path


def save_waveform_plots(
    plot_dir: Path,
    audio: np.ndarray,
    sr: int,
    detections: list[Detection],
    threshold: float,
    plot_window_s: float,
) -> list[Path]:
    plot_dir.mkdir(exist_ok=True)
    saved_paths: list[Path] = []

    for detection in detections:
        center = detection.sample_index
        half_window = int(plot_window_s * sr)
        start = max(0, center - half_window)
        end = min(len(audio), center + half_window)
        t = np.arange(start, end) / sr

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, audio[start:end], linewidth=0.8)
        ax.axvline(
            detection.time_in_wav_s,
            color="tab:red",
            linestyle="--",
            label=detection.role,
        )
        ax.axhline(threshold, color="tab:orange", linestyle=":", label="threshold")
        ax.axhline(-threshold, color="tab:orange", linestyle=":")
        ax.set_xlabel("Time in audio.wav (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(
            f"{detection.role}: wav={detection.time_in_wav_s:.6f}s, "
            f"host={detection.host_time_s:.6f}s"
        )
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right")
        fig.tight_layout()

        path = plot_dir / f"{detection.role}_waveform.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths


def save_overview_plot(
    plot_dir: Path,
    audio: np.ndarray,
    sr: int,
    detections: list[Detection],
    regions: list[tuple[str, float, float, int]],
    threshold: float,
) -> Path:
    duration_s = len(audio) / sr
    t = np.arange(len(audio)) / sr
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, audio, linewidth=0.45, color="tab:blue")
    ax.axhline(threshold, color="tab:orange", linestyle=":", label="threshold")
    ax.axhline(-threshold, color="tab:orange", linestyle=":")

    for name, start_s, end_s, _ in regions:
        ax.axvspan(start_s, end_s, alpha=0.12, label=name)

    for detection in detections:
        ax.axvline(detection.time_in_wav_s, linestyle="--", linewidth=1.6)
        ax.text(
            detection.time_in_wav_s,
            0.95,
            detection.role,
            rotation=90,
            transform=ax.get_xaxis_transform(),
            va="top",
            ha="right",
        )

    ax.set_xlim(0, duration_s)
    ax.set_xlabel("Time in audio.wav (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sync clap search overview")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right")
    fig.tight_layout()

    path = plot_dir / "sync_clap_overview.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    trial_dir = args.trial_dir.resolve()
    audio_path = trial_dir / "audio.wav"
    blocks_path = trial_dir / "audio_blocks.csv"
    realsense_path = trial_dir / "realsense_timestamps.csv"
    plot_dir = trial_dir / "sync_plots"

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    if not blocks_path.exists():
        raise FileNotFoundError(blocks_path)

    audio, sr = load_audio_mono(audio_path)
    blocks = pd.read_csv(blocks_path)
    realsense_rows = pd.read_csv(realsense_path) if realsense_path.exists() else None

    detections, noise_level, threshold, regions = detect_sync_claps(
        audio=audio,
        sr=sr,
        blocks=blocks,
        realsense_rows=realsense_rows,
        short_audio_limit_s=args.short_audio_limit,
        edge_window_s=args.edge_window,
        noise_duration_s=args.noise_duration,
        threshold_multiplier=args.threshold_multiplier,
        min_threshold=args.min_threshold,
        min_distance_s=args.min_distance,
    )

    plot_dir.mkdir(exist_ok=True)
    csv_path = plot_dir / "sync_clap_detections.csv"
    region_csv_path = plot_dir / "sync_clap_search_regions.csv"
    csv_path = save_detection_csv(csv_path, detections)
    region_csv_path = save_region_csv(region_csv_path, regions)
    plot_paths = save_waveform_plots(
        plot_dir=plot_dir,
        audio=audio,
        sr=sr,
        detections=detections,
        threshold=threshold,
        plot_window_s=args.plot_window,
    )
    overview_path = save_overview_plot(plot_dir, audio, sr, detections, regions, threshold)

    print(f"Trial dir: {trial_dir}")
    print(f"Sample rate: {sr} Hz")
    print(f"Audio duration: {len(audio) / sr:.6f} s")
    print(f"Noise level: {noise_level:.6f}")
    print(f"Threshold: {threshold:.6f}")
    for name, start_s, end_s, count in regions:
        print(f"Search region: {name}, {start_s:.3f}-{end_s:.3f}s, candidates={count}")

    if not detections:
        print("No sync clap detected. Try lowering --min_threshold or --threshold_multiplier.")
    else:
        print(f"Selected sync detections: {len(detections)}")
        for detection in detections:
            rs_dt_ms = (
                "NA"
                if detection.nearest_rs_dt_s is None
                else f"{detection.nearest_rs_dt_s * 1000:.3f}"
            )
            print(
                f"{detection.role}: wav={detection.time_in_wav_s:.6f}s, "
                f"host={detection.host_time_s:.6f}s, amp={detection.peak_amplitude:.4f}, "
                f"nearest_rs={detection.nearest_rs_frame_index}, rs_dt_ms={rs_dt_ms}, "
                f"rgb={detection.nearest_rgb_file}, "
                f"prev_rs={detection.previous_rs_frame_index}, next_rs={detection.next_rs_frame_index}"
            )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved region CSV: {region_csv_path}")
    print(f"Saved overview plot: {overview_path}")
    for path in plot_paths:
        print(f"Saved waveform plot: {path}")


if __name__ == "__main__":
    main()
