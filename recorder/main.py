from __future__ import annotations

import argparse
import json
import platform
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from .audio_recorder import AudioRecorder
from .clock import RecorderClock
from .config import AudioConfig, OptiTrackConfig, RealSenseConfig, RecorderConfig
from .optitrack_recorder import OptiTrackNatNetAdapter
from .realsense_recorder import RealSenseRecorder
from .writers import ensure_trial_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record RealSense RGB-D, microphone audio, and OptiTrack NatNet data with one host clock."
    )
    parser.add_argument("--out_dir", required=True, help="Output trial folder, e.g. trial_001")
    parser.add_argument("--duration", type=float, default=None, help="Recording duration in seconds")
    parser.add_argument("--no_realsense", action="store_true", help="Disable RealSense recording")
    parser.add_argument("--no_audio", action="store_true", help="Disable audio recording")
    parser.add_argument("--no_optitrack", action="store_true", help="Disable OptiTrack CSV adapter")
    parser.add_argument("--audio_device", default=None, help="sounddevice input device index or name")
    parser.add_argument("--audio_blocksize", type=int, default=1024)
    parser.add_argument("--rs_width", type=int, default=640)
    parser.add_argument("--rs_height", type=int, default=480)
    parser.add_argument("--rs_fps", type=int, default=30)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--rs_warmup_frames", type=int, default=15)
    parser.add_argument(
        "--optitrack_natnet_path",
        default=None,
        help="Path to official NatNet Python sample folder or NatNetClient.py",
    )
    parser.add_argument("--optitrack_server", default="127.0.0.1", help="Motive/NatNet server IP")
    parser.add_argument("--optitrack_client", default="127.0.0.1", help="Local client IP")
    parser.add_argument("--optitrack_data_port", type=int, default=1511, help="NatNet data UDP port")
    parser.add_argument(
        "--optitrack_multicast_address",
        default="239.255.42.99",
        help="NatNet multicast group address",
    )
    parser.add_argument(
        "--optitrack_unicast",
        action="store_true",
        help="Use unicast instead of multicast for NatNet.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RecorderConfig:
    audio_device = args.audio_device
    if isinstance(audio_device, str) and audio_device.isdigit():
        audio_device = int(audio_device)
    return RecorderConfig(
        out_dir=args.out_dir,
        duration_s=args.duration,
        realsense=RealSenseConfig(
            enabled=not args.no_realsense,
            width=args.rs_width,
            height=args.rs_height,
            fps=args.rs_fps,
            jpeg_quality=args.jpeg_quality,
            warmup_frames=args.rs_warmup_frames,
        ),
        audio=AudioConfig(
            enabled=not args.no_audio,
            device=audio_device,
            blocksize=args.audio_blocksize,
        ),
        optitrack=OptiTrackConfig(
            enabled=not args.no_optitrack,
            natnet_path=args.optitrack_natnet_path,
            server_address=args.optitrack_server,
            client_address=args.optitrack_client,
            use_multicast=not args.optitrack_unicast,
            data_port=args.optitrack_data_port,
            multicast_address=args.optitrack_multicast_address,
        ),
    )


def write_metadata(
    out_dir: Path,
    config: RecorderConfig,
    clock: RecorderClock,
    started_at_utc: str,
    stopped_at_utc: str | None = None,
    stats: dict | None = None,
) -> None:
    metadata = {
        "dataset_context": "human-to-human assisted feeding interaction",
        "sync_design": "software synchronization using one host monotonic clock",
        "host_clock": {
            "python_function": "time.perf_counter_ns",
            "start_perf_counter_ns": clock.start_ns if clock.is_started else None,
            "timestamp_field": "host_time_s",
            "timestamp_definition": "seconds elapsed since recorder start",
        },
        "started_at_utc": started_at_utc,
        "stopped_at_utc": stopped_at_utc,
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "config": config.to_dict(),
        "stats": stats or {},
        "notes": [
            "RealSense depth frames are aligned to color before saving.",
            "Audio callback only enqueues data; audio.wav and audio_blocks.csv are written by a writer thread.",
            "OptiTrack rows are written through OptiTrackNatNetAdapter; connect the NatNet frame callback to submit_rigid_body_frame().",
            "This recorder does not implement hardware trigger synchronization.",
        ],
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    out_dir = Path(config.out_dir).resolve()
    ensure_trial_dirs(out_dir)

    clock = RecorderClock()
    started_at_utc = None

    stop_requested = False

    def _request_stop(signum, frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"\n[Main] Stop requested by signal {signum}")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    recorders = []
    realsense = None
    audio = None
    optitrack = None
    stop_requested_host_time_s = None

    try:
        if config.realsense.enabled:
            realsense = RealSenseRecorder(out_dir, clock, config.realsense)
            realsense.start()
            recorders.append(realsense)

        if config.audio.enabled:
            audio = AudioRecorder(out_dir, clock, config.audio)
            audio.start()
            recorders.append(audio)

        if config.optitrack.enabled:
            optitrack = OptiTrackNatNetAdapter(out_dir, clock, config.optitrack)
            optitrack.start()
            recorders.append(optitrack)

        clock.start()
        started_at_utc = datetime.now(timezone.utc).isoformat()
        write_metadata(out_dir, config, clock, started_at_utc)

        print(f"[Main] Recording to {out_dir}")
        if config.duration_s is None:
            print("[Main] Press Ctrl+C to stop")
        else:
            print(f"[Main] Duration: {config.duration_s:.3f} s")

        last_status_s = 0.0
        while not stop_requested:
            elapsed_s = clock.now_s()
            if config.duration_s is not None and elapsed_s >= config.duration_s:
                break
            if elapsed_s - last_status_s >= 5.0:
                last_status_s = elapsed_s
                status = []
                if realsense is not None:
                    status.append(f"rs captured={realsense.frames_captured} queued={realsense.frame_queue.qsize()}")
                if audio is not None:
                    status.append(f"audio blocks={audio.blocks_captured} queued={audio.audio_queue.qsize()}")
                if optitrack is not None:
                    status.append(f"opti rows={optitrack.rows_submitted} queued={optitrack.queue.qsize()}")
                print(f"[Main] t={elapsed_s:.1f}s " + " | ".join(status))
            time.sleep(0.1)
    finally:
        if clock.is_started:
            stop_requested_host_time_s = clock.now_s()
        print("[Main] Stopping recorders and flushing queues...")
        for recorder in reversed(recorders):
            try:
                recorder.stop()
            except Exception as exc:
                print(f"[Main] WARNING: error while stopping {recorder.__class__.__name__}: {exc}")

        stopped_at_utc = datetime.now(timezone.utc).isoformat()
        stats = {}
        if realsense is not None:
            stats["realsense"] = realsense.stats()
        if audio is not None:
            stats["audio"] = audio.stats()
        if optitrack is not None:
            stats["optitrack"] = optitrack.stats()
        stats["stop_requested_host_time_s"] = stop_requested_host_time_s
        stats["shutdown_complete_host_time_s"] = clock.now_s() if clock.is_started else 0.0
        write_metadata(out_dir, config, clock, started_at_utc, stopped_at_utc, stats)
        print(f"[Main] Done. Metadata and data saved in {out_dir}")


if __name__ == "__main__":
    main()
