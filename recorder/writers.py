from __future__ import annotations

import csv
import queue
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SENTINEL = object()


def ensure_trial_dirs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rgb").mkdir(exist_ok=True)
    (out_dir / "depth").mkdir(exist_ok=True)


@dataclass
class RealSenseFrameRecord:
    index: int
    host_time_s: float
    color_timestamp_ms: float
    depth_timestamp_ms: float
    color_frame_number: int
    depth_frame_number: int
    color_image: np.ndarray
    depth_image: np.ndarray


class RealSenseImageWriter(threading.Thread):
    def __init__(
        self,
        out_dir: Path,
        frame_queue: queue.Queue,
        jpeg_quality: int = 95,
    ) -> None:
        super().__init__(name="RealSenseImageWriter", daemon=True)
        self.out_dir = out_dir
        self.frame_queue = frame_queue
        self.jpeg_quality = int(jpeg_quality)
        self.frames_written = 0
        self.write_errors = 0
        self._csv_file: Any | None = None
        self._csv_writer: csv.DictWriter | None = None

    def run(self) -> None:
        csv_path = self.out_dir / "realsense_timestamps.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "frame_index",
                "host_time_s",
                "rgb_file",
                "depth_file",
                "color_timestamp_ms",
                "depth_timestamp_ms",
                "color_frame_number",
                "depth_frame_number",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                item = self.frame_queue.get()
                try:
                    if item is SENTINEL:
                        return
                    self._write_record(item, writer)
                finally:
                    self.frame_queue.task_done()

    def _write_record(self, record: RealSenseFrameRecord, writer: csv.DictWriter) -> None:
        import cv2

        rgb_rel = f"rgb/{record.index:06d}.jpg"
        depth_rel = f"depth/{record.index:06d}.png"
        rgb_path = self.out_dir / rgb_rel
        depth_path = self.out_dir / depth_rel

        ok_rgb = cv2.imwrite(
            str(rgb_path),
            record.color_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        ok_depth = cv2.imwrite(str(depth_path), record.depth_image)
        if not ok_rgb or not ok_depth:
            self.write_errors += 1
            print(f"[RealSense] WARNING: failed to write frame {record.index}")
            return

        writer.writerow(
            {
                "frame_index": record.index,
                "host_time_s": f"{record.host_time_s:.9f}",
                "rgb_file": rgb_rel,
                "depth_file": depth_rel,
                "color_timestamp_ms": f"{record.color_timestamp_ms:.6f}",
                "depth_timestamp_ms": f"{record.depth_timestamp_ms:.6f}",
                "color_frame_number": record.color_frame_number,
                "depth_frame_number": record.depth_frame_number,
            }
        )
        self.frames_written += 1


@dataclass
class AudioBlockRecord:
    block_index: int
    host_time_s: float
    adc_time_s: float | None
    frames: int
    data: np.ndarray
    status: str


class AudioWriter(threading.Thread):
    def __init__(
        self,
        out_dir: Path,
        audio_queue: queue.Queue,
        samplerate: int,
        channels: int,
    ) -> None:
        super().__init__(name="AudioWriter", daemon=True)
        self.out_dir = out_dir
        self.audio_queue = audio_queue
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.blocks_written = 0
        self.frames_written = 0

    def run(self) -> None:
        wav_path = self.out_dir / "audio.wav"
        csv_path = self.out_dir / "audio_blocks.csv"

        with wave.open(str(wav_path), "wb") as wav, csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)
            wav.setframerate(self.samplerate)
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "block_index",
                    "host_time_s",
                    "adc_time_s",
                    "frames",
                    "first_sample_index",
                    "status",
                ],
            )
            writer.writeheader()

            while True:
                item = self.audio_queue.get()
                try:
                    if item is SENTINEL:
                        return
                    self._write_block(item, wav, writer)
                finally:
                    self.audio_queue.task_done()

    def _write_block(
        self,
        record: AudioBlockRecord,
        wav: wave.Wave_write,
        writer: csv.DictWriter,
    ) -> None:
        audio = np.asarray(record.data)
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767.0).astype(np.int16, copy=False)
        wav.writeframes(pcm16.tobytes())

        writer.writerow(
            {
                "block_index": record.block_index,
                "host_time_s": f"{record.host_time_s:.9f}",
                "adc_time_s": "" if record.adc_time_s is None else f"{record.adc_time_s:.9f}",
                "frames": record.frames,
                "first_sample_index": self.frames_written,
                "status": record.status,
            }
        )
        self.blocks_written += 1
        self.frames_written += record.frames


@dataclass
class OptiTrackRigidBodyRecord:
    host_time_s: float
    frame_number: int | None
    rigid_body_id: int | str | None
    rigid_body_name: str | None
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    tracking_valid: bool | None


class OptiTrackCsvWriter(threading.Thread):
    def __init__(self, out_dir: Path, optitrack_queue: queue.Queue) -> None:
        super().__init__(name="OptiTrackCsvWriter", daemon=True)
        self.out_dir = out_dir
        self.optitrack_queue = optitrack_queue
        self.rows_written = 0

    def run(self) -> None:
        csv_path = self.out_dir / "optitrack.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "host_time_s",
                "natnet_frame_number",
                "rigid_body_id",
                "rigid_body_name",
                "x",
                "y",
                "z",
                "qx",
                "qy",
                "qz",
                "qw",
                "tracking_valid",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                item = self.optitrack_queue.get()
                try:
                    if item is SENTINEL:
                        return
                    self._write_record(item, writer)
                finally:
                    self.optitrack_queue.task_done()

    def _write_record(
        self,
        record: OptiTrackRigidBodyRecord,
        writer: csv.DictWriter,
    ) -> None:
        writer.writerow(
            {
                "host_time_s": f"{record.host_time_s:.9f}",
                "natnet_frame_number": "" if record.frame_number is None else record.frame_number,
                "rigid_body_id": "" if record.rigid_body_id is None else record.rigid_body_id,
                "rigid_body_name": "" if record.rigid_body_name is None else record.rigid_body_name,
                "x": f"{record.x:.9f}",
                "y": f"{record.y:.9f}",
                "z": f"{record.z:.9f}",
                "qx": f"{record.qx:.9f}",
                "qy": f"{record.qy:.9f}",
                "qz": f"{record.qz:.9f}",
                "qw": f"{record.qw:.9f}",
                "tracking_valid": "" if record.tracking_valid is None else int(record.tracking_valid),
            }
        )
        self.rows_written += 1
