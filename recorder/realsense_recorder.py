from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np

from .clock import RecorderClock
from .config import RealSenseConfig
from .writers import RealSenseFrameRecord, RealSenseImageWriter, SENTINEL


class RealSenseRecorder:
    def __init__(self, out_dir: Path, clock: RecorderClock, config: RealSenseConfig) -> None:
        self.out_dir = out_dir
        self.clock = clock
        self.config = config
        self.frame_queue: queue.Queue = queue.Queue(maxsize=config.queue_size)
        self.writer = RealSenseImageWriter(out_dir, self.frame_queue, config.jpeg_quality)
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, name="RealSenseCapture", daemon=True)
        self.frames_captured = 0
        self.frames_dropped = 0
        self._pipeline = None
        self._startup_error: BaseException | None = None

    def start(self) -> None:
        self.writer.start()
        self._thread.start()
        if not self._ready_event.wait(timeout=15):
            self.stop()
            raise RuntimeError("Timed out waiting for RealSense pipeline to start.")
        if self._startup_error is not None:
            self.stop()
            raise RuntimeError("RealSense startup failed.") from self._startup_error

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10)
        self.frame_queue.put(SENTINEL)
        self.frame_queue.join()
        self.writer.join(timeout=10)

    def _capture_loop(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            self._startup_error = exc
            self._ready_event.set()
            return

        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.bgr8,
            self.config.fps,
        )
        rs_config.enable_stream(
            rs.stream.depth,
            self.config.width,
            self.config.height,
            rs.format.z16,
            self.config.fps,
        )
        align = rs.align(rs.stream.color)
        self._pipeline = pipeline

        try:
            profile = pipeline.start(rs_config)
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            for _ in range(max(0, self.config.warmup_frames)):
                if self._stop_event.is_set():
                    break
                pipeline.wait_for_frames(timeout_ms=1000)
            print(
                f"[RealSense] Started {self.config.width}x{self.config.height}@{self.config.fps}, "
                f"depth_scale={depth_scale}, warmup_frames={self.config.warmup_frames}"
            )
            self._ready_event.set()
        except Exception as exc:
            self._startup_error = exc
            self._ready_event.set()
            return

        try:
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                if not self.clock.is_started:
                    continue
                host_time_s = self.clock.now_s()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data()).copy()
                depth_image = np.asanyarray(depth_frame.get_data()).copy()

                self.frames_captured += 1
                record = RealSenseFrameRecord(
                    index=self.frames_captured,
                    host_time_s=host_time_s,
                    color_timestamp_ms=float(color_frame.get_timestamp()),
                    depth_timestamp_ms=float(depth_frame.get_timestamp()),
                    color_frame_number=int(color_frame.get_frame_number()),
                    depth_frame_number=int(depth_frame.get_frame_number()),
                    color_image=color_image,
                    depth_image=depth_image,
                )
                try:
                    self.frame_queue.put(record, timeout=0.25)
                except queue.Full:
                    self.frames_dropped += 1
                    if self.frames_dropped % 10 == 1:
                        print(f"[RealSense] WARNING: frame queue full, dropped {self.frames_dropped} frames")
        except Exception as exc:
            if not self._stop_event.is_set():
                print(f"[RealSense] ERROR: {exc}")
                self._stop_event.set()
        finally:
            pipeline.stop()
            print("[RealSense] Stopped")

    def stats(self) -> dict:
        return {
            "frames_captured": self.frames_captured,
            "frames_written": self.writer.frames_written,
            "frames_dropped": self.frames_dropped,
            "write_errors": self.writer.write_errors,
        }
