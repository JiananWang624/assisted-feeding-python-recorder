from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np

from .clock import RecorderClock
from .config import AudioConfig
from .writers import AudioBlockRecord, AudioWriter, SENTINEL


class AudioRecorder:
    def __init__(self, out_dir: Path, clock: RecorderClock, config: AudioConfig) -> None:
        self.out_dir = out_dir
        self.clock = clock
        self.config = config
        self.audio_queue: queue.Queue = queue.Queue(maxsize=config.queue_size)
        self.writer = AudioWriter(out_dir, self.audio_queue, config.samplerate, config.channels)
        self._stream = None
        self._lock = threading.Lock()
        self.blocks_captured = 0
        self.blocks_dropped = 0
        self.callback_status_count = 0

    def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice is required for audio recording.") from exc

        self.writer.start()
        self._stream = sd.InputStream(
            samplerate=self.config.samplerate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.blocksize,
            device=self.config.device,
            callback=self._callback,
        )
        self._stream.start()
        print(
            f"[Audio] Started {self.config.channels}ch @ {self.config.samplerate} Hz, "
            f"blocksize={self.config.blocksize}, device={self.config.device}"
        )

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            print("[Audio] Stopped")
        self.audio_queue.put(SENTINEL)
        self.audio_queue.join()
        self.writer.join(timeout=10)

    def _callback(self, indata, frames, time_info, status) -> None:
        if not self.clock.is_started:
            return

        host_time_s = self.clock.now_s()
        adc_time = None
        if time_info is not None:
            adc_time = getattr(time_info, "inputBufferAdcTime", None)
            if adc_time is None and isinstance(time_info, dict):
                adc_time = time_info.get("input_buffer_adc_time")
                if adc_time is None:
                    adc_time = time_info.get("inputBufferAdcTime")

        with self._lock:
            self.blocks_captured += 1
            block_index = self.blocks_captured
            if status:
                self.callback_status_count += 1

        record = AudioBlockRecord(
            block_index=block_index,
            host_time_s=host_time_s,
            adc_time_s=None if adc_time is None else float(adc_time),
            frames=int(frames),
            data=np.asarray(indata).copy(),
            status=str(status) if status else "",
        )
        try:
            self.audio_queue.put_nowait(record)
        except queue.Full:
            with self._lock:
                self.blocks_dropped += 1

    def stats(self) -> dict:
        return {
            "blocks_captured": self.blocks_captured,
            "blocks_written": self.writer.blocks_written,
            "frames_written": self.writer.frames_written,
            "blocks_dropped": self.blocks_dropped,
            "callback_status_count": self.callback_status_count,
        }
