from __future__ import annotations

import queue
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .clock import RecorderClock
from .config import OptiTrackConfig
from .writers import OptiTrackCsvWriter, OptiTrackRigidBodyRecord, SENTINEL


class OptiTrackNatNetAdapter:
    """Small adapter between NatNet callbacks and the recorder CSV writer.

    The official NatNet Python samples differ a little across SDK versions.
    Keep their client setup in the sample script if you like, and call
    submit_rigid_body_frame() from the NatNet frame callback.
    """

    def __init__(self, out_dir: Path, clock: RecorderClock, config: OptiTrackConfig) -> None:
        self.out_dir = out_dir
        self.clock = clock
        self.config = config
        self.queue: queue.Queue = queue.Queue(maxsize=config.queue_size)
        self.writer = OptiTrackCsvWriter(out_dir, self.queue)
        self.rows_submitted = 0
        self.rows_dropped = 0

    def start(self) -> None:
        self.writer.start()
        print("[OptiTrack] CSV adapter started. Connect NatNet callback to submit_rigid_body_frame().")

    def stop(self) -> None:
        self.queue.put(SENTINEL)
        self.queue.join()
        self.writer.join(timeout=10)
        print("[OptiTrack] Stopped")

    def submit_rigid_body_frame(
        self,
        rigid_bodies: Iterable[object],
        frame_number: int | None = None,
        host_time_s: float | None = None,
    ) -> None:
        """Submit one Motive/NatNet frame containing one or more rigid bodies.

        rigid_bodies may contain dict-like objects or NatNet sample objects.
        Supported fields/attributes:
        id / rigid_body_id, name, position / pos, rotation / orientation / rot,
        tracking_valid / valid / trackingValid.
        """
        if host_time_s is None and not self.clock.is_started:
            return

        timestamp = self.clock.now_s() if host_time_s is None else float(host_time_s)
        for body in rigid_bodies:
            record = self._coerce_rigid_body(body, frame_number, timestamp)
            try:
                self.queue.put_nowait(record)
                self.rows_submitted += 1
            except queue.Full:
                self.rows_dropped += 1
                if self.rows_dropped % 100 == 1:
                    print(f"[OptiTrack] WARNING: queue full, dropped {self.rows_dropped} rows")

    def record_rigid_body(
        self,
        rigid_body_id: int | str | None,
        position: Sequence[float],
        orientation: Sequence[float],
        frame_number: int | None = None,
        rigid_body_name: str | None = None,
        tracking_valid: bool | None = None,
        host_time_s: float | None = None,
    ) -> None:
        body = {
            "id": rigid_body_id,
            "name": rigid_body_name,
            "position": position,
            "orientation": orientation,
            "tracking_valid": tracking_valid,
        }
        self.submit_rigid_body_frame([body], frame_number=frame_number, host_time_s=host_time_s)

    def example_natnet_callback(self, data_frame) -> None:
        """Example callback for official NatNet sample integration.

        In many SDK versions, data_frame has i_frame and rigid_body_data.rigid_body_list.
        If your sample uses different names, extract frame_number and rigid_bodies there,
        then call submit_rigid_body_frame(rigid_bodies, frame_number).
        """

        frame_number = self._get(data_frame, "i_frame", "frame_number", "frameNumber")
        rigid_body_data = self._get(data_frame, "rigid_body_data", "rigidBodyData")
        rigid_bodies = self._get(rigid_body_data, "rigid_body_list", "rigidBodyList", default=[])
        self.submit_rigid_body_frame(rigid_bodies, frame_number=frame_number)

    def _coerce_rigid_body(
        self,
        body: object,
        frame_number: int | None,
        host_time_s: float,
    ) -> OptiTrackRigidBodyRecord:
        rigid_body_id = self._get(body, "id", "rigid_body_id", "rigidBodyId", "ID")
        name = self._get(body, "name", "rigid_body_name", default=None)
        position = self._get(body, "position", "pos", default=(None, None, None))
        orientation = self._get(body, "orientation", "rotation", "rot", default=(None, None, None, None))
        valid = self._get(body, "tracking_valid", "valid", "trackingValid", "tracking_valid_flag", default=None)

        x, y, z = self._coerce_xyz(position, body)
        qx, qy, qz, qw = self._coerce_quat(orientation, body)

        return OptiTrackRigidBodyRecord(
            host_time_s=host_time_s,
            frame_number=frame_number,
            rigid_body_id=rigid_body_id,
            rigid_body_name=name,
            x=x,
            y=y,
            z=z,
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
            tracking_valid=None if valid is None else bool(valid),
        )

    @staticmethod
    def _get(obj: object, *names: str, default=None):
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            for name in names:
                if name in obj:
                    return obj[name]
            return default
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    def _coerce_xyz(self, position: object, body: object) -> tuple[float, float, float]:
        if not self._is_missing_sequence(position):
            if isinstance(position, Mapping):
                return (
                    float(position.get("x", 0.0)),
                    float(position.get("y", 0.0)),
                    float(position.get("z", 0.0)),
                )
            return float(position[0]), float(position[1]), float(position[2])
        return (
            float(self._get(body, "x", default=0.0)),
            float(self._get(body, "y", default=0.0)),
            float(self._get(body, "z", default=0.0)),
        )

    def _coerce_quat(self, orientation: object, body: object) -> tuple[float, float, float, float]:
        if not self._is_missing_sequence(orientation):
            if isinstance(orientation, Mapping):
                return (
                    float(orientation.get("qx", orientation.get("x", 0.0))),
                    float(orientation.get("qy", orientation.get("y", 0.0))),
                    float(orientation.get("qz", orientation.get("z", 0.0))),
                    float(orientation.get("qw", orientation.get("w", 1.0))),
                )
            return (
                float(orientation[0]),
                float(orientation[1]),
                float(orientation[2]),
                float(orientation[3]),
            )
        return (
            float(self._get(body, "qx", default=0.0)),
            float(self._get(body, "qy", default=0.0)),
            float(self._get(body, "qz", default=0.0)),
            float(self._get(body, "qw", default=1.0)),
        )

    @staticmethod
    def _is_missing_sequence(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, Mapping):
            return False
        try:
            return all(item is None for item in value)
        except TypeError:
            return True

    def stats(self) -> dict:
        return {
            "rows_submitted": self.rows_submitted,
            "rows_written": self.writer.rows_written,
            "rows_dropped": self.rows_dropped,
        }
