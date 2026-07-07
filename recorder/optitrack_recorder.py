from __future__ import annotations

import importlib.util
import queue
import socket
import struct
import sys
import threading
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
        self.frames_received = 0
        self._natnet_client = None
        self._latest_frame_number: int | None = None
        self._raw_socket: socket.socket | None = None
        self._raw_thread: threading.Thread | None = None
        self._raw_stop_event = threading.Event()
        self.raw_packets_received = 0
        self.raw_parse_errors = 0

    def start(self) -> None:
        self.writer.start()
        if self.config.natnet_path:
            self._start_natnet_if_configured()
        elif self.config.adapter_mode == "raw_udp":
            self._start_raw_udp_receiver()
        if self._natnet_client is None:
            print(
                "[OptiTrack] CSV adapter started. "
                "Using raw UDP receiver if configured, or pass --optitrack_natnet_path for official NatNetClient."
            )

    def stop(self) -> None:
        if self._natnet_client is not None:
            self._shutdown_natnet_client()
        self._stop_raw_udp_receiver()
        self.queue.put(SENTINEL)
        self.queue.join()
        self.writer.join(timeout=10)
        print("[OptiTrack] Stopped")

    def _start_raw_udp_receiver(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.5)
        sock.bind(("", int(self.config.data_port)))

        if self.config.use_multicast:
            mreq = socket.inet_aton(self.config.multicast_address) + socket.inet_aton(
                self.config.client_address
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        self._raw_socket = sock
        self._raw_stop_event.clear()
        self._raw_thread = threading.Thread(
            target=self._raw_udp_loop,
            name="OptiTrackRawUdpReceiver",
            daemon=True,
        )
        self._raw_thread.start()

        mode = "multicast" if self.config.use_multicast else "unicast"
        print(
            "[OptiTrack] Raw UDP receiver started: "
            f"client={self.config.client_address}, port={self.config.data_port}, mode={mode}"
        )

    def _stop_raw_udp_receiver(self) -> None:
        self._raw_stop_event.set()
        if self._raw_socket is not None:
            self._raw_socket.close()
            self._raw_socket = None
        if self._raw_thread is not None:
            self._raw_thread.join(timeout=2)
            self._raw_thread = None

    def _raw_udp_loop(self) -> None:
        while not self._raw_stop_event.is_set():
            try:
                if self._raw_socket is None:
                    return
                data, _addr = self._raw_socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return

            self.raw_packets_received += 1
            try:
                frame_number, rigid_bodies = self._parse_natnet_frame_packet(data)
            except Exception as exc:
                self.raw_parse_errors += 1
                if self.raw_parse_errors <= 5:
                    print(f"[OptiTrack] WARNING: failed to parse NatNet packet: {exc}")
                continue

            self._latest_frame_number = frame_number
            if rigid_bodies:
                self.frames_received += 1
                self.submit_rigid_body_frame(rigid_bodies, frame_number=frame_number)

    def _parse_natnet_frame_packet(self, data: bytes) -> tuple[int | None, list[dict]]:
        if len(data) < 12:
            return None, []
        message_id, _packet_size = struct.unpack_from("<HH", data, 0)
        if message_id != 7:
            return None, []

        offset = 4
        frame_number = self._read_i32(data, offset)
        offset += 4

        marker_set_count = self._read_i32(data, offset)
        offset += 4
        marker_set_section_len, offset = self._maybe_read_section_len(data, offset)
        marker_set_start = offset
        for _ in range(marker_set_count):
            _name, offset = self._read_c_string(data, offset)
            marker_count = self._read_i32(data, offset)
            offset += 4 + marker_count * 12
        if marker_set_section_len is not None:
            offset = max(offset, marker_set_start + marker_set_section_len)

        unlabeled_count = self._read_i32(data, offset)
        offset += 4
        unlabeled_section_len, offset = self._maybe_read_section_len(data, offset)
        if unlabeled_section_len is None:
            offset += unlabeled_count * 12
        else:
            offset += unlabeled_section_len

        rigid_body_count = self._read_i32(data, offset)
        offset += 4
        rigid_body_section_len, offset = self._maybe_read_section_len(data, offset)
        rigid_body_start = offset
        rigid_bodies: list[dict] = []
        for _ in range(rigid_body_count):
            if offset + 32 > len(data):
                break
            rigid_body_id = self._read_i32(data, offset)
            offset += 4
            x, y, z, qx, qy, qz, qw = struct.unpack_from("<fffffff", data, offset)
            offset += 28

            tracking_valid = None
            if offset + 6 <= len(data):
                # NatNet 2.6+ appends mean marker error and tracking params.
                _mean_marker_error = self._read_f32(data, offset)
                offset += 4
                params = self._read_i16(data, offset)
                offset += 2
                tracking_valid = bool(params & 0x01)

            rigid_bodies.append(
                {
                    "id": rigid_body_id,
                    "position": (x, y, z),
                    "orientation": (qx, qy, qz, qw),
                    "tracking_valid": tracking_valid,
                }
            )

        if rigid_body_section_len is not None:
            offset = max(offset, rigid_body_start + rigid_body_section_len)

        return frame_number, rigid_bodies

    @staticmethod
    def _read_i32(data: bytes, offset: int) -> int:
        return int(struct.unpack_from("<i", data, offset)[0])

    @staticmethod
    def _read_i16(data: bytes, offset: int) -> int:
        return int(struct.unpack_from("<h", data, offset)[0])

    @staticmethod
    def _read_f32(data: bytes, offset: int) -> float:
        return float(struct.unpack_from("<f", data, offset)[0])

    @staticmethod
    def _read_c_string(data: bytes, offset: int) -> tuple[str, int]:
        end = data.find(b"\0", offset)
        if end < 0:
            raise ValueError("missing null-terminated string")
        return data[offset:end].decode("utf-8", errors="replace"), end + 1

    @staticmethod
    def _maybe_read_section_len(data: bytes, offset: int) -> tuple[int | None, int]:
        if offset + 4 > len(data):
            return None, offset
        value = int(struct.unpack_from("<i", data, offset)[0])
        remaining = len(data) - (offset + 4)
        if 0 <= value <= remaining:
            return value, offset + 4
        return None, offset

    def _start_natnet_if_configured(self) -> None:
        if not self.config.natnet_path:
            return

        client_class = self._load_natnet_client_class(Path(self.config.natnet_path))
        client = client_class()
        self._natnet_client = client

        self._call_if_present(client, "set_client_address", self.config.client_address)
        self._call_if_present(client, "set_server_address", self.config.server_address)
        self._call_if_present(client, "set_use_multicast", bool(self.config.use_multicast))

        # Common official NatNet Python sample callbacks. Different SDK releases
        # use either full-frame callbacks, per-rigid-body callbacks, or both.
        client.new_frame_listener = self._natnet_new_frame_listener
        client.rigid_body_listener = self._natnet_rigid_body_listener

        started = client.run()
        if started is False:
            raise RuntimeError(
                "NatNetClient.run() returned False. Check Motive Streaming settings "
                "and server/client IP addresses."
            )

        mode = "multicast" if self.config.use_multicast else "unicast"
        print(
            "[OptiTrack] NatNet client started: "
            f"server={self.config.server_address}, client={self.config.client_address}, mode={mode}"
        )

    @staticmethod
    def _load_natnet_client_class(path: Path):
        if path.is_dir():
            natnet_file = path / "NatNetClient.py"
            search_path = path
        else:
            natnet_file = path
            search_path = path.parent

        if not natnet_file.exists():
            raise FileNotFoundError(
                f"Could not find NatNetClient.py at {natnet_file}. "
                "Use --optitrack_natnet_path with the official NatNet Python sample folder."
            )

        if str(search_path) not in sys.path:
            sys.path.insert(0, str(search_path))

        spec = importlib.util.spec_from_file_location("NatNetClient", natnet_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load NatNetClient from {natnet_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["NatNetClient"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "NatNetClient"):
            raise ImportError(f"{natnet_file} does not define a NatNetClient class")
        return module.NatNetClient

    @staticmethod
    def _call_if_present(obj: object, method_name: str, *args) -> None:
        method = getattr(obj, method_name, None)
        if callable(method):
            method(*args)

    def _shutdown_natnet_client(self) -> None:
        shutdown = getattr(self._natnet_client, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self._natnet_client = None

    def _natnet_new_frame_listener(self, data_frame) -> None:
        frame_number = self._get_frame_number(data_frame)
        self._latest_frame_number = frame_number
        rigid_bodies = self._extract_rigid_bodies_from_frame(data_frame)
        if rigid_bodies:
            self.submit_rigid_body_frame(rigid_bodies, frame_number=frame_number)
            self.frames_received += 1

    def _natnet_rigid_body_listener(self, rigid_body_id, position, rotation) -> None:
        self.record_rigid_body(
            rigid_body_id=rigid_body_id,
            position=position,
            orientation=rotation,
            frame_number=self._latest_frame_number,
        )

    def _get_frame_number(self, data_frame) -> int | None:
        if isinstance(data_frame, Mapping):
            value = self._get(
                data_frame,
                "frame_number",
                "i_frame",
                "frameNumber",
                "frame",
                default=None,
            )
        else:
            value = self._get(data_frame, "i_frame", "frame_number", "frameNumber", default=None)
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    def _extract_rigid_bodies_from_frame(self, data_frame) -> list[object]:
        if data_frame is None:
            return []

        if isinstance(data_frame, Mapping):
            for key in (
                "rigid_bodies",
                "rigid_body_list",
                "rigidBodies",
                "rigid_body_data",
                "rigidBodyData",
            ):
                value = data_frame.get(key)
                bodies = self._coerce_rigid_body_list(value)
                if bodies:
                    return bodies
            return []

        rigid_body_data = self._get(data_frame, "rigid_body_data", "rigidBodyData")
        bodies = self._coerce_rigid_body_list(rigid_body_data)
        if bodies:
            return bodies

        return self._coerce_rigid_body_list(
            self._get(data_frame, "rigid_body_list", "rigidBodyList", "rigid_bodies", default=[])
        )

    def _coerce_rigid_body_list(self, value: object) -> list[object]:
        if value is None:
            return []
        if isinstance(value, Mapping):
            for key in ("rigid_body_list", "rigidBodyList", "rigid_bodies", "rigidBodies"):
                if key in value:
                    return self._coerce_rigid_body_list(value[key])
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        for attr in ("rigid_body_list", "rigidBodyList", "rigid_bodies", "rigidBodies"):
            if hasattr(value, attr):
                return self._coerce_rigid_body_list(getattr(value, attr))
        return []

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
            "frames_received": self.frames_received,
            "natnet_client_active": self._natnet_client is not None,
            "natnet_path": self.config.natnet_path,
            "server_address": self.config.server_address,
            "client_address": self.config.client_address,
            "use_multicast": self.config.use_multicast,
            "multicast_address": self.config.multicast_address,
            "data_port": self.config.data_port,
            "raw_packets_received": self.raw_packets_received,
            "raw_parse_errors": self.raw_parse_errors,
        }
