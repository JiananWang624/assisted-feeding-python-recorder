from dataclasses import asdict, dataclass, field


@dataclass
class RealSenseConfig:
    enabled: bool = True
    width: int = 640
    height: int = 480
    fps: int = 30
    rgb_format: str = "jpg"
    depth_format: str = "png"
    queue_size: int = 180
    jpeg_quality: int = 95
    warmup_frames: int = 15


@dataclass
class AudioConfig:
    enabled: bool = True
    samplerate: int = 48_000
    channels: int = 1
    dtype: str = "float32"
    blocksize: int = 1024
    queue_size: int = 512
    device: int | str | None = None


@dataclass
class OptiTrackConfig:
    enabled: bool = True
    queue_size: int = 4096
    adapter_mode: str = "raw_udp"
    natnet_path: str | None = None
    server_address: str = "127.0.0.1"
    client_address: str = "127.0.0.1"
    use_multicast: bool = True
    multicast_address: str = "239.255.42.99"
    data_port: int = 1511


@dataclass
class RecorderConfig:
    out_dir: str
    duration_s: float | None = None
    realsense: RealSenseConfig = field(default_factory=RealSenseConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    optitrack: OptiTrackConfig = field(default_factory=OptiTrackConfig)

    def to_dict(self) -> dict:
        return asdict(self)
