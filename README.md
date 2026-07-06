# Multimodal Assisted Feeding Recorder

Python recorder for human-to-human assisted feeding interaction datasets. It records:

- Intel RealSense D415 RGB frames as `rgb/000001.jpg`
- Intel RealSense D415 depth frames aligned to color as 16-bit `depth/000001.png`
- Mono microphone audio as `audio.wav`
- Audio block timestamps as `audio_blocks.csv`
- RealSense frame timestamps as `realsense_timestamps.csv`
- OptiTrack rigid body rows as `optitrack.csv`
- Trial settings and summary stats as `metadata.json`

All streams use one host-side recorder clock based on `time.perf_counter_ns()`. Each saved data row includes `host_time_s`, which is seconds elapsed since recorder start. Device timestamps are saved when available.

This is software synchronization only. It does not implement hardware trigger sync. For validation, perform a clear sync event at trial start, such as a clap or a fast movement of a marker rigid body while making a sound.

## Folder Layout

```text
trial_001/
  rgb/
    000001.jpg
  depth/
    000001.png
  audio.wav
  realsense_timestamps.csv
  audio_blocks.csv
  optitrack.csv
  metadata.json
```

## Install

Use the dedicated conda environment on Windows:

```powershell
conda activate data_collection
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the environment does not exist yet:

```powershell
conda create -n data_collection --override-channels -c conda-forge python=3.13 -y
conda activate data_collection
pip install -r requirements.txt
```

Notes:

- Install Intel RealSense SDK 2.0 if `pyrealsense2` cannot access the camera.
- `sounddevice` uses PortAudio. If no microphone is found, check Windows privacy settings and default input device.
- The OptiTrack/NatNet Python SDK is not listed in `requirements.txt` because versions and sample APIs vary by Motive/NatNet release.

## Run

Record RealSense + audio + OptiTrack CSV adapter for 60 seconds:

```powershell
python main.py --out_dir trial_001 --duration 60
```

Record until Ctrl+C:

```powershell
python main.py --out_dir trial_001
```

Disable a stream:

```powershell
python main.py --out_dir trial_001 --duration 60 --no_optitrack
python main.py --out_dir trial_001 --duration 60 --no_realsense
python main.py --out_dir trial_001 --duration 60 --no_audio
```

Select a sounddevice input device:

```powershell
python -m sounddevice
python main.py --out_dir trial_001 --duration 60 --audio_device 1
```

## Timestamp Files

`realsense_timestamps.csv` contains one row per RGB-D pair:

- `frame_index`
- `host_time_s`
- `rgb_file`
- `depth_file`
- `color_timestamp_ms`
- `depth_timestamp_ms`
- `color_frame_number`
- `depth_frame_number`

`audio_blocks.csv` contains one row per audio callback block:

- `block_index`
- `host_time_s`
- `adc_time_s`
- `frames`
- `first_sample_index`
- `status`

`optitrack.csv` contains one row per rigid body per NatNet/Motive frame:

- `host_time_s`
- `natnet_frame_number`
- `rigid_body_id`
- `rigid_body_name`
- `x`, `y`, `z`
- `qx`, `qy`, `qz`, `qw`
- `tracking_valid`

## OptiTrack / NatNet Integration

Motive should do camera calibration, marker recognition, and rigid body reconstruction. Python only receives the NatNet stream from Motive.

The recorder provides `OptiTrackNatNetAdapter` in `recorder/optitrack_recorder.py`. Start the adapter with the main recorder, then call one of these methods from the official NatNet Python sample callback:

```python
optitrack.submit_rigid_body_frame(
    rigid_bodies=rigid_body_list,
    frame_number=frame_number,
)
```

or for one body:

```python
optitrack.record_rigid_body(
    rigid_body_id=rigid_body_id,
    rigid_body_name=rigid_body_name,
    position=(x, y, z),
    orientation=(qx, qy, qz, qw),
    tracking_valid=tracking_valid,
    frame_number=frame_number,
)
```

Typical NatNet sample connection pattern:

```python
# Inside the official NatNet sample, after creating the adapter:
from recorder.optitrack_recorder import OptiTrackNatNetAdapter

def receive_new_frame(data_frame):
    frame_number = getattr(data_frame, "i_frame", None)
    rigid_body_data = getattr(data_frame, "rigid_body_data", None)
    rigid_bodies = getattr(rigid_body_data, "rigid_body_list", [])
    optitrack.submit_rigid_body_frame(rigid_bodies, frame_number=frame_number)

natnet_client.new_frame_listener = receive_new_frame
```

If your NatNet SDK gives a different object layout, extract the rigid body ID/name, position, quaternion, validity, and frame number in the callback, then call `record_rigid_body()`.

The adapter accepts both dict-like rigid bodies and many NatNet sample object layouts. It checks common fields such as:

- `id`, `rigid_body_id`
- `name`
- `position`, `pos`
- `orientation`, `rotation`, `rot`
- `tracking_valid`, `valid`, `trackingValid`

## Design Notes

- `time.perf_counter_ns()` is the only recorder clock.
- RealSense capture and image writing run in separate threads.
- RealSense opens before the host clock starts and discards warmup frames so the first saved frame is close to `host_time_s = 0`.
- Audio capture uses a `sounddevice` callback. The callback only copies data into a queue.
- Audio WAV and timestamp CSV are written by a writer thread.
- OptiTrack writes through a queue-backed CSV writer.
- Queues are bounded. If a writer cannot keep up, the recorder prints warnings and counts dropped frames/rows in `metadata.json`.

## Files

```text
main.py
recorder/
  main.py
  clock.py
  config.py
  realsense_recorder.py
  audio_recorder.py
  optitrack_recorder.py
  writers.py
requirements.txt
README.md
```
