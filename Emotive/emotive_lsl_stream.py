# emotiv_lsl_stream.py
# Publish Emotiv EEG via LabStreamingLayer (LSL).
# - Uses your EmotivReader from emotiv_acquire.py
# - Publishes ONLY EEG channels (no counters/extras) with proper labels & fs
# - Timestamps are mapped to LSL clock for sub-ms sync with marker stream

import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock
from emotiv_acquire import EmotivReader, StreamOptions

def main():
    # ---- Configure your reader (adjust backend if needed)
    opts = StreamOptions(
        py_backend="pyusb",       # or "pywinusb" / "bluetooth" on Windows
        verbose=True,
        include_extras=True,      # ok; we'll strip before publishing
        include_counter=True,     # ok; we'll strip before publishing
        block_size=32,
        csv_path=None,            # LabRecorder handles recording; skip CSV here
        heartbeat_sec=5.0,
    )
    reader = EmotivReader(opts).start()

    # Probe device to learn fs, labels, model, etc.
    info = reader.device_info(probe_seconds=2.0)
    fs = float(f"{info.sampling_hz:.2f}")
    labels = list(info.labels)  # your EEG channel names (e.g., AF3, F7, ...)
    n_ch = len(labels)

    print(f"[EEG] Emotiv family={info.family}, fs≈{fs} Hz, n_ch={n_ch}, labels={labels}")

    # Create LSL EEG outlet
    si = StreamInfo(name="EmotivEEG",
                    type="EEG",
                    channel_count=n_ch,
                    nominal_srate=fs,
                    channel_format="float32",
                    source_id=f"emotiv_{info.family}_{n_ch}ch")

    # Add per-channel labels to LSL meta (nice for downstream tools)
    chns = si.desc().append_child("channels")
    for lab in labels:
        ch = chns.append_child("channel")
        ch.append_child_value("label", lab)
        ch.append_child_value("type", "EEG")
        ch.append_child_value("unit", "uV")   # Emotiv is microvolts-scale after scaling

    si.desc().append_child_value("manufacturer", "Emotiv")
    si.desc().append_child_value("model", info.family)
    si.desc().append_child_value("version", "LSL-1.0")

    outlet = StreamOutlet(si, chunk_size=0, max_buffered=360)

    # Map Python time.time() → LSL local_clock()
    # (LSL accepts explicit timestamps in its own clock)
    time_offset = local_clock() - time.time()
    print(f"[EEG] LSL time offset (lsl - time.time) ≈ {time_offset:.6f}s")

    try:
        for ts_block, x_block in reader.stream():
            # x_block shape: (B, cols). Strip extras/counter → EEG only
            # We assume EEG columns come FIRST in emotiv_acquire; if not, adapt the slice.
            eeg_only = x_block[:, :n_ch].astype(np.float32)

            # Push each sample with its original timestamp mapped into LSL clock
            for sample, ts in zip(eeg_only, ts_block):
                outlet.push_sample(sample.tolist(), ts + time_offset)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        print("[EEG] Stopped LSL outlet.")

if __name__ == "__main__":
    main()
