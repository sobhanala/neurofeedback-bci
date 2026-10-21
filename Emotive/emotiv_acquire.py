# emotiv_acquire.py 
#python '.\emotiv_acquire.py' --backend pyusb --csv sessionid.csv --verbose
# ------------------------------------------------------------
# A friendly, "just works" acquisition layer for Emotiv headsets
# using your local eeg.py (CyKIT-based) without any game/UI stuff.
#
# What this does:
#   - Connects to the headset (USB/BLE depending on backend).
#   - Detects how many EEG channels the device is sending (14 or 5).
#   - Measures actual sampling rate from real arrival times.
#   - Streams small blocks of samples to your code.
#   - (Optionally) writes a clean CSV: timestamp, counter, EEG..., extras.
#   - Prints useful status lines: connected ✓, labels, fs, heartbeat.
#
# Run it:
#   python emotiv_acquire.py --backend pyusb --csv session.csv --verbose
# or quietly log to a file:
#   python emotiv_acquire.py --quiet --log runs/2025-08-16.log --csv session.csv
#
# Zero dependencies beyond numpy (and, of course, your eeg.py).
# ------------------------------------------------------------

from __future__ import annotations
import time
import threading
import queue
import csv
import os
import sys
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import eeg  # <-- your uploaded driver (same folder)

# Known Emotiv channel orders (taken from your eeg.py / common layouts)
EPOC_14   = ["F3","FC5","AF3","F7","T7","P7","O1","O2","P8","T8","F8","AF4","FC6","F4"]
INSIGHT_5 = ["AF3","T7","Pz","T8","AF4"]


# ----------------------------
# Options & device info structs
# ----------------------------
@dataclass
class StreamOptions:
    # backend for eeg.py: pick one that matches your setup
    py_backend: str = "pyusb"            # "pyusb" | "pywinusb" | "bluetooth"

    # talkative or not? (console prints)
    verbose: bool = True                 # True = print status to console

    # include trailing values after EEG channels (battery/quality/gyro/etc.)
    include_extras: bool = True

    # per-yield block size (in samples); 32 @256 Hz ≈ 125 ms blocks
    block_size: int = 32

    # CSV logging
    csv_path: Optional[str] = None       # pass a path to save CSV
    csv_title: str = "EEG_recording"
    csv_overwrite: bool = False          # if False, auto-index to avoid clobber

    # keep a periodic "we're alive" print
    heartbeat_sec: float = 5.0

    # status logging to file (in addition to or instead of console)
    log_path: Optional[str] = None
    log_append: bool = False

    # NEW: include the device sample counter in CSV (rolls 0..127 typically)
    include_counter: bool = True


@dataclass
class DeviceInfo:
    # super simple "family" hint based on channel count
    family: str                          # "EPOC_PLUS" | "EPOC" | "INSIGHT" | "UNKNOWN"
    sampling_hz: float                   # measured fs (we’ll estimate)
    labels: List[str]                    # EEG labels in order
    serial: Optional[str] = None
    product: Optional[str] = None
    backend: Optional[str] = None        # pyusb / pywinusb / bluetooth


# --------------
# The main reader
# --------------
class EmotivReader:
    """
    Wraps your eeg.py so you can just:
        reader = EmotivReader(StreamOptions(...)).start()
        info = reader.device_info()
        for ts, block in reader.stream(): ...
    """

    def __init__(self, opts: StreamOptions = StreamOptions()):
        self.opts = opts
        self._io = eeg.ControllerIO()

        # Build config flags for eeg.py.
        # We insist on: eegmode + noweb + outputdata
        #  - "eegmode" -> use EEG (not benchmark/demo)
        #  - "noweb"   -> don't spin up websockets; keep it local
        #  - "outputdata" -> ensures our callback gets actual float samples
        flags = ["eegmode", "noweb", "outputdata"]
        if opts.verbose:
            flags.append("verbose")
        if opts.py_backend == "pywinusb":
            flags.append("pywinusb")
        if opts.py_backend == "bluetooth":
            flags.append("bluetooth")
        self._config = "+".join(flags)

        # prime IO object with basic status
        self._io.setInfo("ioObject", self._io)
        self._io.setInfo("config", self._config)
        self._io.setInfo("noweb","True")
        self._io.setInfo("status","True")
        self._io.setInfo("verbose","True" if opts.verbose else "False")

        # our internal queues & state
        self._q: "queue.Queue[Tuple[float, List[float]]]" = queue.Queue(maxsize=4096)
        self._stop = threading.Event()
        self._started = False
        self._eeg = None

        self._devinfo: Optional[DeviceInfo] = None
        self._csv = None
        self._writer = None
        self._csv_cols: List[str] = []

        # quick heartbeat so you know it's alive
        self._sample_count = 0
        self._t0 = None
        self._last_beat = None

        # optional file logger
        self._log_fh = None
        if self.opts.log_path:
            mode = "a" if self.opts.log_append else "w"
            os.makedirs(os.path.dirname(self.opts.log_path) or ".", exist_ok=True)
            self._log_fh = open(self.opts.log_path, mode, encoding="utf-8")

    # -------------
    # tiny logger
    # -------------
    def _log(self, msg: str):
        if self._log_fh:
            self._log_fh.write(msg + "\n")
            self._log_fh.flush()
        if self.opts.verbose:
            print(msg)

    # ----------------
    # lifecycle: start
    # ----------------
    def start(self) -> "EmotivReader":
        if self._started:
            return self

        self._log(f"[Emotiv] Starting with backend='{self.opts.py_backend}', config='{self._config}'")

        # This function is called by eeg.py for every decoded EEG packet.
        def _sink(state_list: List[float]):
            # defensive: make sure it looks like a real packet
            if not state_list or len(state_list) < 4:
                return

            # The second field is often "type"; 16 = EEG, 32 = gyro, etc.
            # If it's missing or weird, assume EEG (16) to be forgiving.
            pkt_type = int(state_list[1]) if isinstance(state_list[1], (int, float)) else 16
            if pkt_type != 16:
                return  # ignore non-EEG packets

            ts = time.time()

            # First time we see a packet? figure out who we are dealing with.
            if self._devinfo is None:
                eeg_len = self._infer_eeg_len(state_list)
                # Grab a friendly label set. If it's something odd, just name them ch1..chN.
                if eeg_len == 14:
                    labels = EPOC_14
                    fam = "EPOC_PLUS"  # good enough hint for 14-ch devices
                elif eeg_len == 5:
                    labels = INSIGHT_5
                    fam = "INSIGHT"
                else:
                    labels = [f"ch{i+1}" for i in range(eeg_len)]
                    fam = "UNKNOWN"

                product = self._io.getInfo("device") or None
                serial  = self._io.getInfo("serial") or None
                self._devinfo = DeviceInfo(
                    family=fam, sampling_hz=0.0, labels=labels,
                    serial=serial, product=product, backend=self.opts.py_backend
                )
                self._log(f"[Emotiv] Headset connected ✓  Product='{product}'  Serial='{serial}'  Backend='{self.opts.py_backend}'")
                self._log(f"[Emotiv] Detected family={fam}  EEG channels={len(labels)}  Labels={labels}")

                # Open CSV once we know the labels
                if self.opts.csv_path:
                    self._open_csv()

                # spin up heartbeat clock
                self._t0 = ts
                self._last_beat = ts

            # Build a row for the queue:
            #   [counter?] + EEG + (extras if asked)
            eeg_len = len(self._devinfo.labels)
            eeg_vals = state_list[2:2+eeg_len]

            row: List[float] = []
            if self.opts.include_counter:
                # the very first field is the device counter (usually rolls 0..127)
                try:
                    row.append(int(state_list[0]))
                except Exception:
                    row.append(state_list[0])  # if it's weird, keep it raw

            row += eeg_vals

            if self.opts.include_extras and len(state_list) > 2 + eeg_len:
                extras = state_list[2 + eeg_len :]
                row += extras

            try:
                self._q.put_nowait((ts, row))
                self._sample_count += 1
            except queue.Full:
                # if consumer is too slow, drop the sample; better than blocking
                pass

        # IMPORTANT:
        #   Your eeg.py typically fires our callback in the "EPOC+ float" path when
        #   "outputdata" is in the config. We already did that above.
        #
        #   If your eeg.py is still hardcoded internally to model=6 (Epoc+ consumer),
        #   that's fine; this still works. If you restore multi-model support later,
        #   this wrapper will keep playing nice.
        self._eeg = eeg.EEG(model=6, io=self._io, config=self._config, passing_variable=_sink)
        self._eeg.start()
        self._started = True

        self._log("[Emotiv] Waiting for first packets…")
        return self

    # ---------------
    # lifecycle: stop
    # ---------------
    def stop(self):
        self._stop.set()
        self._io.setInfo("status","False")
        # Close CSV
        if self._csv:
            try:
                self._csv.flush()
                self._csv.close()
            except Exception:
                pass
        # Close log
        if self._log_fh:
            try:
                self._log_fh.flush()
                self._log_fh.close()
            except Exception:
                pass
        self._log("[Emotiv] Stopped.")

    # ----------------------
    # device info + fs probe
    # ----------------------
    def device_info(self, probe_seconds: float = 2.0) -> DeviceInfo:
        # If we haven't seen any packet yet, wait a bit.
        if self._devinfo is None:
            self.read(timeout=5.0)

        # Collect some timestamps and estimate fs.
        timestamps: List[float] = []
        t_end = time.time() + probe_seconds
        while time.time() < t_end:
            try:
                ts, _ = self.read(timeout=probe_seconds)
                timestamps.append(ts)
            except queue.Empty:
                break

        sr = _estimate_rate(timestamps)
        declared = self._declared_sampling()
        self._devinfo.sampling_hz = sr if sr > 0 else declared

        self._log(f"[Emotiv] Sampling rate ≈ {self._devinfo.sampling_hz:.2f} Hz (declared: {declared:g} Hz)")

        # append a sampling_measured line into CSV (safe and atomic)
        self._write_csv_sampling_measured(self._devinfo.sampling_hz)
        return self._devinfo

    # -------------------------
    # single sample (blocking)
    # -------------------------
    def read(self, timeout: Optional[float] = None) -> Tuple[float, List[float]]:
        ts, row = self._q.get(timeout=timeout)

        # heartbeat so you know it's alive even if you don't print per-block
        if self._last_beat and (ts - self._last_beat) >= max(0.5, self.opts.heartbeat_sec):
            elapsed = ts - self._t0 if self._t0 else 0.0
            rate = self._sample_count / elapsed if elapsed > 0 else 0.0
            self._log(f"[Emotiv] Streaming… samples={self._sample_count} elapsed={elapsed:.1f}s rate≈{rate:.1f} Sa/s")
            self._last_beat = ts

        return ts, row

    # -----------------------
    # main generator (blocks)
    # -----------------------
    def stream(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        ts_buf: List[float] = []
        x_buf: List[List[float]] = []
        while not self._stop.is_set():
            try:
                ts, row = self.read(timeout=0.5)
            except queue.Empty:
                continue

            ts_buf.append(ts)
            x_buf.append(row)

            if len(x_buf) >= self.opts.block_size:
                ts_arr = np.asarray(ts_buf, dtype=np.float64)
                x_arr  = np.asarray(x_buf, dtype=np.float64)
                self._write_csv_rows(ts_arr, x_arr)
                yield ts_arr, x_arr
                ts_buf.clear()
                x_buf.clear()

    # --------------
    # internals
    # --------------
    def _infer_eeg_len(self, state_list: List[float]) -> int:
        # Quick and dirty (but effective):
        #   state_list ≈ [counter, type, EEG..., extras?]
        n = len(state_list)
        if n >= 16: return 14  # [counter, type] + 14 EEG = 16+
        if n >= 7:  return 5   # [counter, type] + 5 EEG = 7+
        return max(0, n - 2)

    def _declared_sampling(self) -> float:
        s = self._io.getInfo("sampling")
        try:
            return float(s)
        except Exception:
            return 0.0

    # --------------
    # CSV handling
    # --------------
    def _open_csv(self):
        path = self.opts.csv_path
        if (not self.opts.csv_overwrite) and path and os.path.exists(path):
            base, ext = os.path.splitext(path)
            i = 1
            while os.path.exists(f"{base}_{i}{ext or '.csv'}"):
                i += 1
            path = f"{base}_{i}{ext or '.csv'}"

        self._csv = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv)

        dev = self._io.getInfo("device") or ""
        ser = self._io.getInfo("serial") or ""
        fam = self._devinfo.family if self._devinfo else ""
        backend = self.opts.py_backend

        # Row 1: a quick who/what/when so the file is self-explanatory
        self._writer.writerow([
            f"title:{self.opts.csv_title}",
            f"recorded:{time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"device:{dev}",
            f"serial:{ser}",
            f"family:{fam}",
            f"backend:{backend}",
        ])

        # Row 2: EEG labels only (no timestamp/counter)
        labels = list(self._devinfo.labels)
        self._writer.writerow(["labels:"] + labels)

        # Row 3: sampling (as declared by the device)
        self._writer.writerow([f"sampling:{self._declared_sampling()}"])

        # Row 4: a tiny "column dictionary" so future-you knows what’s what
        desc = [
            "columns:",
            "timestamp (seconds, UNIX epoch, float)",
        ]
        if self.opts.include_counter:
            desc.append("counter (device sample counter, uint8, wraps modulo 128)")
        desc.append("EEG channels in order above (Emotiv EDK-scaled units)")
        self._writer.writerow(desc)

        # Row 5: the actual CSV columns we'll use for data
        self._csv_cols = ["timestamp"]
        if self.opts.include_counter:
            self._csv_cols.append("counter")
        self._csv_cols += labels
        self._writer.writerow(self._csv_cols)
        self._csv.flush()

        self._log(f"[Emotiv] CSV opened at '{path}'")
        self._log(f"[Emotiv] CSV columns = {self._csv_cols}")

    def _write_csv_sampling_measured(self, sr: float):
        # Keep it simple/safe: just append a new meta line with the measured fs.
        if not self._writer or not np.isfinite(sr) or sr <= 0:
            return
        self._writer.writerow([f"sampling_measured:{sr}"])
        if self._csv:
            self._csv.flush()

    def _write_csv_rows(self, ts: np.ndarray, x: np.ndarray):
        if not self._writer:
            return

        # If extras show up later at runtime, we extend headers once.
        # (We'll add EXTRA_1..N names, print a little line marking it,
        #  then repeat the full header so spreadsheet tools don't freak out.)
        want_cols = 1 + x.shape[1]  # timestamp + row length
        if want_cols > len(self._csv_cols):
            n_new = want_cols - len(self._csv_cols)
            extras = [f"EXTRA_{i+1}" for i in range(n_new)]
            self._csv_cols += extras
            self._writer.writerow(["extended_columns:"] + extras)
            self._writer.writerow(self._csv_cols)

        for t, row in zip(ts, x):
            self._writer.writerow([f"{t:.6f}"] + list(np.asarray(row, dtype=object)))

        if self._csv:
            self._csv.flush()


# --------------------------
# small helper: estimate fs
# --------------------------
def _estimate_rate(timestamps: List[float]) -> float:
    if len(timestamps) < 5:
        return 0.0
    diffs = np.diff(np.asarray(timestamps))
    diffs = diffs[(diffs > 0) & (diffs < 1.0)]  # chuck out big pauses/glitches
    if len(diffs) < 3:
        return 0.0
    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return 0.0
    sr = 1.0 / median_dt
    # snap to nice rates if we're close (helps people sleep better)
    for target in (128.0, 256.0, 512.0):
        if abs(sr - target) < target * 0.15:
            return target
    return sr


# ---------------
# nice CLI runner
# ---------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Emotiv EEG acquisition (general-purpose, CSV-friendly)")
    ap.add_argument("--csv", type=str, default=None, help="CSV output path")
    ap.add_argument("--backend", type=str, default="pyusb",
                    choices=["pyusb","pywinusb","bluetooth"],
                    help="Transport stack to use (try pyusb first)")
    ap.add_argument("--no-extras", action="store_true",
                    help="Don't include trailing extras after EEG (battery/quality/gyro etc.)")
    ap.add_argument("--no-counter", action="store_true",
                    help="Don't include the device sample counter column")
    ap.add_argument("--block", type=int, default=32,
                    help="Samples per yielded block (default: 32)")

    # verbosity: --quiet vs --verbose (mutually exclusive)
    v = ap.add_mutually_exclusive_group()
    v.add_argument("--quiet", action="store_true", help="Silence console prints")
    v.add_argument("--verbose", action="store_true", help="Force verbose prints")

    # optional log file
    ap.add_argument("--log", type=str, default=None, help="Write status logs to this file")
    ap.add_argument("--append-log", action="store_true",
                    help="Append to the log file instead of overwriting")

    args = ap.parse_args()

    opts = StreamOptions(
        py_backend=args.backend,
        verbose=(args.verbose or not args.quiet),   # quiet unless --verbose is set
        include_extras=not args.no_extras,
        include_counter=not args.no_counter,
        block_size=args.block,
        csv_path=args.csv,
        csv_title="EEG_recording",
        csv_overwrite=False,
        heartbeat_sec=5.0,
        log_path=args.log,
        log_append=args.append_log,
    )

    reader = EmotivReader(opts).start()
    try:
        info = reader.device_info(probe_seconds=2.5)
        if opts.verbose:
            print(f"[Emotiv] Ready. Family={info.family}  fs≈{info.sampling_hz:.2f} Hz  Labels={info.labels}")
            print("[Emotiv] Streaming… (Ctrl+C to stop)")
        for ts_block, x_block in reader.stream():
            # Tiny demo: compute block RMS over EEG channels.
            # If you included the 'counter' column, that's column 0 of x_block rows.
            eeg_cols = len(info.labels)
            offset = 1 if opts.include_counter else 0
            eeg_only = x_block[:, offset:offset + eeg_cols]
            rms = np.sqrt((eeg_only**2).mean(axis=0))
            if opts.verbose:
                print("RMS:", np.round(rms, 2))
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()