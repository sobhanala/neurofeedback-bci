#!/usr/bin/env python3
# analyze_xdf.py
# Comprehensive XDF exploration + MI analysis pipeline (plots + report).
# Works with markers like "event=cue_on;trial=7;cond=left" OR "cue_on/left".
#python analyze_xdf.py --xdf session.xdf --out out_reports --sfreq_override 256
#python analyze_xdf.py --xdf sub-P101_ses-S004_task-Default_run-001_eeg.xdf --out out_reports --sfreq_override 256
import os, sys, re, json, argparse, textwrap, datetime, warnings, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Safe for headless/Windows; we save figures instead of showing
import matplotlib.pyplot as plt

import mne
import pyxdf
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

# --------------------------
# Utility helpers
# --------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def savefig(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def parse_marker(payload):
    """Return (event, cond, trial) from either 'event=...;trial=...;cond=...' OR 'event/cond'."""
    s = str(payload)
    # Preferred schema
    if ";" in s and "=" in s:
        parts = dict(p.split("=", 1) for p in s.split(";") if "=" in p)
        ev = parts.get("event", "").strip()
        cond = parts.get("cond", "").strip() or None
        tri = parts.get("trial", "").strip()
        tri = int(tri) if tri.isdigit() else None
        if ev:
            return ev, cond, tri
    # Fallback "ev/cond"
    if "/" in s:
        ev, cond = s.split("/", 1)
        return ev.strip(), cond.strip(), None
    # Just event name
    return s.strip(), None, None

def pick_emotiv_montage(raw):
    """
    Set a reasonable montage if channel names match standard 10-20-ish.
    Emotiv EPOC+ commonly uses: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4.
    We'll try standard_1020; if some labels are unknown, MNE will ignore them.
    """
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")
    except Exception:
        pass

def bandpower_epochs(epochs, fmin, fmax, tmin, tmax):
    ep = epochs.copy().crop(tmin=tmin, tmax=tmax)
    psd, freqs = mne.time_frequency.psd_welch(ep, fmin=fmin, fmax=fmax, n_fft=256)
    # psd shape: (n_epochs, n_channels, n_freqs) -> average over freq, then epochs
    return psd.mean(axis=(0, 2)), psd.mean(axis=(2))  # (per-channel), (per-epoch, per-chan)

def safe_title(s):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s)[:64]

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Explore and analyze a LabRecorder .xdf file (Motor Imagery).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--xdf", required=True, help="Path to .xdf file")
    parser.add_argument("--out", default="reports_out", help="Output root folder")
    parser.add_argument("--subject", default="01", help="Subject ID for report labels")
    parser.add_argument("--session", default="01", help="Session ID for report labels")
    parser.add_argument("--task", default="mi", help="Task label")
    parser.add_argument("--sfreq_override", type=float, default=None,
                        help="Force sampling rate if XDF nominal_srate is unreliable")
    parser.add_argument("--use_flip_event", default="cue_on", help="Primary epoching event (cue_on or imagery_on)")
    parser.add_argument("--tmin", type=float, default=-0.5, help="Epoch start (s) relative to event")
    parser.add_argument("--tmax", type=float, default=4.0, help="Epoch end (s) relative to event")
    parser.add_argument("--baseline_start", type=float, default=-0.5, help="Baseline start (s)")
    parser.add_argument("--baseline_end", type=float, default=0.0, help="Baseline end (s)")
    args = parser.parse_args()

    out_root = ensure_dir(args.out)
    out_dir = ensure_dir(os.path.join(out_root, f"{ts()}_{safe_title(os.path.basename(args.xdf))}"))

    # Report text chunks we’ll collect
    report_lines = []
    def add_report(line=""):
        report_lines.append(line)

    add_report(f"# XDF Analysis Report")
    add_report(f"- File: `{args.xdf}`")
    add_report(f"- Subject: {args.subject} | Session: {args.session} | Task: {args.task}")
    add_report(f"- Output folder: `{out_dir}`")

    print(f"Loading XDF: {args.xdf}")
    streams, header = pyxdf.load_xdf(args.xdf, dejitter_timestamps=True)
    add_report(f"\n## Streams in XDF")
    for i, s in enumerate(streams):
        info = s["info"]
        sname = info["name"][0]
        stype = info["type"][0]
        sr = info.get("nominal_srate", ["0"])[0]
        add_report(f"- Stream {i}: name={sname}, type={stype}, nominal_srate={sr}")

    # Identify EEG and markers
    def stype(s): return s["info"]["type"][0].upper()
    eeg = None; mrk = None
    for s in streams:
        t = stype(s)
        if t == "EEG" and eeg is None:
            eeg = s
        if t in ("MARKERS", "MARKER", "EVENTS") and mrk is None:
            mrk = s
    if eeg is None:
        raise RuntimeError("No EEG stream found.")
    if mrk is None:
        print("WARNING: No MARKERS stream found. Proceeding without event parsing.")

    # Build raw
    if args.sfreq_override is not None:
        sfreq = float(args.sfreq_override)
    else:
        sfreq = float(eeg["info"]["nominal_srate"][0])

    ch_names = []
    try:
        ch_names = [c["label"][0] for c in eeg["info"]["desc"][0]["channels"][0]["channel"]]
    except Exception:
        # fallback generic names
        ch_names = [f"EEG{i+1}" for i in range(eeg["time_series"].shape[1])]

    ch_types = ["eeg"] * len(ch_names)
    data = np.asarray(eeg["time_series"], dtype=float).T  # (n_ch, n_samples)
    times = np.asarray(eeg["time_stamps"], dtype=float)
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_meas_date(None)        # keep relative timing
    raw.info["line_freq"] = 50.0   # EU default; change to 60 if needed

    # Montage for better plots/topos (optional)
    pick_emotiv_montage(raw)

    add_report(f"\n## EEG Summary")
    add_report(f"- Channels: {len(ch_names)}")
    add_report(f"- sfreq: {sfreq} Hz")
    add_report(f"- Duration: {raw.n_times/sfreq:.2f} s")
    add_report(f"- Ch names: {', '.join(ch_names)}")

    # Convert markers to annotations
    if mrk is not None:
        m_ts = np.asarray(mrk["time_stamps"], dtype=float)
        m_vals = mrk["time_series"]
        descs = []
        for v in m_vals:
            val = v[0] if isinstance(v, (list, tuple, np.ndarray)) else v
            ev, cond, tri = parse_marker(val)
            if cond:
                descs.append(f"{ev}/{cond}")
            else:
                descs.append(ev)
        # Map LSL absolute to samples relative to EEG start
        t0 = times[0]
        onset_samples = (m_ts - t0) * sfreq
        onsets = np.maximum(onset_samples / sfreq, 0.0)
        raw.set_annotations(mne.Annotations(onset=onsets,
                                            duration=np.zeros_like(onsets),
                                            description=descs))
        add_report(f"\n## Markers")
        unique_descs = sorted(set(descs))
        add_report(f"- Unique annotations: {unique_descs}")
        add_report(f"- Total annotations: {len(descs)}")
    else:
        unique_descs = []

    # ------------- Plots: Raw & PSD -------------
    fig = raw.plot(n_channels=min(14, len(ch_names)), scalings="auto", show=False, block=False)
    savefig(fig, os.path.join(out_dir, "raw_overview.png"))

    fig = raw.plot_psd(fmax=45, show=False)
    savefig(fig, os.path.join(out_dir, "psd.png"))

    # ------------- Events timeline -------------
    event_id = {d: i + 1 for i, d in enumerate(sorted(set(raw.annotations.description)))} if len(raw.annotations) else {}
    if event_id:
        events, _ = mne.events_from_annotations(raw, event_id=event_id)
        fig = mne.viz.plot_events(events, sfreq=raw.info["sfreq"], show=False)
        savefig(fig, os.path.join(out_dir, "events_timeline.png"))
        add_report(f"\n## Events")
        counts = pd.Series(raw.annotations.description).value_counts().to_dict()
        add_report(f"- Counts by annotation: {json.dumps(counts, indent=2)}")

    # ------------- Epoching -------------
    def get_event_subset(prefix):
        return {k: v for k, v in event_id.items() if k.startswith(prefix + "/") or k == prefix}

    # Choose primary event to epoch (cue_on by default)
    target_event_prefix = args.use_flip_event  # "cue_on" or "imagery_on"
    ev_primary = get_event_subset(target_event_prefix)

    # If imagery_on exists, also prepare those epochs for ERD check
    ev_imagery = get_event_subset("imagery_on")

    epochs = None; epochs_imagery = None
    if ev_primary:
        events_all, _ = mne.events_from_annotations(raw, event_id=event_id)
        mask = np.isin(events_all[:, 2], list(ev_primary.values()))
        events_use = events_all[mask]
        if len(events_use):
            epochs = mne.Epochs(raw, events_use, ev_primary,
                                tmin=args.tmin, tmax=args.tmax,
                                baseline=(args.baseline_start, args.baseline_end),
                                preload=True, detrend=1, reject_by_annotation=False)
            add_report(f"\n## Epochs ({target_event_prefix})")
            add_report(f"- n_epochs total: {len(epochs)}")
            add_report(f"- tmin={args.tmin}, tmax={args.tmax}, baseline=({args.baseline_start},{args.baseline_end})")

    if ev_imagery:
        events_all, _ = mne.events_from_annotations(raw, event_id=event_id)
        mask = np.isin(events_all[:, 2], list(ev_imagery.values()))
        events_use = events_all[mask]
        if len(events_use):
            epochs_imagery = mne.Epochs(raw, events_use, ev_imagery,
                                        tmin=-0.5, tmax=4.0,
                                        baseline=(-0.5, 0.0),
                                        preload=True, detrend=1, reject_by_annotation=False)
            add_report(f"\n## Epochs (imagery_on)")
            add_report(f"- n_epochs total: {len(epochs_imagery)}")
            add_report(f"- tmin=-0.5, tmax=4.0, baseline=(-0.5,0.0)")

    # ------------- Bandpower per condition (μ and β during imagery) -------------
    if epochs_imagery is not None and len(epochs_imagery) >= 2:
        # Split by cond using label suffix in event_id (e.g., imagery_on/left)
        cond_to_idx = {}
        for lbl, code in epochs_imagery.event_id.items():
            cond = lbl.split("/", 1)[1] if "/" in lbl else lbl
            cond_to_idx[cond] = (epochs_imagery.events[:, 2] == code)

        mu_band = (8.0, 13.0)
        beta_band = (13.0, 30.0)
        im_win = (0.5, 3.5)

        bp_results = {}
        for cond, mask in cond_to_idx.items():
            epc = epochs_imagery[mask]
            if len(epc) == 0:
                continue
            bp_mu_ch, _ = bandpower_epochs(epc, *mu_band, *im_win)
            bp_beta_ch, _ = bandpower_epochs(epc, *beta_band, *im_win)
            bp_results[cond] = {"mu": bp_mu_ch, "beta": bp_beta_ch}

        # Plot per-channel bar for μ and β (mean across channels)
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = sorted(bp_results.keys())
        mu_vals = [bp_results[c]["mu"].mean() for c in labels]
        beta_vals = [bp_results[c]["beta"].mean() for c in labels]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, mu_vals, width=0.4, label="mu 8-13Hz")
        ax.bar(x + 0.2, beta_vals, width=0.4, label="beta 13-30Hz")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Bandpower (a.u.)")
        ax.set_title("Imagery bandpower per condition")
        ax.legend()
        savefig(fig, os.path.join(out_dir, "bandpower_imagery_conditions.png"))

        add_report("\n## Bandpower (imagery 0.5-3.5s)")
        for cond in sorted(bp_results):
            add_report(f"- {cond}: mu_mean={bp_results[cond]['mu'].mean():.3f}, beta_mean={bp_results[cond]['beta'].mean():.3f}")

    # ------------- Time–Frequency (ERD/ERS quick view) -------------
    if epochs_imagery is not None and len(epochs_imagery) >= 2:
        # Pick a couple of motor-ish channels if present (Emotiv lacks C3/C4; use FC5/FC6 or T7/T8 as proxy)
        picks = [ch for ch in ["FC5", "FC6", "T7", "T8", "F3", "F4"] if ch in raw.ch_names]
        if len(picks) == 0:
            picks = mne.pick_types(raw.info, eeg=True)[:4]  # fallback few channels

        freqs = np.arange(6, 31, 1.0)
        n_cycles = freqs * 0.25  # ~4 cycles at 16 Hz
        try:
            power = mne.time_frequency.tfr_morlet(epochs_imagery.copy().pick_channels(picks),
                                                  freqs=freqs, n_cycles=n_cycles,
                                                  return_itc=False, average=True, decim=2, n_jobs="auto")
            fig = power.plot(baseline=(-0.5, 0.0), mode="percent", show=False)
            # mne may create multiple figs (one per channel)
            if isinstance(fig, list):
                for i, f in enumerate(fig):
                    savefig(f, os.path.join(out_dir, f"tfr_{picks[i]}.png"))
            else:
                savefig(fig, os.path.join(out_dir, "tfr.png"))
            add_report("\n## TFR")
            add_report("- Saved time–frequency representation with baseline in percent.")
        except Exception as e:
            add_report(f"\n## TFR")
            add_report(f"- Skipped (error: {e})")

    # ------------- Classification: CSP + LDA on imagery epochs -------------
    clf_acc = None
    if epochs_imagery is not None and len(epochs_imagery) >= 8:
        # Build X, y using only left vs right imagery
        y = []
        X = []
        for lbl, code in epochs_imagery.event_id.items():
            if "left" in lbl or "right" in lbl:
                mask = (epochs_imagery.events[:, 2] == code)
                if mask.sum() == 0:
                    continue
                ep = epochs_imagery[mask]
                # Crop to imagery window e.g. 0.5–3.5s
                epc = ep.copy().crop(0.5, 3.5)
                X.append(epc.get_data())  # (n_ep, n_ch, n_time)
                y.extend([0 if "left" in lbl else 1] * len(epc))

        if len(X):
            X = np.concatenate(X, axis=0)
            y = np.array(y)
            # CSP needs enough epochs
            if len(np.unique(y)) == 2 and X.shape[0] >= 8:
                csp = CSP(n_components=min(6, X.shape[1]), reg="ledoit_wolf", log=True, norm_trace=False)
                lda = LinearDiscriminantAnalysis()
                pipe = Pipeline([("csp", csp), ("lda", lda)])
                cv = StratifiedKFold(n_splits=min(5, np.bincount(y).min()), shuffle=True, random_state=42)
                scores = cross_val_score(pipe, X, y, cv=cv, n_jobs=None)
                clf_acc = scores.mean()
                add_report(f"\n## Classification (CSP+LDA on imagery 0.5-3.5s)")
                add_report(f"- CV folds: {cv.get_n_splits()}")
                add_report(f"- Mean accuracy: {clf_acc:.3f}")
                # Plot boxplot of scores
                fig, ax = plt.subplots()
                ax.boxplot(scores, labels=["CSP+LDA"])
                ax.set_ylim(0.0, 1.0)
                ax.set_ylabel("Accuracy")
                ax.set_title("Cross-validated accuracy")
                savefig(fig, os.path.join(out_dir, "csp_lda_cv_scores.png"))
            else:
                add_report("\n## Classification")
                add_report("- Skipped: not enough epochs or only one class present.")
        else:
            add_report("\n## Classification")
            add_report("- Skipped: no left/right imagery epochs found.")

    # ------------- Save a quick summary of timing sanity -------------
    if len(raw.annotations):
        rows = []
        for d, o in zip(raw.annotations.description, raw.annotations.onset):
            m = re.match(r"(fix_on|cue_on|imagery_on|rest_on|trial_end)(?:/(left|right))?", str(d))
            if m:
                rows.append({"event": m.group(1), "cond": m.group(2) or "NA", "onset": float(o)})
        tdf = pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)
        # Attempt to parse into 5-event blocks (fix, cue, imagery, rest, end)
        expected = ["fix_on", "cue_on", "imagery_on", "rest_on", "trial_end"]
        blocks = []
        i = 0
        while i <= len(tdf) - 5:
            blk = tdf.iloc[i:i+5]
            if list(blk["event"]) == expected and len(set(blk["cond"])) == 1:
                times = dict(zip(blk["event"], blk["onset"]))
                blocks.append({
                    "cond": blk["cond"].iloc[0],
                    "fix_to_cue": times["cue_on"] - times["fix_on"],
                    "cue_to_imagery": times["imagery_on"] - times["cue_on"],
                    "imagery_to_rest": times["rest_on"] - times["imagery_on"],
                    "rest_to_end": times["trial_end"] - times["rest_on"],
                    "trial_dur": times["trial_end"] - times["fix_on"],
                })
                i += 5
            else:
                i += 1
        if blocks:
            bdf = pd.DataFrame(blocks)
            bdf.to_csv(os.path.join(out_dir, "timing_summary.csv"), index=False)
            add_report("\n## Timing sanity (s)")
            desc = bdf[["fix_to_cue","cue_to_imagery","imagery_to_rest","rest_to_end","trial_dur"]].describe().round(3)
            add_report(desc.to_string())
            # Plot quick bars
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(["fix->cue","cue->imag","imag->rest","rest->end"],
                   [bdf[k].median() for k in ["fix_to_cue","cue_to_imagery","imagery_to_rest","rest_to_end"]])
            ax.set_ylim(0, max(2.5, float(bdf[["fix_to_cue","cue_to_imagery","imagery_to_rest","rest_to_end"]].max().max())+0.5))
            ax.set_ylabel("Seconds (median)")
            ax.set_title("Median phase durations")
            savefig(fig, os.path.join(out_dir, "timing_medians.png"))

    # ------------- Save report -------------
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nAll done. Report: {report_path}")
    print(f"Figures and CSVs saved in: {out_dir}")

if __name__ == "__main__":
    # Quiet a few benign warnings for cleaner output
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
