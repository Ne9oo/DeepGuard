"""DeepGuard forensic evidence record + PDF report builder.

Used by app.py:
    sha256_file(path)                 -> hash of the uploaded evidence file
    probe_audio(path)                 -> technical metadata of the audio file
    build_evidence(...)               -> dict saved as evidence/scan_<id>.json
    build_report_pdf(...)             -> bytes of the forensic PDF report
"""
import hashlib
import os
import tempfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from fpdf import FPDF
from PIL import Image

TOOL_NAME = "DeepGuard Forensic Terminal"
TOOL_VERSION = "1.0"
MODEL_NAME = "DeepGuard CNN v1 (PyTorch)"
TRAINING_DATA = "ASVspoof 2019 Logical Access (LA) training partition"
# >>> Update this sentence whenever you retrain or replace the model. <<<
MODEL_EVAL_NOTE = ("Measured on the ASVspoof 2019 LA evaluation set (attack types never seen in "
                   "training): 15.1% Equal Error Rate.")


# ------------------------------------------------------------------ evidence
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def probe_audio(path):
    """Technical metadata. Returns {} for formats soundfile cannot read (.mpeg, .mp4 ...)."""
    try:
        i = sf.info(path)
        return {"container": i.format, "codec": i.subtype, "sample_rate_hz": i.samplerate,
                "channels": i.channels, "duration_s": round(i.duration, 2)}
    except Exception:
        return {}


def tier(pct):
    """Same bands as the dashboard: 0-30 low, 31-70 medium, 71-100 high."""
    if pct <= 30:
        return ("Authentic Human Voice", "LOW RISK", (30, 132, 73),
                "No indicators of synthetic speech were found above the detection threshold. "
                "The examined audio is consistent with a genuine human speaker.")
    if pct <= 70:
        return ("Suspicious / Inconclusive", "MEDIUM RISK", (214, 137, 16),
                "The audio shows ambiguous characteristics. The result is inconclusive and "
                "review by a human expert is recommended.")
    return ("Synthetic Deepfake", "HIGH RISK", (192, 57, 43),
            "The audio shows strong indicators consistent with synthetic (AI-generated or cloned) "
            "speech. Independent verification is recommended before this result is relied upon.")


def build_evidence(scan_id, original_filename, sha256, size_bytes, probe, uploaded_utc,
                   username, scan_mode, file_retained, details, result, confidence,
                   risk_level, model_sha256):
    return {
        "report_no": f"DG-{scan_id:06d}",
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "examiner": username,
        "evidence": {
            "original_filename": original_filename,
            "sha256": sha256,
            "size_bytes": size_bytes,
            "uploaded_utc": uploaded_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "file_retained_on_server": bool(file_retained),
            "technical": probe,
        },
        "analysis": {
            "scan_mode": scan_mode,
            "audio_seconds_analysed": round(details["audio_seconds"], 2),
            "window_seconds": details["window_seconds"],
            "model_sample_rate_hz": details["model_sample_rate_hz"],
            "synthetic_probability_pct": confidence,
            "classification": result,
            "risk_level": risk_level,
            "decision_threshold": 0.5,
            "windows": details["windows"],
        },
        "model": {"name": MODEL_NAME, "weights_sha256": model_sha256,
                  "training_data": TRAINING_DATA, "evaluation": MODEL_EVAL_NOTE},
    }


# ------------------------------------------------------------------ PDF
def _t(s):
    """fpdf 1.7 only supports latin-1 text."""
    return str(s).encode("latin-1", "replace").decode("latin-1")


class ForensicPDF(FPDF):
    def __init__(self, report_no, sha_short):
        FPDF.__init__(self)
        self.report_no = report_no
        self.sha_short = sha_short
        self.alias_nb_pages()
        self.set_margins(10, 12, 10)
        self.set_auto_page_break(True, 18)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Arial", "", 8)
            self.set_text_color(110, 110, 110)
            self.cell(0, 5, _t(f"{TOOL_NAME} - Forensic Report {self.report_no}"), 0, 1, "L")
            self.set_text_color(0, 0, 0)
            self.ln(2)

    def footer(self):
        self.set_y(-14)
        self.set_font("Arial", "I", 8)
        self.set_text_color(110, 110, 110)
        self.cell(130, 5, _t(f"Report {self.report_no} | Evidence SHA-256: {self.sha_short}"), 0, 0, "L")
        self.cell(0, 5, "Page " + str(self.page_no()) + " of {nb}", 0, 0, "R")
        self.set_text_color(0, 0, 0)


def _section(pdf, title):
    pdf.ln(4)
    if pdf.get_y() > 255:
        pdf.add_page()
    pdf.set_fill_color(15, 23, 42)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7.5, "  " + _t(title), 0, 1, "L", True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _kv(pdf, label, value, mono=False):
    pdf.set_font("Arial", "B", 9.5)
    pdf.cell(48, 6.5, _t(label))
    if mono:
        pdf.set_font("Courier", "", 8.5)
    else:
        pdf.set_font("Arial", "", 9.5)
    pdf.multi_cell(0, 6.5, _t(value))


def _para(pdf, text, italic=False, size=9.5):
    pdf.set_font("Arial", "I" if italic else "", size)
    pdf.multi_cell(0, 5.5, _t(text))
    pdf.ln(1)


def _bullet(pdf, text):
    pdf.set_font("Arial", "", 9.5)
    pdf.set_x(14)
    pdf.multi_cell(0, 5.5, _t("- " + text))
    pdf.ln(0.5)


def _window_chart(windows):
    fig, ax = plt.subplots(figsize=(7.4, 2.6), dpi=150)
    xs = [w["start_s"] for w in windows]
    ps = [w["p_fake"] for w in windows]
    ws = [max(w["end_s"] - w["start_s"], 0.2) for w in windows]
    cols = ["#c0392b" if p > 0.7 else "#d68910" if p > 0.3 else "#1e8449" for p in ps]
    ax.bar(xs, ps, width=ws, align="edge", color=cols, edgecolor="white", linewidth=0.5, alpha=0.9)
    ax.axhline(0.5, ls="--", lw=0.9, color="#444444")
    ax.text(0.99, 0.52, "decision threshold (0.5)", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color="#444444")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(w["end_s"] for w in windows))
    ax.set_xlabel("Time into analysed audio (s)", fontsize=8)
    ax.set_ylabel("P(synthetic)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(tmp.name, format="png")
    plt.close(fig)
    return tmp.name


def build_report_pdf(scan, evidence, username, spectrogram_path=None, generated=None):
    generated = generated or datetime.utcnow()
    pct = float(scan.confidence)
    verdict, risk, color, conclusion = tier(pct)
    rn = f"DG-{scan.id:06d}"
    ev = evidence["evidence"] if evidence else None
    an = evidence["analysis"] if evidence else None
    sha = ev["sha256"] if ev else "not recorded"
    pdf = ForensicPDF(rn, sha[:16] + ("..." if ev else ""))
    pdf.add_page()

    # ---- title band
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_xy(10, 8)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 9, "DEEPGUARD", 0, 1)
    pdf.set_x(10)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, "Digital Audio Forensic Examination Report - Synthetic Speech Detection", 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(36)

    # ---- case information
    _section(pdf, "1. CASE INFORMATION")
    _kv(pdf, "Report No.:", rn)
    _kv(pdf, "Report generated:", generated.strftime("%Y-%m-%d %H:%M:%S UTC"))
    _kv(pdf, "Examination date:", scan.scan_date.strftime("%Y-%m-%d %H:%M:%S UTC"))
    _kv(pdf, "Requested by:", username)
    _kv(pdf, "Examination type:", "Automated analysis (no manual intervention)")
    _kv(pdf, "Tool:", f"{TOOL_NAME} v{TOOL_VERSION}")

    # ---- evidence identification
    _section(pdf, "2. EVIDENCE IDENTIFICATION")
    _kv(pdf, "File name:", scan.filename)
    if ev:
        _kv(pdf, "SHA-256:", ev["sha256"], mono=True)
        _kv(pdf, "File size:", f"{ev['size_bytes']:,} bytes")
        tech = ev.get("technical") or {}
        if tech:
            _kv(pdf, "Container / codec:", f"{tech.get('container', '?')} / {tech.get('codec', '?')}")
            _kv(pdf, "Sample rate / channels:", f"{tech.get('sample_rate_hz', '?')} Hz / {tech.get('channels', '?')}")
            _kv(pdf, "Total duration:", f"{tech.get('duration_s', '?')} s")
        else:
            _kv(pdf, "Technical metadata:", "Not available for this file format")
        _kv(pdf, "Received (UTC):", ev["uploaded_utc"])
        _kv(pdf, "Copy kept on server:", "Yes" if ev["file_retained_on_server"]
            else "No (deleted after analysis, per account setting)")
    else:
        _kv(pdf, "Integrity record:", "Not available - this scan was made before forensic logging was enabled")

    # ---- findings
    _section(pdf, "3. FINDINGS")
    y = pdf.get_y()
    pdf.set_fill_color(*color)
    pdf.rect(10, y, 190, 22, "F")
    pdf.set_xy(14, y + 3)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 15)
    pdf.cell(125, 8, _t(verdict))
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, risk, 0, 1, "R")
    pdf.set_x(14)
    pdf.set_font("Arial", "", 10)
    extra = f"   |   Audio analysed: {an['audio_seconds_analysed']} s" if an else ""
    pdf.cell(0, 6, f"Synthetic probability: {pct:.1f}%   |   Decision threshold: 50%{extra}")
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(y + 26)
    _para(pdf, "Conclusion: " + conclusion)

    # ---- segment analysis
    windows = an["windows"] if an else []
    if windows:
        _section(pdf, "4. SEGMENT-LEVEL ANALYSIS")
        ps = np.array([w["p_fake"] for w in windows])
        flagged = int((ps > 0.5).sum())
        _para(pdf, f"The audio was divided into {len(windows)} window(s) of {an['window_seconds']} seconds, "
                   "each scored independently. The overall score is the mean of the window scores. "
                   "Consistent scores across windows give a more reliable result; large swings between "
                   "windows may indicate mixed, edited or low-quality audio.")
        _kv(pdf, "Windows scored:", str(len(windows)))
        _kv(pdf, "Windows above 50%:", f"{flagged} of {len(windows)}")
        _kv(pdf, "Score range:", f"min {ps.min() * 100:.1f}%   max {ps.max() * 100:.1f}%   "
                                  f"mean {ps.mean() * 100:.1f}%   std. dev. {ps.std() * 100:.1f}%")
        pdf.ln(2)
        chart = _window_chart(windows)
        try:
            if pdf.get_y() + 70 > 275:
                pdf.add_page()
            yy = pdf.get_y()
            pdf.image(chart, x=10, y=yy, w=190)
            pdf.set_y(yy + 190 * 2.6 / 7.4 + 3)
        finally:
            os.remove(chart)

        pdf.set_font("Arial", "B", 9)
        pdf.set_fill_color(225, 229, 238)
        for w_, lbl in ((16, "No."), (50, "Time range (s)"), (40, "P(synthetic)"), (40, "Window result")):
            pdf.cell(w_, 6.5, lbl, 1, 0, "C", True)
        pdf.ln()
        pdf.set_font("Arial", "", 9)
        for w in windows[:150]:
            p = w["p_fake"] * 100
            lbl = "LOW" if p <= 30 else "MEDIUM" if p <= 70 else "HIGH"
            pdf.cell(16, 6, str(w["index"] + 1), 1, 0, "C")
            pdf.cell(50, 6, f"{w['start_s']:.2f} - {w['end_s']:.2f}", 1, 0, "C")
            pdf.cell(40, 6, f"{p:.1f}%", 1, 0, "C")
            pdf.cell(40, 6, lbl, 1, 1, "C")
        if len(windows) > 150:
            _para(pdf, f"(First 150 of {len(windows)} windows shown. Full data is in the JSON evidence export.)",
                  italic=True, size=8.5)

    # ---- spectrogram
    if spectrogram_path and os.path.exists(spectrogram_path):
        _section(pdf, "5. SPECTROGRAM")
        with Image.open(spectrogram_path) as im:
            h = 190.0 * im.size[1] / im.size[0]
        if pdf.get_y() + h > 275:
            pdf.add_page()
        yy = pdf.get_y()
        pdf.image(spectrogram_path, x=10, y=yy, w=190)
        pdf.set_y(yy + h + 2)
        _para(pdf, "Mel-spectrogram of the analysed audio (frequency vs. time, colour = energy in dB). "
                   "Shown for visual inspection; the detector's score is not read from this image.",
              italic=True, size=8.5)

    # ---- methodology
    _section(pdf, "6. METHODOLOGY")
    _para(pdf, "The file is decoded and converted to 16 kHz mono audio. It is cut into 4-second windows "
               "(a final overlapping window covers any remainder). Each window is peak-normalised and "
               "converted to an 80-band log-mel spectrogram (1024-point FFT, hop length 256). A convolutional "
               "neural network (4 convolution blocks, global pooling, 2-class output) scores each window for "
               "synthetic speech. The synthetic probability is the mean of the window scores.")
    _kv(pdf, "Model:", MODEL_NAME)
    _kv(pdf, "Training data:", TRAINING_DATA)
    if evidence:
        _kv(pdf, "Model weights SHA-256:", evidence["model"]["weights_sha256"], mono=True)
    _kv(pdf, "Risk bands:", "0-30% Low (consistent with genuine speech); 31-70% Medium (inconclusive); "
                            "71-100% High (consistent with synthetic speech).")

    # ---- limitations
    _section(pdf, "7. LIMITATIONS AND INTERPRETATION")
    _bullet(pdf, "This is an automated statistical assessment. It is NOT conclusive proof that audio is "
                 "genuine or fabricated and must be corroborated by independent examination before it is "
                 "relied on in any legal, disciplinary or investigative proceeding.")
    _bullet(pdf, MODEL_EVAL_NOTE)
    _bullet(pdf, "Accuracy can drop with heavy compression, telephone or messaging-app audio, background "
                 "noise, very short clips, and languages or speech-synthesis systems not represented in "
                 "the training data.")
    _bullet(pdf, "Scores are model outputs, not calibrated probabilities.")
    _bullet(pdf, "The analysis does not establish speaker identity and does not detect editing or splicing "
                 "of genuine recordings.")

    # ---- custody
    _section(pdf, "8. INTEGRITY AND VERIFICATION")
    if ev:
        _para(pdf, "The SHA-256 value in Section 2 was computed on the file exactly as received, before any "
                   "processing. To verify that a file is the one examined, compute its hash and compare:")
        pdf.set_font("Courier", "", 8.5)
        pdf.set_x(14)
        pdf.multi_cell(0, 5, "Windows PowerShell:  Get-FileHash <file> -Algorithm SHA256")
        pdf.set_x(14)
        pdf.multi_cell(0, 5, "Linux / macOS:        sha256sum <file>")
        pdf.ln(1)
        _para(pdf, "Identical hashes mean the two files are bit-for-bit identical. The full machine-readable "
                   "record (including every window score) is available as a JSON evidence export.")
    else:
        _para(pdf, "No integrity hash was recorded for this scan. Re-scan the original file to produce a "
                   "hashed evidence record.")

    out = pdf.output(dest="S")
    return out.encode("latin1") if isinstance(out, str) else bytes(out)