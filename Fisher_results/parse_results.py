"""Parse the optimized best-fit points out of src/IMRI/results_combined.txt and
src/EMRI/results_compiled.txt into a flat list of case dicts, ready to feed to
compute_fishers.py (Fisher matrix computation) and plot_bias.py (plotting).

Each results_*.txt file is organized as repeated blocks:

    #=== SECTION banner (optionally "SUPERSEDED") ===
    # <point_name>   a=...  e0=...  SNR=...  dt=...  T=...  [chi2(sec)=...]
    signal_param = {...}                       # truth, spans multiple lines

    ## <model>   (optional seed note)   overlap = X   chi2 = Y   [# comment]
    x_bf = [...]                               # single line
    #  [optional extra comment lines, ignored]
    ...

We do not use a regex-per-line state machine; instead we locate every
"signal_param = {...}" block (multi-line dict literal, safe to
ast.literal_eval) and every "## <model> ... x_bf = [...]" pair that follows it,
up to the next signal_param block.
"""
import ast
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
IMRI_TXT = os.path.join(HERE, "..", "src", "IMRI", "results_combined.txt")
EMRI_TXT = os.path.join(HERE, "..", "src", "EMRI", "results_compiled.txt")

PARAM_NAMES_14 = ["m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS",
                  "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]
PARAMS_9 = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0"]
PARAMS_11 = PARAMS_9 + ["dev_1", "dev_2"]

# Section-1 "grid_idxN" cases supersede these earlier idx1_dt5/idx16_dt5 duplicates
# (idx16_dt5 is an exact re-derivation of grid_idx16; idx1_dt5 is kept, see IMRI
# results_combined.txt Section 3 header note).
IMRI_EXCLUDE_POINTS = {"idx16_dt5"}

HEADER_RE = re.compile(
    r"^#\s*(\S+)(?:.*?\bdt=([\d.]+))?(?:.*?\bT=([\d.]+))?(?:.*?chi2\(sec\)=([\d.]+))?",
)
MODEL_RE = re.compile(
    r"^##\s*(\S+)\b.*?overlap\s*=\s*([\d.eE+-]+)\s*chi2\s*=\s*([\d.eE+-]+)",
)
SNR_RE = re.compile(r"\bSNR=([\d.eE+-]+)")


def _find_signal_param_blocks(text):
    """Yield (start, end, header_line, dict_text) for every signal_param={...} block."""
    for m in re.finditer(r"^signal_param\s*=\s*\{", text, re.M):
        start = m.start()
        # nearest preceding line that looks like an actual point header (has
        # "dt=" on it) -- skips intervening sub-comments like
        # "#   SNR per channel: A=... E=... T=..." that appear between the
        # point header and its signal_param block.
        pre_lines = text[:start].rstrip("\n").splitlines()
        header_line = ""
        for line in reversed(pre_lines):
            if not line.strip():
                continue
            if re.match(r"^#\s*\S", line) and re.search(r"\bdt=[\d.]+\s*,\s*T=", line):
                header_line = line
                break
            if not line.strip().startswith("#"):
                break
        # find the matching closing brace (dict body has no nested braces here)
        close = text.index("}", m.end())
        dict_text = text[m.start() + len("signal_param = "): close + 1]
        yield start, close + 1, header_line, dict_text


def _superseded_before(text, pos):
    """True if the nearest preceding '# SECTION ... banner' contains SUPERSEDED."""
    banner_iter = list(re.finditer(r"^#\s*SECTION.*$", text[:pos], re.M))
    if not banner_iter:
        return False
    last = banner_iter[-1].group(0)
    return "SUPERSEDED" in last.upper()


def parse_file(path, system, default_chi2_sec, default_T, exclude_points=()):
    text = open(path).read()
    blocks = list(_find_signal_param_blocks(text))
    cases = []
    per_point_snr = {}

    for i, (start, end, header_line, dict_text) in enumerate(blocks):
        m = HEADER_RE.match(header_line.strip())
        if not m:
            continue
        point = m.group(1)
        dt = float(m.group(2)) if m.group(2) else None
        T = float(m.group(3)) if m.group(3) else default_T
        chi2_sec = float(m.group(4)) if m.group(4) else default_chi2_sec
        snr_m = SNR_RE.search(header_line)
        snr = float(snr_m.group(1)) if snr_m else None
        if snr is not None:
            per_point_snr[point] = snr

        if point in exclude_points or _superseded_before(text, start):
            continue

        signal_param = ast.literal_eval(dict_text)

        # scan model blocks between this signal_param and the next one
        seg_end = blocks[i + 1][0] if i + 1 < len(blocks) else len(text)
        segment = text[end:seg_end]
        lines = segment.splitlines()
        j = 0
        while j < len(lines):
            mm = MODEL_RE.match(lines[j].strip())
            if mm:
                model = mm.group(1)
                overlap = float(mm.group(2))
                chi2_val = float(mm.group(3))
                # next non-blank line should be x_bf = [...]
                k = j + 1
                while k < len(lines) and not lines[k].strip():
                    k += 1
                x_bf = None
                if k < len(lines) and lines[k].strip().startswith("x_bf"):
                    arr_text = lines[k].split("=", 1)[1].strip()
                    x_bf = ast.literal_eval(arr_text)
                if x_bf is not None:
                    n = len(x_bf)
                    param_names = PARAMS_9 if n == 9 else PARAMS_11
                    if dt is None:
                        raise ValueError(f"{path}: {point} missing dt=... in header: {header_line!r}")
                    cases.append(dict(
                        system=system, point=point, model=model,
                        dt=dt, T=T, chi2_sec=chi2_sec,
                        signal_param=signal_param,
                        x_bf=list(x_bf), param_names=param_names,
                        overlap=overlap, chi2_val=chi2_val,
                        snr=snr,  # filled in below if missing
                        branch="dev_a_pe" if model == "simple_pe" else "hybrid",
                    ))
            j += 1

    # backfill missing per-point SNR from a sibling model at the same point
    for c in cases:
        if c["snr"] is None:
            c["snr"] = per_point_snr.get(c["point"])

    return cases


def parse_all():
    imri = parse_file(IMRI_TXT, "IMRI", default_chi2_sec=0.95, default_T=1.0,
                       exclude_points=IMRI_EXCLUDE_POINTS)
    emri = parse_file(EMRI_TXT, "EMRI", default_chi2_sec=0.95, default_T=2.5)
    return imri + emri


def fisher_filename(case):
    return f"Fisher_{case['point']}_{case['model']}.npy"


if __name__ == "__main__":
    cases = parse_all()
    by_sys = {}
    for c in cases:
        by_sys.setdefault(c["system"], set()).add(c["point"])
    for sys_name, pts in by_sys.items():
        n = sum(1 for c in cases if c["system"] == sys_name)
        print(f"{sys_name}: {len(pts)} points, {n} (point, model) cases")
        for p in sorted(pts):
            models = [c["model"] for c in cases if c["system"] == sys_name and c["point"] == p]
            snrs = {c["snr"] for c in cases if c["system"] == sys_name and c["point"] == p}
            print(f"  {p:15s} models={models}  snr={snrs}")
    print(f"\nTotal cases: {len(cases)}")
    missing_snr = [c for c in cases if c["snr"] is None]
    if missing_snr:
        print(f"[WARN] {len(missing_snr)} cases missing SNR: "
              f"{[(c['system'], c['point'], c['model']) for c in missing_snr]}")
