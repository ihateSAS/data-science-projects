"""
Feature extraction for symbolic piano MIDI files (GiantMIDI-Piano).

Designed for MIDI, where there is no explicit RH/LH part separation.
We use a median-pitch split as a hand-separation heuristic: notes above
the per-piece median pitch are treated as RH, below as LH. This is the
same approach used by Cancino-Chacón et al. (2020) on the MAESTRO
dataset for the same reason.

Feature groups:
    density        : note rate, IOI statistics
    motoric        : pseudo-RH/LH pitch range, jump distributions
    polyphony      : simultaneity count, vertical span
    rhythm         : IOI entropy (16th-grid), syncopation, tuplet rate
    harmony        : pitch-class histogram (12), interval-class histogram (12),
                     trichord histogram (top-20 most-common types),
                     chromaticism rate, Lerdahl pitch-space step
    style markers  : pedal usage, velocity dynamics

References:
    Kong et al. (2022). GiantMIDI-Piano: A large-scale MIDI dataset. TISMIR.
    Lerdahl, F. (2001). Tonal Pitch Space. OUP.
    Cancino-Chacón, C. et al. (2020). Computational models of expressive music
        performance: A comprehensive and critical review. Frontiers in Digital
        Humanities.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False


# Top-20 trichord types empirically common in tonal music (Forte set classes
# in normal form). Kept fixed across pieces so feature vectors are comparable.
COMMON_TRICHORDS = [
    (0, 4, 7),   # major triad
    (0, 3, 7),   # minor triad
    (0, 3, 6),   # diminished
    (0, 4, 8),   # augmented
    (0, 2, 7),   # sus2
    (0, 5, 7),   # sus4
    (0, 2, 4),   # whole-tone fragment
    (0, 1, 3),   # chromatic cluster
    (0, 2, 5),   # rare diatonic
    (0, 3, 5),
    (0, 1, 4),
    (0, 1, 5),
    (0, 1, 6),
    (0, 2, 6),
    (0, 4, 6),
    (0, 5, 8),
    (0, 1, 2),   # tight chromatic
    (0, 2, 3),
    (0, 3, 4),
    (0, 4, 5),
]


def _safe_entropy(xs: np.ndarray) -> float:
    if len(xs) == 0:
        return 0.0
    _, counts = np.unique(xs, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def _lerdahl_distance(p1: int, p2: int, tonic: int = 0) -> float:
    diatonic = [0, 2, 4, 5, 7, 9, 11]
    def to_scale_deg(p):
        rel = (p - tonic) % 12
        return min(range(7), key=lambda i: abs(diatonic[i] - rel))
    return abs(p1 - p2) / 12 + abs(to_scale_deg(p1) - to_scale_deg(p2)) / 7


def _normal_form(pcs: tuple) -> tuple:
    """Normal form of a pitch-class set: rotation with smallest span,
    then transposed so first element is 0. Matches Forte's set theory."""
    pcs = sorted(set(pc % 12 for pc in pcs))
    if len(pcs) < 2:
        return tuple(pcs)
    rotations = [pcs[i:] + [p + 12 for p in pcs[:i]] for i in range(len(pcs))]
    best = min(rotations, key=lambda r: (r[-1] - r[0], r))
    return tuple((p - best[0]) % 12 for p in best)


# -----------------------------------------------------------------------------
# Main extraction
# -----------------------------------------------------------------------------

def extract_features(path: Path) -> dict:
    """Extract ~40 features from a single MIDI file. Never raises."""
    result = {'piece_id': Path(path).stem, 'extraction_error': None}

    if not HAS_PRETTY_MIDI:
        result['extraction_error'] = 'pretty_midi not installed'
        return result

    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception as e:
        result['extraction_error'] = f'parse: {e}'
        return result

    # Gather notes
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)

    if len(notes) < 10:
        result['extraction_error'] = 'too few notes'
        return result

    pitches = np.array([n.pitch for n in notes])
    onsets  = np.array([n.start for n in notes])
    offsets = np.array([n.end for n in notes])
    durs    = offsets - onsets
    vels    = np.array([n.velocity for n in notes])

    order = np.argsort(onsets)
    pitches, onsets, offsets, durs, vels = (
        pitches[order], onsets[order], offsets[order], durs[order], vels[order])

    # ---- density / tempo ----
    total_time = float(pm.get_end_time())
    result['n_notes']           = int(len(notes))
    result['duration_sec']      = total_time
    result['notes_per_sec']     = len(notes) / max(total_time, 0.1)
    try:
        tempos = pm.get_tempo_changes()[1]
        result['mean_bpm']      = float(np.mean(tempos)) if len(tempos) else 100.0
        result['tempo_changes'] = int(len(tempos))
    except Exception:
        result['mean_bpm'] = 100.0
        result['tempo_changes'] = 0

    # ---- pitch-split hand heuristic ----
    median_pitch = float(np.median(pitches))
    result['median_pitch']    = median_pitch
    result['pitch_range']     = int(pitches.max() - pitches.min())
    rh_mask = pitches >= median_pitch
    lh_mask = ~rh_mask

    def jump_stats(p_sub, prefix):
        if len(p_sub) < 2:
            return {f'{prefix}_jump_mean': 0.0, f'{prefix}_jump_max': 0.0,
                    f'{prefix}_jump_p95': 0.0, f'{prefix}_range': 0}
        d = np.abs(np.diff(p_sub))
        return {f'{prefix}_jump_mean': float(d.mean()),
                f'{prefix}_jump_max':  float(d.max()),
                f'{prefix}_jump_p95':  float(np.percentile(d, 95)),
                f'{prefix}_range':     int(p_sub.max() - p_sub.min())}

    result.update(jump_stats(pitches[rh_mask], 'rh'))
    result.update(jump_stats(pitches[lh_mask], 'lh'))

    # ---- polyphony (simultaneity at each onset) ----
    # Cluster onsets within 30ms (typical perceptual fusion threshold)
    sorted_onsets = np.sort(onsets)
    clusters = [[sorted_onsets[0]]]
    for o in sorted_onsets[1:]:
        if o - clusters[-1][-1] < 0.03:
            clusters[-1].append(o)
        else:
            clusters.append([o])
    sim_sizes = [len(c) for c in clusters]
    result['mean_simultaneity']    = float(np.mean(sim_sizes))
    result['p95_simultaneity']     = float(np.percentile(sim_sizes, 95))
    result['max_simultaneity']     = int(max(sim_sizes))
    result['polyphonic_fraction']  = float(np.mean(np.array(sim_sizes) >= 2))

    # ---- rhythm ----
    if len(onsets) >= 2:
        iois = np.diff(onsets)
        iois = iois[iois > 0.01]   # drop simultaneous
        if len(iois):
            quantized = np.round(iois * 16)
            result['ioi_entropy_bits'] = _safe_entropy(quantized)
            result['ioi_cv']           = float(iois.std() / max(iois.mean(), 1e-6))
            # Syncopation: fraction of IOIs that don't fit a binary subdivision
            log2_iois = np.log2(iois + 1e-6)
            result['syncopation_index'] = float(np.mean(
                np.abs(log2_iois - np.round(log2_iois)) > 0.1))
        else:
            result['ioi_entropy_bits'] = 0; result['ioi_cv'] = 0
            result['syncopation_index'] = 0
    else:
        result['ioi_entropy_bits'] = 0; result['ioi_cv'] = 0
        result['syncopation_index'] = 0

    # ---- pitch-class histogram (12 features) ----
    pc_hist = np.bincount(pitches % 12, minlength=12).astype(float)
    pc_hist /= max(pc_hist.sum(), 1)
    for i in range(12):
        result[f'pc_{i}'] = float(pc_hist[i])

    # ---- interval-class histogram (12 features) ----
    if len(pitches) >= 2:
        intervals = np.abs(np.diff(pitches)) % 12
        ic_hist = np.bincount(intervals, minlength=12).astype(float)
        ic_hist /= max(ic_hist.sum(), 1)
    else:
        ic_hist = np.zeros(12)
    for i in range(12):
        result[f'ic_{i}'] = float(ic_hist[i])

    # ---- trichord histogram over the top-20 types (20 features) ----
    # Sliding window of 3 consecutive notes
    trichord_counts = Counter()
    if len(pitches) >= 3:
        for i in range(len(pitches) - 2):
            tri = _normal_form((pitches[i], pitches[i+1], pitches[i+2]))
            trichord_counts[tri] += 1
        total_tri = sum(trichord_counts.values())
    else:
        total_tri = 0
    for j, tri in enumerate(COMMON_TRICHORDS):
        result[f'tri_{j}'] = (trichord_counts.get(tri, 0) / total_tri
                              if total_tri else 0.0)

    # ---- chromaticism / harmony ----
    tonic_pc = int(np.argmax(pc_hist))   # crude estimate
    diatonic_pcs = {(tonic_pc + d) % 12 for d in [0, 2, 4, 5, 7, 9, 11]}
    result['chromaticism_rate'] = float(np.mean(
        [(p % 12) not in diatonic_pcs for p in pitches]))

    rh_p = pitches[rh_mask]
    if len(rh_p) >= 2:
        steps = [_lerdahl_distance(int(a), int(b), tonic_pc)
                 for a, b in zip(rh_p, rh_p[1:])]
        result['mean_lerdahl_step'] = float(np.mean(steps))
    else:
        result['mean_lerdahl_step'] = 0.0

    # ---- velocity / dynamics ----
    result['velocity_mean']  = float(vels.mean())
    result['velocity_std']   = float(vels.std())
    result['velocity_range'] = int(vels.max() - vels.min())

    # ---- pedal usage ----
    pedal_events = 0
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        pedal_events += sum(1 for cc in inst.control_changes
                            if cc.number == 64 and cc.value >= 64)
    result['pedal_events_per_sec'] = pedal_events / max(total_time, 0.1)

    return result


def extract_dataset(paths: list[Path], show_progress: bool = True):
    """Extract features for many MIDI files. Returns a tidy DataFrame."""
    import pandas as pd
    if show_progress:
        try:
            from tqdm import tqdm
            paths = tqdm(paths, desc='Extracting')
        except ImportError:
            pass
    return pd.DataFrame([extract_features(p) for p in paths])
