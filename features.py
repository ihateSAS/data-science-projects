"""
Feature extraction for symbolic piano scores.

Each piece -> a fixed-length vector of musicologically motivated features.

Feature groups:
    density   : note rate, IOI statistics
    motoric   : per-hand pitch range, jump distribution, simultaneous span
    polyphony : voice count, chord density, voice-leading parsimony
    rhythm    : IOI entropy, syncopation index, n-tuplet rate
    harmony   : Lerdahl pitch-space distance between consecutive verticalities,
                modulation count, chromaticism rate

References:
    Lerdahl, F. (2001). Tonal Pitch Space. Oxford University Press.
    Toiviainen, P. & Eerola, T. (2016). MIDI Toolbox 1.1.
    Ramoneda, P. et al. (2024). Combining piano performance dimensions for
        score difficulty classification. Expert Systems with Applications.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from music21 import converter, note, chord, stream, tempo, key, interval


# ----------------------------------------------------------------------------- 
# Helpers
# -----------------------------------------------------------------------------

def _safe_entropy(xs: np.ndarray) -> float:
    """Shannon entropy in bits over a discrete distribution; 0 if degenerate."""
    if len(xs) == 0:
        return 0.0
    _, counts = np.unique(xs, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def _quantize_ioi(durations: np.ndarray, grid: float = 1/16) -> np.ndarray:
    """Quantize inter-onset intervals to a 16th-note grid for entropy stability."""
    return np.round(durations / grid).astype(int)


def _lerdahl_distance(p1: int, p2: int, key_tonic: int = 0) -> float:
    """
    Simplified Lerdahl pitch-space distance between two MIDI pitches in a key.

    Full Lerdahl space requires region/chord/pitch hierarchies; here we use
    the basic-space component which captures the bulk of cognitive distance:
        dist = |octave shift| + diatonic_distance(p1, p2, key_tonic)
    """
    diatonic = [0, 2, 4, 5, 7, 9, 11]
    def to_scale_deg(p):
        rel = (p - key_tonic) % 12
        return min(range(7), key=lambda i: abs(diatonic[i] - rel))
    return abs(p1 - p2) / 12 + abs(to_scale_deg(p1) - to_scale_deg(p2)) / 7


# -----------------------------------------------------------------------------
# Feature container
# -----------------------------------------------------------------------------

@dataclass
class PieceFeatures:
    piece_id: str

    # density
    notes_per_quarter: float = 0.0
    notes_per_sec_est: float = 0.0
    duration_quarters: float = 0.0
    n_notes: int = 0

    # motoric (per hand)
    rh_range_semitones: int = 0
    lh_range_semitones: int = 0
    rh_jump_mean: float = 0.0
    rh_jump_max: float = 0.0
    rh_jump_p95: float = 0.0
    lh_jump_mean: float = 0.0
    lh_jump_max: float = 0.0
    lh_jump_p95: float = 0.0
    max_simultaneous_span: int = 0     # widest vertical interval anywhere
    hand_independence: float = 0.0     # see __post_init__-style note below

    # polyphony
    n_parts: int = 1
    mean_chord_size: float = 1.0
    p95_chord_size: float = 1.0
    chord_fraction: float = 0.0
    voice_leading_parsimony: float = 0.0  # mean |interval| in melodic motion

    # rhythm
    ioi_entropy_bits: float = 0.0
    syncopation_index: float = 0.0
    tuplet_fraction: float = 0.0
    shortest_note_quarter: float = 1.0

    # harmony
    chromaticism_rate: float = 0.0     # fraction of non-diatonic notes
    modulation_count: int = 0
    mean_lerdahl_step: float = 0.0     # avg harmonic distance between verticals

    # tempo
    mean_bpm: float = 100.0
    tempo_changes: int = 0

    # error flag
    extraction_error: Optional[str] = None


# -----------------------------------------------------------------------------
# Main extraction
# -----------------------------------------------------------------------------

def extract_features(path: Path) -> PieceFeatures:
    """Extract feature vector from one MusicXML file. Never raises; on failure
    returns a PieceFeatures with extraction_error set."""
    feats = PieceFeatures(piece_id=Path(path).stem)
    try:
        s = converter.parse(str(path))
    except Exception as e:
        feats.extraction_error = f"parse: {e}"
        return feats

    parts = list(s.parts)
    feats.n_parts = len(parts)
    all_notes = list(s.flatten().notes)
    feats.n_notes = len(all_notes)

    if not all_notes:
        feats.extraction_error = "empty score"
        return feats

    feats.duration_quarters = float(s.duration.quarterLength or 1)
    feats.notes_per_quarter = feats.n_notes / feats.duration_quarters

    # ---- tempo ----
    mms = s.flatten().getElementsByClass(tempo.MetronomeMark)
    bpms = [m.number for m in mms if m.number]
    feats.mean_bpm = float(np.mean(bpms)) if bpms else 100.0
    feats.tempo_changes = len(bpms)
    feats.notes_per_sec_est = feats.notes_per_quarter * feats.mean_bpm / 60

    # ---- per-hand motoric stats ----
    def _hand_pitches(part_idx: int) -> list[int]:
        if part_idx >= len(parts):
            return []
        out = []
        for n in parts[part_idx].flatten().notes:
            if isinstance(n, note.Note):
                out.append(n.pitch.midi)
            elif isinstance(n, chord.Chord):
                out.extend(p.midi for p in n.pitches)
        return out

    rh_p = _hand_pitches(0)
    lh_p = _hand_pitches(1)

    def _jump_stats(pitches):
        if len(pitches) < 2:
            return 0, 0.0, 0.0, 0.0
        diffs = np.abs(np.diff(pitches))
        return (max(pitches) - min(pitches),
                float(diffs.mean()),
                float(diffs.max()),
                float(np.percentile(diffs, 95)))

    (feats.rh_range_semitones, feats.rh_jump_mean,
     feats.rh_jump_max, feats.rh_jump_p95) = _jump_stats(rh_p)
    (feats.lh_range_semitones, feats.lh_jump_mean,
     feats.lh_jump_max, feats.lh_jump_p95) = _jump_stats(lh_p)

    # Hand independence: low when hands move in lockstep, high when they
    # require different gestural patterns. Computed as 1 - |corr(rh_diffs,
    # lh_diffs)| over an aligned beat grid. Falls back to jump-std product
    # when alignment fails.
    feats.hand_independence = _hand_independence(parts)

    # ---- vertical span ----
    spans = []
    for el in s.flatten().notes:
        if isinstance(el, chord.Chord) and len(el.pitches) >= 2:
            ps = [p.midi for p in el.pitches]
            spans.append(max(ps) - min(ps))
    feats.max_simultaneous_span = max(spans) if spans else 0

    # ---- polyphony ----
    chord_sizes = [len(n.pitches) for n in all_notes if isinstance(n, chord.Chord)]
    feats.mean_chord_size = float(np.mean(chord_sizes)) if chord_sizes else 1.0
    feats.p95_chord_size = float(np.percentile(chord_sizes, 95)) if chord_sizes else 1.0
    feats.chord_fraction = len(chord_sizes) / feats.n_notes

    melodic_intervals = []
    for p in parts:
        ns = [n for n in p.flatten().notes if isinstance(n, note.Note)]
        for a, b in zip(ns, ns[1:]):
            melodic_intervals.append(abs(interval.Interval(a, b).semitones))
    feats.voice_leading_parsimony = (float(np.mean(melodic_intervals))
                                     if melodic_intervals else 0.0)

    # ---- rhythm ----
    durs = np.array([float(n.quarterLength) for n in all_notes if n.quarterLength > 0])
    if len(durs):
        feats.ioi_entropy_bits = _safe_entropy(_quantize_ioi(durs))
        feats.shortest_note_quarter = float(durs.min())
        # tuplet detection: durations that are not powers of 2 of a quarter
        feats.tuplet_fraction = float(np.mean(
            [abs(np.log2(d) - round(np.log2(d))) > 0.05 for d in durs if d > 0]
        ))
        # syncopation: fraction of onsets falling off the beat
        offsets = np.array([float(n.offset) for n in all_notes])
        feats.syncopation_index = float(np.mean(np.abs(offsets - np.round(offsets)) > 0.05))

    # ---- harmony ----
    ks = s.flatten().getElementsByClass(key.KeySignature)
    feats.modulation_count = max(len(ks) - 1, 0)

    try:
        analyzed_key = s.analyze('key')
        tonic_midi = analyzed_key.tonic.midi % 12
        diatonic = {(tonic_midi + d) % 12 for d in [0, 2, 4, 5, 7, 9, 11]}
        chrom = sum(1 for n in all_notes
                    if isinstance(n, note.Note) and (n.pitch.midi % 12) not in diatonic)
        feats.chromaticism_rate = chrom / feats.n_notes
    except Exception:
        feats.chromaticism_rate = 0.0
        tonic_midi = 0

    # mean Lerdahl step between consecutive top-line pitches
    if len(rh_p) >= 2:
        steps = [_lerdahl_distance(a, b, tonic_midi) for a, b in zip(rh_p, rh_p[1:])]
        feats.mean_lerdahl_step = float(np.mean(steps))

    return feats


def _hand_independence(parts) -> float:
    """1 - |Pearson r| of beat-aligned pitch-change series across hands.

    Returns 0 if alignment fails or one hand is empty.
    """
    if len(parts) < 2:
        return 0.0
    try:
        def beat_pitches(part):
            xs = []
            for n in part.flatten().notes:
                p = n.pitch.midi if isinstance(n, note.Note) else (
                    int(np.mean([p.midi for p in n.pitches])) if isinstance(n, chord.Chord) else None)
                if p is not None:
                    xs.append((float(n.offset), p))
            return xs

        rh = beat_pitches(parts[0])
        lh = beat_pitches(parts[1])
        if len(rh) < 5 or len(lh) < 5:
            return 0.0

        max_offset = max(rh[-1][0], lh[-1][0])
        grid = np.arange(0, max_offset, 1.0)  # 1-quarter grid

        def sample(seq, grid):
            offsets = np.array([o for o, _ in seq])
            pitches = np.array([p for _, p in seq])
            idx = np.searchsorted(offsets, grid, side='right') - 1
            return pitches[np.clip(idx, 0, len(pitches) - 1)]

        rh_g, lh_g = sample(rh, grid), sample(lh, grid)
        rh_d, lh_d = np.diff(rh_g), np.diff(lh_g)
        if rh_d.std() < 1e-6 or lh_d.std() < 1e-6:
            return 1.0
        return float(1 - abs(np.corrcoef(rh_d, lh_d)[0, 1]))
    except Exception:
        return 0.0


def extract_dataset(score_paths: list[Path]) -> "pd.DataFrame":
    """Extract features for many scores, return tidy DataFrame."""
    import pandas as pd
    rows = [asdict(extract_features(p)) for p in score_paths]
    return pd.DataFrame(rows)
