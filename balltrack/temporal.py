"""Offline beam search over detector observations; missing states never create boxes."""

import math

import numpy as np


def select_trajectory(forward, backward, scale=1.0, beam_size=32):
    if len(forward) != len(backward):
        raise ValueError("Pass lengths differ")
    # score, last position/time, preceding position/time, selected observations
    beams = [(0.0, None, None, [])]
    for frame, (left, right) in enumerate(zip(forward, backward)):
        candidates = []
        for obs in (left, right):
            if obs is None:
                continue
            if not any(
                np.linalg.norm(
                    (obs[0][:2] + obs[0][2:]) / 2 - (v[0][:2] + v[0][2:]) / 2
                )
                < 2 * scale
                for v in candidates
            ):
                candidates.append(obs)
        expanded = []
        for score, last, previous, path in beams:
            expanded.append((score - 0.04, last, previous, path + [None]))
            for obs in candidates:
                box, conf, _ = obs
                center = (box[:2] + box[2:]) / 2
                reward = 2.0 + math.log(max(conf, 0.01))
                penalty = 0.0
                if last is not None:
                    elapsed = frame - last[0]
                    velocity = (
                        np.zeros(2)
                        if previous is None
                        else (last[1] - previous[1]) / (last[0] - previous[0])
                    )
                    prediction = last[1] + velocity * elapsed
                    uncertainty = (
                        max(14 * scale, float(np.max(box[2:] - box[:2])) * 0.8)
                        + 0.6 * scale * elapsed * elapsed
                    )
                    penalty = (
                        0.7 * (np.linalg.norm(center - prediction) / uncertainty) ** 2
                    )
                expanded.append(
                    (score + reward - penalty, (frame, center), last, path + [obs])
                )
        expanded.sort(key=lambda x: x[0], reverse=True)
        # Keep diverse last-observation states, not 32 equivalent copies of one track.
        beams = []
        keys = set()
        for item in expanded:
            last = item[1]
            previous = item[2]
            key = (
                (
                    None
                    if last is None
                    else (last[0], *np.round(last[1] / (3 * scale)).astype(int))
                ),
                None if previous is None else previous[0],
            )
            if key in keys:
                continue
            keys.add(key)
            beams.append(item)
            if len(beams) >= beam_size:
                break
    return max(beams, key=lambda x: x[0])[3]


def select_anchored_trajectory(forward, backward, scale=1.0, max_gap=15):
    """A selected identity cannot be replaced by an unrelated reverse-pass track.

    Reverse evidence is usable only inside a short gap bracketed by measured
    forward observations and near their motion corridor. No interpolated point
    itself is ever returned as a detection.
    """
    allowed = [None] * len(forward)
    for i, obs in enumerate(backward):
        if obs is None or forward[i] is not None:
            continue
        left = next(
            (
                j
                for j in range(i - 1, max(-1, i - max_gap - 1), -1)
                if forward[j] is not None
            ),
            None,
        )
        right = next(
            (
                j
                for j in range(i + 1, min(len(forward), i + max_gap + 1))
                if forward[j] is not None
            ),
            None,
        )
        if left is None or right is None or right - left > max_gap:
            continue
        a = (forward[left][0][:2] + forward[left][0][2:]) / 2
        b = (forward[right][0][:2] + forward[right][0][2:]) / 2
        expected = a + (b - a) * (i - left) / (right - left)
        center = (obs[0][:2] + obs[0][2:]) / 2
        width = max(obs[0][2:] - obs[0][0:2])
        if np.linalg.norm(center - expected) <= max(15 * scale, width * 2):
            allowed[i] = obs
    return select_trajectory(forward, allowed, scale)
