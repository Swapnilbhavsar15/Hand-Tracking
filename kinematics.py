"""Joint-angle extraction from MediaPipe hand landmarks for the 16-DoF LEAP hand.

MediaPipe returns 21 landmarks per hand (0 = wrist, 1-4 = thumb, 5-8 = index,
9-12 = middle, 13-16 = ring, 17-20 = pinky). This module turns them into the
bend (MCP/PIP/DIP) and spread (abduction) angles used by the LEAP hand model.

All angles are in radians and are computed from MediaPipe's normalized landmark
coordinates (see the README for the limitations of that choice).
"""
import numpy as np


def _angle_between(vector_a, vector_b):
    """Unsigned angle in radians between two vectors."""
    cosine_angle = np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )
    return np.arccos(np.clip(cosine_angle, -1.0, 1.0))


def calculate_angle(p1, p2, p3):
    """Calculates the hinge bend in radians at p2, between the points p1-p2-p3."""
    v1 = np.array([p1.x, p1.y, p1.z])
    v2 = np.array([p2.x, p2.y, p2.z])
    v3 = np.array([p3.x, p3.y, p3.z])
    return _angle_between(v1 - v2, v3 - v2)


def calculate_spread(base1, tip1, base2, tip2):
    """Calculates the side-to-side spread (abduction) between two finger vectors."""
    # Vector 1 (the anchor finger)
    v1 = np.array([tip1.x - base1.x, tip1.y - base1.y, tip1.z - base1.z])
    # Vector 2 (the target finger)
    v2 = np.array([tip2.x - base2.x, tip2.y - base2.y, tip2.z - base2.z])
    return _angle_between(v1, v2)


def get_leap_state(hand_landmarks):
    """
    Extracts MediaPipe landmarks and calculates all 16 bend/spread angles
    for the LEAP hand. Returns a dictionary of angles ready for the simulator.
    """
    lm = hand_landmarks.landmark

    # --- INDEX FINGER (Anchor: Middle Finger) ---
    index_abd = calculate_spread(lm[9], lm[10], lm[5], lm[6])
    index_mcp = calculate_angle(lm[0], lm[5], lm[6])
    index_pip = calculate_angle(lm[5], lm[6], lm[7])
    index_dip = calculate_angle(lm[6], lm[7], lm[8])

    # --- MIDDLE FINGER (Anchor: Wrist-to-Knuckle Line) ---
    middle_abd = calculate_spread(lm[0], lm[9], lm[9], lm[10])
    middle_mcp = calculate_angle(lm[0], lm[9], lm[10])
    middle_pip = calculate_angle(lm[9], lm[10], lm[11])
    middle_dip = calculate_angle(lm[10], lm[11], lm[12])

    # --- RING FINGER (Anchor: Middle Finger) ---
    ring_abd = calculate_spread(lm[9], lm[10], lm[13], lm[14])
    ring_mcp = calculate_angle(lm[0], lm[13], lm[14])
    ring_pip = calculate_angle(lm[13], lm[14], lm[15])
    ring_dip = calculate_angle(lm[14], lm[15], lm[16])

    # --- THUMB (Anchor: Index Finger) ---
    thumb_abd = calculate_spread(lm[5], lm[6], lm[1], lm[2])
    thumb_base = calculate_angle(lm[0], lm[1], lm[2])
    thumb_pip = calculate_angle(lm[1], lm[2], lm[3])
    thumb_dip = calculate_angle(lm[2], lm[3], lm[4])

    # Package all 16 angles into the master dictionary
    return {
        "index": {"abd": index_abd, "mcp": index_mcp, "pip": index_pip, "dip": index_dip},
        "middle": {"abd": middle_abd, "mcp": middle_mcp, "pip": middle_pip, "dip": middle_dip},
        "ring": {"abd": ring_abd, "mcp": ring_mcp, "pip": ring_pip, "dip": ring_dip},
        "thumb": {"abd": thumb_abd, "base": thumb_base, "pip": thumb_pip, "dip": thumb_dip},
    }
