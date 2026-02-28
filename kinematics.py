import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

def calculate_angle(p1, p2, p3):
    """Calculates the angle in radians between three 3D points."""
    v1 = np.array([p1.x, p1.y, p1.z])
    v2 = np.array([p2.x, p2.y, p2.z])
    v3 = np.array([p3.x, p3.y, p3.z])
    
    vectorA = v1 - v2
    vectorB = v3 - v2
    
    cosine_angle = np.dot(vectorA, vectorB) / (np.linalg.norm(vectorA) * np.linalg.norm(vectorB))
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return angle_rad

def get_leap_state(hand_landmarks):
    """
    Extracts MediaPipe landmarks and calculates the bend angles for the LEAP URDF.
    Returns a dictionary of angles ready to be sent to Isaac Lab.
    """
    lm = hand_landmarks.landmark
    
    # --- INDEX FINGER (LEAP Fingers 1: joints 0, 2, 3) ---
    index_mcp = calculate_angle(lm[0], lm[5], lm[6]) # Base knuckle
    index_pip = calculate_angle(lm[5], lm[6], lm[7]) # Middle joint
    index_dip = calculate_angle(lm[6], lm[7], lm[8]) # Tip joint
    
    # --- MIDDLE FINGER (LEAP Finger 2: joints 4, 6, 7) ---
    middle_mcp = calculate_angle(lm[0], lm[9], lm[10])
    middle_pip = calculate_angle(lm[9], lm[10], lm[11])
    middle_dip = calculate_angle(lm[10], lm[11], lm[12])
    
    # --- RING FINGER (LEAP Finger 3: joints 8, 10, 11) ---
    ring_mcp = calculate_angle(lm[0], lm[13], lm[14])
    ring_pip = calculate_angle(lm[13], lm[14], lm[15])
    ring_dip = calculate_angle(lm[14], lm[15], lm[16])
    
    # --- THUMB (LEAP Thumb: joints 12, 14, 15) ---
    # The thumb moves differently, so we use slightly different anchor points
    thumb_base = calculate_angle(lm[0], lm[1], lm[2])
    thumb_pip = calculate_angle(lm[1], lm[2], lm[3])
    thumb_dip = calculate_angle(lm[2], lm[3], lm[4])

    # Package all angles into a dictionary matching your URDF structure
    leap_state = {
        "index": {"mcp": index_mcp, "pip": index_pip, "dip": index_dip},
        "middle": {"mcp": middle_mcp, "pip": middle_pip, "dip": middle_dip},
        "ring": {"mcp": ring_mcp, "pip": ring_pip, "dip": ring_dip},
        "thumb": {"base": thumb_base, "pip": thumb_pip, "dip": thumb_dip}
    }
    
    return leap_state