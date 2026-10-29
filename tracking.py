"""Live hand tracking with MediaPipe and LEAP-hand joint angles.

Reads a webcam, detects hands, and for the selected hand (left by default) draws the
skeleton and a heads-up display of the 16 joint angles from kinematics.get_leap_state().
Press ESC in the video window to exit.
"""
import argparse
import sys
import time

import cv2
import mediapipe as mp

import kinematics

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

MAX_READ_FAILURES = 30  # consecutive failed reads before giving up


def parse_args():
    parser = argparse.ArgumentParser(description="Hand tracking with LEAP-hand joint angles")
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index (0 is usually the built-in webcam; try 1 or 2 for USB cameras)")
    parser.add_argument("--hand", choices=["Left", "Right"], default="Left",
                        help="which hand to track (default: Left)")
    return parser.parse_args()


def draw_hud(image, hand_landmarks, leap_angles, hand_label):
    """Draw the skeleton, the joint angles and a status line on the frame."""
    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    y_pos = 375  # starting Y position on the screen
    for finger, joints in leap_angles.items():
        # The thumb uses 'base' instead of 'mcp', so we check for both
        mcp_val = joints.get("mcp", joints.get("base"))
        info = (f"{finger.upper()} - ABD: {joints['abd']:.2f} | MCP: {mcp_val:.2f} | "
                f"PIP: {joints['pip']:.2f} | DIP: {joints['dip']:.2f}")
        cv2.putText(image, info, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2)
        y_pos += 30

    cv2.putText(image, f"{hand_label} Hand Locked", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Index Z-Depth: {index_tip.z:.2f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


def main():
    args = parse_args()
    window_name = f"{args.hand} Hand Teleoperation Tracker"

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera {args.camera}. Try --camera 1 or 2.", file=sys.stderr)
        return 1

    failures = 0
    try:
        with mp_hands.Hands(
            max_num_hands=2,  # scan for 2 so the other hand can be actively ignored
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:
            print("Camera active. Press ESC in the video window to exit.")

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    failures += 1
                    if failures >= MAX_READ_FAILURES:
                        print("Error: the camera stopped delivering frames.", file=sys.stderr)
                        return 1
                    time.sleep(0.01)
                    continue
                failures = 0

                # Flip horizontally for a natural "mirror" feel
                image = cv2.flip(image, 1)

                # MediaPipe expects RGB, OpenCV delivers BGR
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks, results.multi_handedness
                    ):
                        hand_label = handedness.classification[0].label
                        if hand_label == args.hand:
                            leap_angles = kinematics.get_leap_state(hand_landmarks)
                            draw_hud(image, hand_landmarks, leap_angles, hand_label)

                cv2.imshow(window_name, image)
                if cv2.waitKey(5) & 0xFF == 27:  # ESC
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
