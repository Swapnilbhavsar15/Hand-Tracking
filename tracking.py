import cv2
import mediapipe as mp
import kinematics

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 0 is usually the built-in webcam. Change to 1 or 2 for your USB webcam!
cam_index = 0  
cap = cv2.VideoCapture(cam_index)

# Start MediaPipe Hands
with mp_hands.Hands(
    max_num_hands=2, # scan for 2 so we can actively ignore the right hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    print("Camera active. Press 'ESC' in the video window to exit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip the image horizontally for a natural "mirror" feel
        image = cv2.flip(image, 1)
        
        # Convert BGR to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to find hands
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                # Extract the "Left" or "Right" label
                hand_label = handedness.classification[0].label
                
                # Execute tracking ONLY if it's the Left hand
                if hand_label == "Left":
                    
                    # Draw the skeletal map on your hand
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Grab the 3D data for the tip of the Index Finger (Landmark 8)
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    # Call the master function to get all LEAP hand angles
                    leap_angles = kinematics.get_leap_state(hand_landmarks)
                    
                    # Print the live data payload to the terminal
                    print(f"Index PIP: {leap_angles['index']['pip']:.2f} | Middle PIP: {leap_angles['middle']['pip']:.2f} |  Ring PIP: {leap_angles['ring']['pip']:.2f} | Thumb PIP: {leap_angles['thumb']['pip']:.2f} ", flush=True)                    
                    # Display confirmation on the screen
                    cv2.putText(image, "Left Hand Locked", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Index Z-Depth: {index_tip.z:.2f}", (10, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the video feed
        cv2.imshow('Left Hand Teleoperation Tracker', image)
        
        # Press the 'ESC' key to close the window
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()