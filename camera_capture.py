# camera_capture.py
import cv2

# Function to capture video from the camera
def capture_video():
    cap = cv2.VideoCapture(0)  # Using the default camera (change to IP camera URL if needed)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Create background subtractor for vehicle detection
    fgbg = cv2.createBackgroundSubtractorMOG2()

    vehicle_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Apply background subtraction to detect vehicles
        fgmask = fgbg.apply(frame)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Ignore small contours (noise)
                x, y, w, h = cv2.boundingRect(contour)
                vehicle_count += 1  # Count detected vehicles
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box

        # Display the current video frame with detected vehicles
        cv2.imshow("Vehicle Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return vehicle_count  # Return the count of detected vehicles

if __name__ == "__main__":
    capture_video()
