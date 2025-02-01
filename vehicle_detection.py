import cv2
import os
import pandas as pd

# Load pre-trained car detection model
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

# Open camera (0 = default webcam)
cap = cv2.VideoCapture(0)

# Create a list to store vehicle count data
vehicle_count_data = []

# Create a folder to save detected vehicle images
output_folder = "detections"  # Save in a subfolder called "detections"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0  # Frame counter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)  # Detect cars

    vehicle_count = len(cars)  # Count the number of detected vehicles
    vehicle_count_data.append({"frame": frame_count, "car_count": vehicle_count})

    # Draw rectangles around detected vehicles
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with detected vehicles
    cv2.putText(frame, f"Cars Detected: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Vehicle Detection", frame)

    # Save image when a vehicle is detected
    if vehicle_count > 0:
        img_filename = f"{output_folder}/frame_{frame_count}.jpg"
        cv2.imwrite(img_filename, frame)

    # Press 'q' to stop the camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()

# Save vehicle count data to CSV
df = pd.DataFrame(vehicle_count_data)
df.to_csv("vehicle_count_results.csv", index=False)
print("Detection results saved to vehicle_count_results.csv")
