# main.py
from camera_capture import capture_video
from traffic_model import train_model, predict_signal_duration

# Capture vehicle count from the camera
vehicle_count = capture_video()

# Train the model
model = train_model()

# Predict the signal duration based on the vehicle count
predicted_duration = predict_signal_duration(vehicle_count, model)

print(f"Predicted Signal Duration for {vehicle_count} vehicles: {predicted_duration:.2f} seconds")
