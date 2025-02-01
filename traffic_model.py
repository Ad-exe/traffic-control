# traffic_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression

# Function to load and train a traffic signal prediction model
def train_model():
    # Load the data (replace with your dataset)
    data = pd.read_csv('traffic_data.csv')

    # Define features and target
    X = data[['Vehicle_Count']]
    y = data['Signal_Duration']

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    return model

# Function to predict signal duration based on vehicle count
def predict_signal_duration(vehicle_count, model):
    predicted_duration = model.predict([[vehicle_count]])
    return predicted_duration[0]

# Example usage of the model
if __name__ == "__main__":
    model = train_model()

    # Example: predict signal duration for 60 vehicles
    vehicle_count = 60
    predicted_duration = predict_signal_duration(vehicle_count, model)
    print(f"Predicted Signal Duration for {vehicle_count} vehicles: {predicted_duration:.2f} seconds")
