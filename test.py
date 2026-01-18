import joblib
import numpy as np
# Load model
loaded_model = joblib.load("model.pkl")


new_input = np.array([[ 
    2,     # Junction
    12345, # ID 
    9,     # Hour
    15,    # Day
    7,     # Month
    1,     # Weekday (Tuesday)
    0,     # Is_Weekend
    420,   # Lag_1
    390,   # Lag_24
    410,   # Lag_168
    415,   # Roll_Mean_3
    405,   # Roll_Mean_6
    380    # Roll_Mean_24
]])

print("Model loaded and predictions done")
# Predict using loaded model
predictions = loaded_model.predict(new_input)

print(predictions)

# print(loaded_model.feature_names_in_)




