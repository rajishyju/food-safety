import pickle
import pandas as pd
import streamlit as st

# 1. Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Try loading scaler (if it exists)
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    scaler = None

# 3. Preprocessing and Prediction Function
def preprocess_and_predict(features):
    # Convert to single-row DataFrame
    input_data = pd.DataFrame([features])
    
    # Apply One-Hot Encoding (Do NOT use drop_first=True on a single row, 
    # or it will delete the single selected category! Filtering with required_columns handles it.)
    input_data = pd.get_dummies(input_data, columns=["product_type", "packaging_type"])

    # Get the exact column names the model was trained on
    required_columns = model.feature_names_in_

    # Align columns: adds missing dummy columns as 0, and drops any extra base categories
    input_data = input_data.reindex(columns=required_columns, fill_value=0)

    # Apply scaling if scaler exists
    if scaler is not None:
        input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)
    
    # Safely get probability if the model supports predict_proba
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[:, 1][0]
    else:
        probability = None
        
    return prediction[0], probability


# --- UI ---
st.title("SHELF LIFE PREDICTION APP")
st.write("Enter the product details below:")

shelf_life_days = st.number_input("shelf_life_days", value=0, step=1)
storage_temperature_C = st.number_input("storage_temperature_C", value=0.0, step=0.5)
humidity_level_percent = st.number_input("humidity_level_percent", value=0.0, step=0.5)
moisture_content = st.number_input("moisture_content", value=0.0, step=0.1)
preservative_level = st.number_input("preservative_level", value=0.0, step=0.1)
pH_level = st.number_input("pH_level", value=7.0, step=0.1)
inspection_score_0_100 = st.number_input("inspection_score_0_100", value=0, step=1)
customer_rating_1_5 = st.selectbox("customer_rating_1_5", [1, 2, 3, 4, 5])

product_type = st.selectbox(
    "product_type", 
    options=["Dairy", "Ready-to-Eat", "Snacks", "Canned Food", "Frozen Food", "Juice"]
)

packaging_type = st.selectbox(
    "packaging_type", 
    options=["Metal", "Plastic", "Paper", "Tetra Pack","Vacuum"]
)

# --- Prediction Button ---
if st.button("Predict"):
    features = {
        "shelf_life_days": shelf_life_days,
        "storage_temperature_C": storage_temperature_C,
        "humidity_level_percent": humidity_level_percent,
        "moisture_content": moisture_content,
        "preservative_level": preservative_level,
        "pH_level": pH_level,
        "product_type": product_type,
        "packaging_type": packaging_type,
        "inspection_score_0_100": inspection_score_0_100,
        "customer_rating_1_5": customer_rating_1_5
    }
    
    prediction, probability = preprocess_and_predict(features)
    
    # Swapped so Approved is green (st.success) and Rejected is red (st.error)
    if prediction == 1:
        if probability is not None:
            st.success(f"Approved (Confidence: {probability:.2f})")
        else:
            st.success("Approved")
    else:
        if probability is not None:
            st.error(f"Rejected (Confidence: {1 - probability:.2f})")
        else:
            st.error("Rejected")