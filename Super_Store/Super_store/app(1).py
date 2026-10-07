import streamlit as st
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Outlet Sales Prediction App",
    page_icon="🛍️",
    layout="centered"
)

# --- LOAD TRAINED MODEL ---
@st.cache_resource
def load_model():
    try:
        with open("knn_regression_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("❌ 'knn_regression_model.pkl' not found. Please ensure the model file is in the same directory.")
        return None

model = load_model()

# --- APP INTERFACE ---
st.title("🛍️ Big Mart Outlet Sales Prediction")
st.write("Enter the item and outlet characteristics below to predict total outlet sales using your trained KNN Regression model.")

st.markdown("---")

if model is not None:
    # --- INPUT FIELDS ---
    st.subheader("📦 Item Details")
    col1, col2 = st.columns(2)
    
    with col1:
        item_weight = st.number_input("Item Weight", min_value=0.0, max_value=30.0, value=12.8)
        item_visibility = st.number_input("Item Visibility", min_value=0.0, max_value=1.0, value=0.06, format="%.6f")
    
    with col2:
        item_fat_content = st.selectbox("Item Fat Content", options=["Low Fat", "Regular"])
        item_mrp = st.number_input("Item MRP", min_value=0.0, max_value=500.0, value=141.0)
        
    item_type = st.selectbox(
        "Item Type",
        options=[
            "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods", 
            "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household", 
            "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"
        ]
    )

    st.markdown("---")
    st.subheader("🏪 Outlet Details")
    col3, col4 = st.columns(2)
    
    with col3:
        outlet_identifier = st.selectbox(
            "Outlet Identifier",
            options=["OUT010", "OUT013", "OUT017", "OUT018", "OUT019", "OUT027", "OUT035", "OUT045", "OUT046", "OUT049"]
        )
        outlet_establishment_year = st.number_input("Outlet Establishment Year", min_value=1950, max_value=2030, value=1997)
    
    with col4:
        outlet_size = st.selectbox("Outlet Size", options=["Small", "Medium", "High"])
        outlet_location_type = st.selectbox("Outlet Location Type", options=["Tier 1", "Tier 2", "Tier 3"])
        
    outlet_type = st.selectbox(
        "Outlet Type",
        options=["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
    )

    st.markdown("---")

    # --- ENCODING & PREPROCESSING (Matching your Jupyter Notebook) ---
    # 1. Map Fat Content (Low Fat: 0, Regular: 1)
    fat_encoded = 1 if item_fat_content == "Regular" else 0

    # 2. Map Outlet Size (Small: 0, Medium: 1, High: 2)
    size_map = {"Small": 0.0, "Medium": 1.0, "High": 2.0}
    size_encoded = size_map[outlet_size]

    # 3. Map Outlet Location Type (Tier 1: 0, Tier 2: 1, Tier 3: 2)
    loc_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
    loc_encoded = loc_map[outlet_location_type]

    # 4. Item Type Dummy Encoding (15 columns, dropped first category 'Baking Goods')
    item_types_list = [
        "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods", "Fruits and Vegetables", 
        "Hard Drinks", "Health and Hygiene", "Household", "Meat", "Others", "Seafood", 
        "Snack Foods", "Soft Drinks", "Starchy Foods"
    ]
    item_type_encoded = [True if item_type == t else False for t in item_types_list]

    # 5. Outlet Identifier Dummy Encoding (10 columns)
    outlets_list = ["OUT010", "OUT013", "OUT017", "OUT018", "OUT019", "OUT027", "OUT035", "OUT045", "OUT046", "OUT049"]
    outlet_id_encoded = [True if outlet_identifier == o else False for o in outlets_list]

    # 6. Outlet Type Dummy Encoding (3 columns, dropped first category 'Grocery Store')
    outlet_types_list = ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
    outlet_type_encoded = [True if outlet_type == ot else False for ot in outlet_types_list]

    # --- COMPILE 35 FEATURES LIST ---
    features = [
        item_weight, 
        fat_encoded, 
        item_visibility, 
        item_mrp, 
        outlet_establishment_year, 
        size_encoded, 
        loc_encoded
    ] + item_type_encoded + outlet_id_encoded + outlet_type_encoded

    # Convert to array for prediction
    final_features = np.array([features])

    # --- PREDICTION BUTTON ---
    if st.button("🔮 Predict Outlet Sales", type="primary"):
        # Make the prediction using your trained model
        prediction = model.predict(final_features)
        
        # Display the result
        st.balloons()
        st.success(f"### Predicted Item Outlet Sales: **${prediction[0]:,.2f}**")
