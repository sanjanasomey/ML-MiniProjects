import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Load Pickle Files
# -----------------------------
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    dv = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("features.pkl", "rb") as f:
    features = pickle.load(f)

# -----------------------------
# Title
# -----------------------------
st.title("🏠 House Price Prediction System")
st.markdown("Predict the estimated house price using Machine Learning.")

st.divider()

# -----------------------------
# Input Form
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    state = st.selectbox(
        "State",
        ["maharashtra", "karnataka", "gujarat", "delhi", "tamil nadu"]
    )

    city = st.text_input("City")

    property_type = st.selectbox(
        "Property Type",
        ["Apartment", "Independent House", "Villa"]
    )

    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=10,
        value=2
    )

    size_sqft = st.number_input(
        "Size (Sq.Ft)",
        min_value=300,
        max_value=10000,
        value=1200
    )

    price_sqft = st.number_input(
        "Price Per Sq.Ft",
        min_value=1000,
        value=5000
    )

    year = st.number_input(
        "Year Built",
        min_value=1980,
        max_value=2025,
        value=2018
    )

    furnished = st.selectbox(
        "Furnished Status",
        [
            "Unfurnished",
            "Semi-furnished",
            "Furnished"
        ]
    )

    floor = st.number_input(
        "Floor Number",
        min_value=0,
        value=2
    )

    total_floor = st.number_input(
        "Total Floors",
        min_value=1,
        value=10
    )

with col2:

    age = st.number_input(
        "Age of Property",
        min_value=0,
        value=5
    )

    school = st.number_input(
        "Nearby Schools",
        min_value=0,
        value=5
    )

    hospital = st.number_input(
        "Nearby Hospitals",
        min_value=0,
        value=3
    )

    transport = st.selectbox(
        "Public Transport",
        ["Low", "Medium", "High"]
    )

    parking = st.selectbox(
        "Parking Space",
        ["Yes", "No"]
    )

    security = st.selectbox(
        "Security",
        ["No", "Yes"]
    )

    amenities = st.text_input("Amenities")

    facing = st.selectbox(
        "Facing",
        ["South", "East", "West", "North"]
    )

    owner = st.text_input("Owner Type")

    availability = st.text_input("Availability Status")
# ------------------------------------------
# Encode Ordinal Features
# ------------------------------------------

ordinal_df = pd.DataFrame({
    "Property_Type": [property_type],
    "Furnished_Status": [furnished],
    "Public_Transport_Accessibility": [transport],
    "Facing": [facing],
    "Security": [security]
})

ordinal_encoded = encoder.transform(ordinal_df)

property_type = ordinal_encoded[0][0]
furnished = ordinal_encoded[0][1]
transport = ordinal_encoded[0][2]
facing = ordinal_encoded[0][3]
security = ordinal_encoded[0][4]


# ------------------------------------------
# Prediction Button
# ------------------------------------------

if st.button("Predict House Price"):

    input_data = {
        "State": state.lower(),
        "City": city.lower(),
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size_sqft,
        "Price_per_SqFt": price_sqft,
        "Year_Built": year,
        "Furnished_Status": furnished,
        "Floor_No": floor,
        "Total_Floors": total_floor,
        "Age_of_Property": age,
        "Nearby_Schools": school,
        "Nearby_Hospitals": hospital,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking.lower(),
        "Security": security,
        "Amenities": amenities.lower(),
        "Facing": facing,
        "Owner_Type": owner.lower(),
        "Availability_Status": availability.lower()
    }

    try:

        # Convert dictionary into vector
        X = dv.transform([input_data])

        # Predict
        prediction = model.predict(X)[0]

        st.success("Prediction Successful")

        st.metric(
            label="Estimated House Price",
            value=f"₹ {prediction:.2f} Lakhs"
        )

        st.balloons()

    except Exception as e:

        st.error("Prediction Failed")
        st.error(e)
# =====================================================
# Sidebar
# =====================================================

st.sidebar.title("🏠 House Price Prediction")

st.sidebar.markdown("""
### Machine Learning Model
- Decision Tree Regressor

### Features
- 20 Input Features
- DictVectorizer
- Ordinal Encoding
- Streamlit Web App

### Technologies
- Python
- Scikit-Learn
- Pandas
- Streamlit
""")

st.sidebar.divider()

st.sidebar.info(
    "Enter the property details and click **Predict House Price**."
)

# =====================================================
# Show Input Data
# =====================================================

with st.expander("📋 View Input Data"):

    st.write(input_data if 'input_data' in locals() else "No data entered yet.")

# =====================================================
# Footer
# =====================================================

st.divider()

st.markdown(
    """
    <center>

    ### 🏠 House Price Prediction System

    Developed using **Streamlit** and **Scikit-Learn**

    Decision Tree Regressor Model

    </center>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# Hide Streamlit Menu & Footer
# =====================================================

hide_streamlit_style = """
<style>

#MainMenu {
visibility:hidden;
}

footer {
visibility:hidden;
}

header {
visibility:hidden;
}

</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)