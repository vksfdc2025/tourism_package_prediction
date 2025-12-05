
# =============================================================================
# STREAMLIT APPLICATION FOR TOURISM PACKAGE PURCHASE PREDICTION
# This application provides an interactive interface for model inference
# =============================================================================

import streamlit as st  # Import Streamlit for creating web applications
import pandas as pd     # Import pandas for data manipulation
import joblib           # Import joblib for loading the trained model
import os               # Import os for environment variable access
from huggingface_hub import HfApi  # Import HfApi for downloading models from Hugging Face

# Configure Streamlit page settings
st.set_page_config(
    page_title="Tourism Package Purchase Predictor",  # Title shown in browser tab
    page_icon="✈️",                                  # Icon for the browser tab
    layout="centered"                                # Layout of the page content
)

# Get Hugging Face username from environment variables (set in Colab config)
# This is used to construct the model repository ID dynamically
HF_USERNAME = "vksfdc2024"

# Define the Hugging Face model repository ID
HF_MODEL_REPO = f"{HF_USERNAME}/tourism-package-model"

# =============================================================================
# MODEL LOADING FUNCTION
# Caches the model to avoid reloading on each interaction, improving performance
# =============================================================================

@st.cache_resource(show_spinner="Loading model...") # Cache the model for efficient reuse
def load_model():
    api = HfApi(token=os.getenv("HF_TOKEN")) # Initialize HfApi with token
    # Define the local path where the model will be saved
    model_local_path = "best_tourism_model_v1.joblib"

    try:
        # Download the model file from Hugging Face Hub
        api.hf_hub_download(
            repo_id=HF_MODEL_REPO,           # Repository ID of the model
            filename="best_tourism_model_v1.joblib", # Name of the file in the repo
            local_dir=".",                   # Directory to save the file locally
            local_dir_use_symlinks=False     # Avoid symlinks for better compatibility
        )
        # Load the model using joblib
        model = joblib.load(model_local_path)
        return model # Return the loaded model
    except Exception as e:
        st.error(f"Error loading model: {e}") # Display error if model loading fails
        st.stop() # Stop the Streamlit app if model cannot be loaded

# Load the model at the start of the application
model = load_model()

# =============================================================================
# CATEGORICAL FEATURE MAPPINGS
# These dictionaries map user-friendly labels to numerical values as used by LabelEncoder
# during model training. The order is crucial for consistent predictions.
# =============================================================================

# Assuming LabelEncoder sorts alphabetically for the given unique values
TYPEOFCONTACT_MAP = {'Company Invited': 0, 'Self Inquiry': 1}
OCCUPATION_MAP = {'Freelancer': 0, 'Other': 1, 'Salaried': 2, 'Small Business': 3}
GENDER_MAP = {'Female': 0, 'Male': 1}
PRODUCTPITCHED_MAP = {'Basic': 0, 'Deluxe': 1, 'King': 2, 'Resort': 3, 'Standard': 4, 'Super Deluxe': 5}
MARITALSTATUS_MAP = {'Divorced': 0, 'Married': 1, 'Single': 2}
DESIGNATION_MAP = {'AVP': 0, 'Executive': 1, 'Manager': 2, 'Senior Manager': 3, 'VP': 4}

# Reverse mappings for display purposes (optional, but good for debugging/user feedback)
TYPEOFCONTACT_REV_MAP = {v: k for k, v in TYPEOFCONTACT_MAP.items()}
OCCUPATION_REV_MAP = {v: k for k, v in OCCUPATION_MAP.items()}
GENDER_REV_MAP = {v: k for k, v in GENDER_MAP.items()}
PRODUCTPITCHED_REV_MAP = {v: k for k, v in PRODUCTPITCHED_MAP.items()}
MARITALSTATUS_REV_MAP = {v: k for k, v in MARITALSTATUS_MAP.items()}
DESIGNATION_REV_MAP = {v: k for k, v in DESIGNATION_MAP.items()}

# =============================================================================
# STREAMLIT UI COMPONENTS AND PREDICTION LOGIC
# =============================================================================

st.title("✈️ Tourism Package Purchase Predictor")
st.write("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

with st.form("prediction_form"): # Create a form for user inputs
    st.header("Customer Information")

    # Input fields for all features. The order should match X_train columns.
    # 'Unnamed: 0' is likely an index artifact; setting it to 0 is reasonable.
    unnamed_0 = st.number_input("Unnamed: 0 (Internal ID, set to 0)", value=0, format="%d")
    age = st.slider("Age", 18, 80, 30)
    typeofcontact_option = st.selectbox("Type of Contact", list(TYPEOFCONTACT_MAP.keys()))
    citytier = st.selectbox("City Tier", [1, 2, 3])
    durationofpitch = st.slider("Duration of Pitch (minutes)", 0, 60, 10)
    occupation_option = st.selectbox("Occupation", list(OCCUPATION_MAP.keys()))
    gender_option = st.selectbox("Gender", list(GENDER_MAP.keys()))
    numberofpersonvisiting = st.slider("Number of People Visiting (including customer)", 1, 10, 2)
    numberoffollowups = st.slider("Number of Follow-ups", 0, 10, 3)
    productpitched_option = st.selectbox("Product Pitched", list(PRODUCTPITCHED_MAP.keys()))
    preferredpropertystar = st.slider("Preferred Property Star (1-5)", 1, 5, 3)
    maritalstatus_option = st.selectbox("Marital Status", list(MARITALSTATUS_MAP.keys()))
    numberoftrips = st.slider("Number of Trips Annually", 0, 50, 5)
    passport = st.selectbox("Has Passport?", ["No", "Yes"]) # Map to 0/1
    pitchsatisfactionscore = st.slider("Pitch Satisfaction Score (1-5)", 1, 5, 3)
    owncar = st.selectbox("Owns Car?", ["No", "Yes"]) # Map to 0/1
    numberofchildrenvisiting = st.slider("Number of Children Visiting (<5 years old)", 0, 5, 0)
    designation_option = st.selectbox("Designation", list(DESIGNATION_MAP.keys()))
    monthlyincome = st.number_input("Monthly Income", min_value=0.0, value=25000.0, step=1000.0)

    submitted = st.form_submit_button("Predict Purchase") # Submit button for the form

    if submitted: # Actions to take when the form is submitted
        # Convert user inputs to the numerical format expected by the model
        typeofcontact_encoded = TYPEOFCONTACT_MAP[typeofcontact_option]
        occupation_encoded = OCCUPATION_MAP[occupation_option]
        gender_encoded = GENDER_MAP[gender_option]
        productpitched_encoded = PRODUCTPITCHED_MAP[productpitched_option]
        maritalstatus_encoded = MARITALSTATUS_MAP[maritalstatus_option]
        designation_encoded = DESIGNATION_MAP[designation_option]
        passport_encoded = 1 if passport == "Yes" else 0
        owncar_encoded = 1 if owncar == "Yes" else 0

        # Create a DataFrame from the collected inputs
        # Ensure the column names and order match the training data (X_train)
        input_data = pd.DataFrame([[unnamed_0, age, typeofcontact_encoded, citytier, durationofpitch,
                                      occupation_encoded, gender_encoded, numberofpersonvisiting, numberoffollowups,
                                      productpitched_encoded, preferredpropertystar, maritalstatus_encoded,
                                      numberoftrips, passport_encoded, pitchsatisfactionscore, owncar_encoded,
                                      numberofchildrenvisiting, designation_encoded, monthlyincome]],
                                    columns=['Unnamed: 0', 'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch',
                                             'Occupation', 'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                                             'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus',
                                             'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
                                             'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'])

        try:
            # Make prediction using the loaded model
            prediction = model.predict(input_data)[0]
            # Get probability for class 1 (purchase)
            prediction_proba = model.predict_proba(input_data)[:, 1][0]

            st.subheader("Prediction Result:")
            if prediction == 1:
                st.success(f"The customer is likely to purchase the package! (Probability: {prediction_proba:.2f})")
            else:
                st.info(f"The customer is not likely to purchase the package. (Probability: {prediction_proba:.2f})")

            st.write("--- VAMSEE KRISHNA K ---")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

