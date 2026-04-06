import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Dataset Hub", page_icon="🚀", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
        text-align: center;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 20px;
        font-size: 3rem !important;
        font-weight: 800;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

if "access_token" not in st.session_state:
    st.session_state.access_token = None

def get_headers():
    return {"Authorization": f"Bearer {st.session_state.access_token}"}

# ----------------- SIDEBAR AUTH -----------------
with st.sidebar:
    st.title("🛡️ Authentication")
    
    if st.session_state.access_token:
        st.success("✅ Logged in")
        if st.button("Logout"):
            st.session_state.access_token = None
            st.rerun()
    else:
        auth_mode = st.radio("Choose Action", ["Login", "Register"], horizontal=True)
        
        if auth_mode == "Login":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                res = requests.post(
                    f"{API_URL}/auth/login",
                    data={"username": username, "password": password}
                )
                if res.status_code == 200:
                    st.session_state.access_token = res.json()["access_token"]
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        
        elif auth_mode == "Register":
            username = st.text_input("New Username")
            email = st.text_input("Email")
            password = st.text_input("New Password", type="password")
            if st.button("Register"):
                res = requests.post(
                    f"{API_URL}/auth/register",
                    json={"username": username, "email": email, "password": password}
                )
                if res.status_code == 201:
                    st.success("Registered successfully! Please login.")
                else:
                    st.error(res.json().get("detail", "Registration failed."))

# ----------------- MAIN APP -----------------
st.markdown("<h1 class='main-header'>Neural Network Project Request Hub</h1>", unsafe_allow_html=True)

if not st.session_state.access_token:
    st.info("### 👋 Welcome! \nPlease log in using the sidebar on the left to submit your dataset and requirements.")
else:
    st.markdown("### 📤 Upload Your Dataset & Requirements")
    st.write("Fill out the details below to define the architecture and expectations you have from the model.")
    
    with st.container():
        with st.form("submission_form", border=True):
            dataset_file = st.file_uploader("Upload Dataset (CSV limits apply)", type=["csv", "txt"])
            target_column = st.text_input("🎯 Target Column Name", placeholder="e.g. Sales, Price, Quality")
            use_case = st.text_area("💼 Use Case Description", placeholder="Describe the business scenario where this model will be used.")
            requirement = st.text_area("📋 Specific Requirements", placeholder="List any particular nuances needed in the model output or architecture.")
            
            submitted = st.form_submit_button("Submit Request 🚀")
            
            if submitted:
                if not dataset_file or not target_column or not use_case or not requirement:
                    st.error("⚠️ Please fill out all fields and upload a dataset.")
                else:
                    with st.spinner("⏳ Uploading directly to Cloudinary and securing metadata... (this may take a few moments)"):
                        # Prepare the multipart payload
                        files = {"dataset": (dataset_file.name, dataset_file, dataset_file.type)}
                        data = {
                            "target_column": target_column,
                            "use_case": use_case,
                            "requirement": requirement
                        }
                        
                        try:
                            # Send to FastAPI
                            res = requests.post(f"{API_URL}/submit/", headers=get_headers(), files=files, data=data)
                            if res.status_code == 200:
                                st.success("🎉 Successfully submitted request! Your dataset is backed up securely to Cloudinary.")
                                st.json(res.json())
                                st.balloons()
                            else:
                                st.error(f"❌ Error submitting request: {res.text}")
                        except requests.exceptions.ConnectionError:
                            st.error("🔌 Could not connect to the Backend API. Ensure it is running on port 8000.")
