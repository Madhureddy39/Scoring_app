import streamlit as st
import pandas as pd
import openai
from fuzzywuzzy import fuzz  # Fallback for local similarity scoring


# OpenAI API Key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Tarento logo
st.set_page_config(page_title="Semantic Scoring Tool", layout="wide")
st.image("tarento_logo.png", width=200)  # Adjust the path and width as needed

# Function to compute LLM-based semantic similarity
def compute_llm_similarity(input_value, reference_value):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant evaluating semantic similarity between two texts."},
                {"role": "user",
                 "content": f"Rate the similarity between these two values on a scale of 0 to 1:\n\nValue 1: {input_value}\nValue 2: {reference_value}"}
            ],
        )
        score = float(response["choices"][0]["message"]["content"].strip())
        return score
    except Exception as e:
        print(f"Error during LLM similarity evaluation: {e}")
        return 0  # Default to 0 similarity in case of error

# Function to preprocess strings
def preprocess_string(value):
    return value.strip().lower().replace(" ", "")

# Streamlit UI
st.title("Semantic Similarity Scoring Tool")
st.sidebar.header("Settings")

uploaded_input_file = st.sidebar.file_uploader("Upload Input File", type=["xlsx"])
uploaded_ref_file = st.sidebar.file_uploader("Upload Reference File", type=["xlsx"])

if uploaded_ref_file:
    try:
        ref_df = pd.read_excel(uploaded_ref_file, dtype=str)
        st.session_state["ref_columns"] = ref_df.columns.tolist()
        st.sidebar.success("Reference file loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading reference file: {e}")
else:
    st.sidebar.warning("Please upload a reference file.")

if "ref_columns" not in st.session_state:
    st.session_state["ref_columns"] = []

mandatory_columns = st.sidebar.multiselect("Select Mandatory Columns", st.session_state["ref_columns"], key="mandatory")
good_to_have_columns = st.sidebar.multiselect("Select Good-to-Have Columns", st.session_state["ref_columns"], key="good_to_have")
output_path = st.sidebar.text_input("Output File Path", "output.xlsx")

mandatory_weight = 70
good_to_have_weight = 30

if st.sidebar.button("Run Scoring"):
    if uploaded_input_file and uploaded_ref_file:
        try:
            # (The remaining code for processing and scoring goes here...)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please upload both input and reference files.")
