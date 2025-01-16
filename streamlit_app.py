import streamlit as st
import pandas as pd
import openai
from fuzzywuzzy import fuzz
import openpyxl

# OpenAI API Key
openai.api_key = "YOUR_OPENAI_API_KEY"

st.image("https://strapi.tarento.com/uploads/Tarento_logo_749f934596.svg", width=200)

# Function to compute LLM-based semantic similarity
def compute_llm_similarity_bulk(input_values, reference_values):
    similarities = []
    for input_value in input_values:
        max_similarity = 0
        best_match = None
        for reference_value in reference_values:
            try:
                processed_input = preprocess_string(input_value)
                processed_ref = preprocess_string(reference_value)
                similarity = fuzz.ratio(processed_input, processed_ref) / 100  # Fallback similarity
                max_similarity = max(max_similarity, similarity)
                best_match = reference_value if similarity == max_similarity else best_match
            except:
                continue
        similarities.append((max_similarity, best_match))
    return similarities

# Function to preprocess strings
def preprocess_string(value):
    return value.strip().lower().replace(" ", "") if isinstance(value, str) else ""

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

# Add weightage selection in the sidebar
st.sidebar.subheader("Weightage Settings")
weight_option = st.sidebar.radio(
    "Select weightage between Mandatory and Good-to-Have fields:",
    options=["70:30", "60:40"],
    index=0
)

# Adjust weights based on user selection
mandatory_weight, good_to_have_weight = (70, 30) if weight_option == "70:30" else (60, 40)
st.sidebar.write(f"Selected Weightage: {mandatory_weight}% Mandatory, {good_to_have_weight}% Good-to-Have")
output_path = st.sidebar.text_input("Output File Path", "output.xlsx")

if st.sidebar.button("Run Scoring"):
    if uploaded_input_file and uploaded_ref_file:
        try:
            # Load files
            input_df = pd.read_excel(uploaded_input_file, dtype=str)
            ref_df = pd.read_excel(uploaded_ref_file, dtype=str)

            results = []

            for col_type, weight, columns in [
                ("Mandatory", mandatory_weight, mandatory_columns),
                ("Good-to-Have", good_to_have_weight, good_to_have_columns),
            ]:
                for ref_col in columns:
                    if ref_col not in ref_df.columns:
                        continue

                    input_values = input_df.stack().unique()
                    reference_values = ref_df[ref_col].dropna().unique()

                    # Bulk similarity computation
                    similarities = compute_llm_similarity_bulk(input_values, reference_values)

                    for max_similarity, best_match in similarities:
                        if max_similarity > 0.3:
                            results.append({
                                "Reference Column": ref_col,
                                "Matched Value": best_match,
                                "Similarity Score": max_similarity,
                                "Weight": weight,
                            })

            # Save and display results
            final_df = pd.DataFrame(results)
            final_df.to_excel(output_path, index=False)
            st.success(f"Scoring completed. Results saved to {output_path}")
            st.write("### Results:")
            st.dataframe(final_df)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please upload both input and reference files.")
