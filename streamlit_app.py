


import os
import subprocess

try:
    import openpyxl
except ImportError:
    subprocess.check_call(["pip", "install", "openpyxl"])
    import openpyxl    

import streamlit as st
import pandas as pd
import openai
from fuzzywuzzy import fuzz  # Fallback for local similarity scoring


# OpenAI API Key
openai.api_key = "YOUR_OPENAI_API_KEY"

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
            # Load files
            input_df = pd.read_excel(uploaded_input_file, dtype=str)
            ref_df = pd.read_excel(uploaded_ref_file, dtype=str)

            results = []

            for _, row in ref_df.iterrows():
                row_score = 0
                mandatory_score = 0
                good_to_have_score = 0

                fully_matched_columns = []
                fully_matched_values = []
                partially_matched_columns = []
                partially_matched_values = []

                mandatory_values = []
                good_to_have_values = []

                # Process mandatory and good-to-have columns
                for col, weight in [(mandatory_columns, mandatory_weight), (good_to_have_columns, good_to_have_weight)]:
                    for input_col in col:
                        input_value = str(row.get(input_col, "")).strip() if pd.notna(row.get(input_col)) else ""

                        if input_col in mandatory_columns:
                            mandatory_values.append(input_value)
                        if input_col in good_to_have_columns:
                            good_to_have_values.append(input_value)

                        if not input_value or input_value == "No Data":
                            continue

                        full_match_found = False
                        max_similarity = 0
                        best_ref_value = None
                        best_ref_col = None

                        # Compare input value against all reference columns
                        for ref_col in input_df.columns:
                            ref_values = input_df[ref_col].dropna().astype(str).tolist()

                            # Check for full match
                            if input_value in ref_values:
                                row_score += weight
                                if input_col in mandatory_columns:
                                    mandatory_score += weight
                                else:
                                    good_to_have_score += weight

                                fully_matched_columns.append(f"{input_col} -> {ref_col}")
                                fully_matched_values.append(input_value)
                                full_match_found = True
                                break

                            # Skip partial matching if full match found
                            if full_match_found:
                                continue

                            # Use LLM for partial matching
                            for ref_value in ref_values:
                                if ref_value in fully_matched_values:
                                    continue

                                processed_input = preprocess_string(input_value)
                                processed_ref = preprocess_string(ref_value)

                                llm_similarity = compute_llm_similarity(processed_input, processed_ref)
                                if llm_similarity > max_similarity:
                                    max_similarity = llm_similarity
                                    best_ref_value = ref_value
                                    best_ref_col = ref_col

                        # Handle partial matches
                        if max_similarity > 0.3:
                            row_score += max_similarity * weight
                            if input_col in mandatory_columns:
                                mandatory_score += max_similarity * weight
                            else:
                                good_to_have_score += max_similarity * weight

                            partially_matched_columns.append(f"{input_col} -> {best_ref_col}")
                            partially_matched_values.append(best_ref_value)

                # Normalize scores
                max_possible_score = (len(mandatory_columns) * mandatory_weight) + (len(good_to_have_columns) * good_to_have_weight)
                normalized_score = (row_score / max_possible_score) * 100
                mandatory_score_normalized = (mandatory_score / (len(mandatory_columns) * mandatory_weight)) * 100
                good_to_have_score_normalized = (good_to_have_score / (len(good_to_have_columns) * good_to_have_weight)) * 100

                # Determine priority
                priority = 1 if normalized_score > 90 else 2 if 75 <= normalized_score <= 90 else 3 if 50 <= normalized_score < 75 else 4

                results.append({
                    "ID_S": row["ID_S"] if "ID_S" in row else "No Data",
                    "Mandatory_Columns": ", ".join(mandatory_columns),
                    "Good_to_Have_Columns": ", ".join(good_to_have_columns),
                    "Mandatory_Columns_Values": ", ".join(mandatory_values),
                    "Good_to_Have_Columns_Values": ", ".join(good_to_have_values),
                    "Fully_Matched_Columns": ", ".join(fully_matched_columns),
                    "Fully_Matched_Values": ", ".join(fully_matched_values),
                    "Partially_Matched_Columns": ", ".join(partially_matched_columns),
                    "Partially_Matched_Values": ", ".join(partially_matched_values),
                    "Normalized_Score": round(normalized_score, 2),
                    "Priority": priority,
                    "Mandatory_Score": round(mandatory_score_normalized, 2),
                    "Good_to_Have_Score": round(good_to_have_score_normalized, 2)
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
