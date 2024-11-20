import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Set the page title
st.title("AI-Powered Redirect Mapping Tool")

st.markdown("""
Script made by Daniel Emery, Head of Search at Bang Digital.
Script adapted for Streamlit by Nathaniel Alzaga

👉🏼 **What It Is**  
This tool automates redirect mappings during site migrations by matching URLs from an old site to a new site based on content similarity.

👉🏼 **How to Use It:**  
1. Upload `origin.csv` and `destination.csv` files. Ensure that your file contains the URL, Title, Meta Description, and H1
2. Click **"Let's Go!"** to initiate the matching process.
3. Download the resulting `output.csv` file containing matched URLs with similarity scores.

👉🏼 **Note:**  
- Ensure your files are in `.csv` format with relevant columns for URL matching.
- Files should not exceed 2MB each.
""")

# Step 1: Upload Files
st.header("Upload Your Files")
uploaded_origin = st.file_uploader("Upload origin.csv", type="csv")
uploaded_destination = st.file_uploader("Upload destination.csv", type="csv")

if uploaded_origin and uploaded_destination:
    st.success("Files uploaded successfully!")
    
    # Step 2: Load Data
    origin_df = pd.read_csv(uploaded_origin)
    destination_df = pd.read_csv(uploaded_destination)

    # Combine all columns for similarity matching
    origin_df['combined_text'] = origin_df.fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
    destination_df['combined_text'] = destination_df.fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)

    # Step 3: Button to Process Matching
    if st.button("Let's Go!"):
        st.info("Processing data... This may take a while.")

        # Use a pre-trained model for embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Vectorize the combined text
        origin_embeddings = model.encode(origin_df['combined_text'].tolist(), show_progress_bar=True)
        destination_embeddings = model.encode(destination_df['combined_text'].tolist(), show_progress_bar=True)

        # Create a FAISS index
        dimension = origin_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(destination_embeddings.astype('float32'))

        # Perform the search for the nearest neighbors
        D, I = faiss_index.search(origin_embeddings.astype('float32'), k=1)

        # Calculate similarity scores
        similarity_scores = 1 - (D / np.max(D))

        # Create the output DataFrame with similarity scores
        matches_df = pd.DataFrame({
            'origin_url': origin_df['Address'],  # Replace 'Address' with the actual column name in your dataset
            'matched_url': destination_df['Address'].iloc[I.flatten()].values,
            'similarity_score': np.round(similarity_scores.flatten(), 4)
        })

        # Step 4: Display and Download Results
        st.success("Matching complete! Download your results below.")
        st.write(matches_df)

        st.download_button(
            label="Download Results as CSV",
            data=matches_df.to_csv(index=False),
            file_name="redirect_mapping_output.csv",
            mime="text/csv",
        )
