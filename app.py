import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Set the page title
st.title("AI-Powered Redirect Mapping Tool")

st.markdown("""
Script made by Daniel Emery, Head of Search at Bang Digital.

👉🏼 **What It Is**  
This is a Python tool for Google Colab that automates redirect mappings during site migrations by matching URLs from an old site to a new site based on content similarity. Users can interactively choose the relevant columns from their CSV data for URL matching.

👉🏼 **What It's Made With:**  
- **faiss-cpu**: A library for efficient similarity search and clustering of dense vectors.  
- **sentence-transformers**: A Python framework for state-of-the-art sentence, text, and image embeddings.  
- **pandas**: An open-source data manipulation and analysis library.  
- **ipywidgets**: An interactive widget library for Jupyter notebooks.  

👉🏼 **How to Use It:**  
1. Prepare `origin.csv` and `destination.csv` with the page URL in the first column, followed by titles, meta descriptions, and headings. Remove unwanted URLs and duplicates.  
2. Open the Redirect Matchmaker Script in Google Colab.  
3. Install libraries by running **Cell #1**.  
4. Upload the `origin.csv` and `destination.csv` files when prompted by **Cell #2**.  
5. Select columns for matching in **Cell #3** using the displayed widget.  
6. Click **"Let's Go!"** to initiate the matching process.  
7. **Cell #4** processes the data and creates `output.csv` with matched URLs and a similarity score.  
8. Review and manually correct any inaccuracies in `output.csv`.  
9. Download `output.csv` automatically from the browser's download directory after the script completion.

👉🏼 **Note:**  
- I recommend using relative URLs (do a find and replace in Google Sheets for the domain).  
- Be sure to use only URLs with 200 status code and no UTM parameters, etc.  
- This script can be a time saver but please check everything thoroughly, especially on client/production sites.
""")

# Step 1: Upload Files
st.header("Upload Files")
uploaded_origin = st.file_uploader("Upload origin.csv", type="csv")
uploaded_destination = st.file_uploader("Upload destination.csv", type="csv")

if uploaded_origin and uploaded_destination:
    origin_df = pd.read_csv(uploaded_origin)
    destination_df = pd.read_csv(uploaded_destination)

    st.success("Files uploaded successfully!")
    
    # Step 2: Column Selection
    st.header("Select Columns for Similarity Matching")
    common_columns = list(set(origin_df.columns) & set(destination_df.columns))
    selected_columns = st.multiselect("Choose columns for matching", common_columns)

    if selected_columns:
        # Step 3: Perform Matching
        st.header("Running Matching Process")
        origin_df['combined_text'] = origin_df[selected_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)
        destination_df['combined_text'] = destination_df[selected_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)

        # Load pre-trained model
        st.info("Loading pre-trained model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Vectorize text
        st.info("Generating embeddings...")
        origin_embeddings = model.encode(origin_df['combined_text'].tolist(), show_progress_bar=True)
        destination_embeddings = model.encode(destination_df['combined_text'].tolist(), show_progress_bar=True)

        # Create FAISS index
        dimension = origin_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(destination_embeddings.astype('float32'))

        # Perform search
        D, I = faiss_index.search(origin_embeddings.astype('float32'), k=1)
        similarity_scores = 1 - (D / np.max(D))

        # Prepare output
        matches_df = pd.DataFrame({
            'origin_url': origin_df['Address'],
            'matched_url': destination_df['Address'].iloc[I.flatten()].values,
            'similarity_score': np.round(similarity_scores.flatten(), 4)
        })

        # Display and download output
        st.header("Results")
        st.write(matches_df)
        st.download_button(
            label="Download Results as CSV",
            data=matches_df.to_csv(index=False),
            file_name="redirect_mapping_output.csv",
            mime="text/csv",
        )
