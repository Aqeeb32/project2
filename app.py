import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="NLP Text Similarity App", layout="wide")

st.title("Text Similarity Analysis with Pretrained NLP Model")
st.markdown("This app uses a free pretrained model to analyze text similarity without any preprocessing.")

# --- Requirement 2: Load one free pretrained model directly ---
@st.cache_resource
def load_model():
    # Using a free, lightweight pretrained model from HuggingFace
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- Requirement 1: Text input box ---
st.subheader("1. Enter Text")
default_text = "I love programming.\nCoding is my passion.\nI enjoy walking in the park.\nDogs are very loyal pets."
user_input = st.text_area("Enter sentences or words (one per line):", value=default_text, height=150)

# Process input
texts = [t.strip() for t in user_input.split('\n') if t.strip()]

if len(texts) < 3:
    st.warning("Please enter at least 3 lines of text to generate meaningful graphs.")
else:
    # --- Strict Rule: No Preprocessing, directly pass to model ---
    embeddings = model.encode(texts)
    
    # Calculate similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # Extract pairwise scores for the bar chart (excluding self-similarity)
    pairs = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            pairs.append({"Pair": f'"{texts[i]}" & "{texts[j]}"', "Score": sim_matrix[i][j]})
    
    df_pairs = pd.DataFrame(pairs).sort_values(by="Score", ascending=False).head(5) # Top 5

    # --- Requirement 3: Show similarity scores ---
    st.subheader("2. Top Similarity Results")
    st.dataframe(df_pairs.style.format({"Score": "{:.4f}"}))

    # --- Requirement 4: Show at least three clear graphs ---
    st.subheader("3. Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graph 1: Bar Chart
        st.markdown("**Top Similar Word/Sentence Pairs**")
        fig_bar = px.bar(df_pairs, x="Score", y="Pair", orientation='h', text_auto='.3f',
                         title="Top Pairwise Similarity Scores")
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Graph 2: Heatmap
        st.markdown("**Pairwise Similarity Heatmap**")
        fig_heat = px.imshow(sim_matrix, x=texts, y=texts, text_auto=".2f", aspect="auto",
                             color_continuous_scale="Viridis", title="Similarity Heatmap")
        st.plotly_chart(fig_heat, use_container_width=True)

    # Graph 3: 2D Embedding Plot (PCA)
    st.markdown("**2D Embedding Plot (PCA)**")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    df_pca = pd.DataFrame({"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "Text": texts})
    
    fig_pca = px.scatter(df_pca, x="x", y="y", text="Text", size_max=60,
                         title="2D Plot of Text Embeddings (Related terms group together)")
    fig_pca.update_traces(textposition='top center')
    st.plotly_chart(fig_pca, use_container_width=True)

    # --- Requirement 5: Paul's Critical Thinking Standards ---
    st.subheader("4. Critical Thinking Notes (Paul's Standards)")
    st.markdown("""
    * **Clarity:** The input is a list of raw text strings. The output shows how mathematically similar these texts are in a dense vector space.
    * **Accuracy:** The results are derived directly from the free pretrained `all-MiniLM-L6-v2` model from HuggingFace, ensuring valid semantic representations without unsupported modifications.
    * **Precision:** Exact cosine similarity scores (ranging from -1.0 to 1.0) are calculated and displayed up to 4 decimal places, avoiding vague terms like 'high' or 'low similarity'.
    * **Relevance:** All three graphs (Bar Chart, Heatmap, and PCA plot) directly visualize the cosine similarity matrix and embeddings, strongly supporting the raw numerical data.
    * **Logic:** The highest scored pair makes logical sense because the model maps semantically similar contexts (e.g., programming and coding) closer together in the vector space compared to unrelated topics.
    * **Significance:** The most significant insight from the Bar Chart is identifying the most contextually overlapping text pair, filtering out noise.
    * **Fairness:** A limitation of this approach is that the pretrained model might not understand domain-specific jargon or nuances since no fine-tuning or custom dataset cleaning was permitted.
    """)
