import streamlit as st
from pypdf import PdfReader
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI Document Assistant")
st.write("Upload a PDF and ask questions about its contents.")

# --------------------------------
# Gemini setup
# --------------------------------

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found.")
    st.stop()

client = genai.Client(api_key=api_key)

# --------------------------------
# PDF upload
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:

    st.success("PDF uploaded successfully! ✅")

    # Read PDF
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    # --------------------------------
    # Split document into chunks
    # --------------------------------

    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    if not chunks:
        st.error("Could not extract text from this PDF.")
        st.stop()

    # --------------------------------
    # Question
    # --------------------------------

    question = st.text_input(
        "Ask a question about the PDF:"
    )

    if question:

        # --------------------------------
        # TF-IDF retrieval
        # --------------------------------

        documents = chunks + [question]

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(documents)

        question_vector = vectors[-1]

        chunk_vectors = vectors[:-1]

        similarities = cosine_similarity(
            question_vector,
            chunk_vectors
        )[0]

        # Get most relevant chunks
        top_indices = similarities.argsort()[-3:][::-1]

        relevant_chunks = [
            chunks[i]
            for i in top_indices
            if similarities[i] > 0
        ]

        if not relevant_chunks:

            st.warning(
                "I could not find relevant information in the document."
            )

            st.stop()

        # --------------------------------
        # Build context
        # --------------------------------

        context = "\n\n".join(relevant_chunks)

        # --------------------------------
        # Gemini prompt
        # --------------------------------

        prompt = f"""
You are an AI assistant for answering questions about documents.

Answer the user's question using ONLY the provided document context.

If the answer is not present in the context, say:

"Sorry, I could not find that information in the document."

Keep the answer short and clear.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}
"""

        # --------------------------------
        # Gemini response
        # --------------------------------

        with st.spinner("🤖 Finding the answer..."):

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

        # --------------------------------
        # Display answer
        # --------------------------------

        st.subheader("🤖 AI Answer")

        st.write(response.text)

        # --------------------------------
        # Show retrieved information
        # --------------------------------

        with st.expander("🔎 View Retrieved Information"):

            st.write(context)