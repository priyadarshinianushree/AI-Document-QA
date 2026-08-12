from pypdf import PdfReader
from google import genai
import os

# -----------------------------
# 1. Connect to Gemini
# -----------------------------

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# -----------------------------
# 2. Read the PDF
# -----------------------------

pdf_path = "college_rules.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

# -----------------------------
# 3. Create small chunks
# -----------------------------

lines = [
    line.strip()
    for line in text.split("\n")
    if line.strip()
]

chunks = []

for i in range(0, len(lines), 2):
    chunk = " ".join(lines[i:i + 2])
    chunks.append(chunk)

print("PDF loaded successfully!")
print("Number of chunks:", len(chunks))

# -----------------------------
# 4. Ask the user
# -----------------------------

question = input("\nAsk a question about the PDF: ")

# -----------------------------
# 5. Find relevant information
# -----------------------------

question_words = question.lower().split()

stop_words = {
    "what", "is", "the", "a", "an",
    "of", "how", "much", "are",
    "do", "does", "can", "i",
    "need", "to"
}

results = []

for index, chunk in enumerate(chunks):

    score = 0
    chunk_lower = chunk.lower()

    for word in question_words:

        if word in stop_words:
            continue

        if word in chunk_lower:
            score += 1

    results.append((score, index, chunk))

results.sort(
    key=lambda x: x[0],
    reverse=True
)

best_score, best_index, best_chunk = results[0]

# -----------------------------
# 6. Prepare context
# -----------------------------

if best_score > 0:

    context = best_chunk

    # Include next chunk for additional context
    if best_index + 1 < len(chunks):
        context += " " + chunks[best_index + 1]

else:

    context = "No relevant information was found."

# -----------------------------
# 7. Send context to Gemini
# -----------------------------

prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context below.

If the answer is not present in the context,
say that the information is not available in the document.

Do not invent information.

Document context:
{context}

User question:
{question}

Give a short and clear answer.
"""

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)

# -----------------------------
# 8. Display AI answer
# -----------------------------

print("\n----- AI Answer -----")

print(response.text)