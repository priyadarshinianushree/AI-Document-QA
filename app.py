from pypdf import PdfReader

pdf_path = "college_rules.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

# Split text into chunks
chunk_size = 200
chunks = []

for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i + chunk_size])

# Ask the user a question
question = input("\nAsk a question about the PDF: ")

question_words = question.lower().split()

# Calculate a simple relevance score
results = []

for chunk in chunks:
    chunk_lower = chunk.lower()

    score = 0

    for word in question_words:
        if word in chunk_lower:
            score += 1

    results.append((score, chunk))

# Sort by relevance
results.sort(reverse=True)

print("\n----- Most Relevant Information -----")

if results[0][0] > 0:
    print(results[0][1])
else:
    print("Sorry, I couldn't find relevant information.")