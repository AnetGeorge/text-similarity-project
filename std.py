import streamlit as st
import PyPDF2
import re
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

#Web Interface
# Set page layout to wide
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;  /* White background color */
    }
    h1 {
        color: #007bff;  /* Vibrant blue for the main heading */
        text-align: center;  /* Center align the heading */
        font-size: 48px;  /* Increase font size */
        font-weight: bold;  /* Make the font bold */
    }
    .subheader {
        text-align: center;  /* Center align the subheader */
        color: #ffffff;  /* White text for subheader */
        font-size: 24px;  /* Font size for subheader */
        margin-top: 20px;  /* Space above subheader */
        background-color: ;  /* Blue background for subheader */
        padding: 10px;  /* Padding around the text */
        border-radius: 5px;  /* Rounded corners */
    }
    .description {
        text-align: center;  /* Center align the description */
        color: #ffffff;  /* White text for description */
        font-size: 18px;  /* Font size for description */
        margin-bottom: 20px;  /* Space below description */
        background-color: ;  /* Blue background for description */
        padding: 10px;  /* Padding around the text */
        border-radius: 5px;  /* Rounded corners */
    }
    .score-label {
        text-align: center;  /* Center align score labels */
        font-size: 24px;  /* Font size for score labels */
        color: #CB6040;  /* Green for similarity score labels */
        margin: 20px 0;  /* Space between labels */
         background-color:#3C3D37;
    }
    .score{
       font-size: 24px;  /* Font size for score labels */
       color: #4CC9FE;  /* Green for similarity score labels */
    }
    .upload-label {
        text-align: center;
        color: #6c757d;  /* Grey for upload labels */
        font-size: 18px;  /* Slightly smaller than score labels */
        margin-bottom: 10px;  /* Space below upload labels */
    }
    .result-text {
        text-align: center; 
        font-size: 20px; 
        color: 7ED4AD;  /* Dark grey for result text */
        margin-top: 40px;  /* Space above result text */
    }
    .submit-button {
        display: flex;
        justify-content: center;  /* Center align the button */
        margin: 30px 0;  /* Space around the button */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main heading
st.markdown("<h1>PLAGIAGURAD</h1>", unsafe_allow_html=True)

# Centered subheader and description with white text on blue background
st.markdown("<div class='subheader'>Welcome to the Plagiarism Guard Application!</div>", unsafe_allow_html=True)
st.markdown("<div class='description'>Please upload your documents to check for plagiarism.</div>", unsafe_allow_html=True)

# Horizontal layout for PDF upload boxes
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='upload-label'>Upload Document 1:</div>", unsafe_allow_html=True)
    document1 = st.file_uploader("Document1:", type="pdf")

with col2:
    st.markdown("<div class='upload-label'>Upload Document 2:</div>", unsafe_allow_html=True)
    document2 = st.file_uploader("Document2:", type="pdf")
    st.write("\n")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    # Read the file content from the uploaded file object
    pdf_bytes = uploaded_file.read()

    # Use PyPDF2 to read the PDF from the in-memory bytes
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))  # In-memory byte stream
    text = ""
    
    for page in reader.pages:
        text += page.extract_text()  # Extract text from each page
    
    return text
    
# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Function to check similarity using TF-IDF
def check_similarity(doc1, doc2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

def load_model_from_drive(load_path='resources\\sentence_transformers'):
    model = SentenceTransformer(load_path)
    print(f"Model loaded from Google Drive at: {load_path}")
    return model

# Example usage
model = load_model_from_drive()
def semantic_similarity(doc1, doc2):
  embedding1 = model.encode(doc1, convert_to_tensor=True)
  embedding2 = model.encode(doc2, convert_to_tensor=True)
  similarity = util.pytorch_cos_sim(embedding1, embedding2)
  return similarity.item()

# Function to generate a plagiarism report
def generate_report(similarity_score, semantic_score, pdf1, pdf2):
    print("Plagiarism Report")
    print("=================")
    print(f"Similarity Score (TF-IDF): {similarity_score:.4f}")
    print(f"Semantic Similarity Score (BERT): {semantic_score:.4f}")
    print(f"Document 1: {pdf1}")
    print(f"Document 2: {pdf2}")
    tfidf_threshold = 0.7  # 70%
    semantic_threshold = 0.8  # 80%

    # Determine if the documents are plagiarized
    if similarity_score >= tfidf_threshold and semantic_score >= semantic_threshold:
        return "Result: These files are likely plagiarized."
    else:
        return "Result: These files are not plagiarized."

#Text Extraction
def submit_documents(doc1, doc2):
    # Your logic to process the documents goes here
    if doc1 and doc2:
        # Example processing logic
        st.success("Documents submitted successfully!")
        print(f"Docs uploaded")
    else:
        st.error("Please upload both documents.")

    extracted_text1 = extract_text_from_pdf(document1)
    extracred_text2 = extract_text_from_pdf(document2)
    cleaned_doc1 = preprocess_text(extracted_text1)
    cleaned_doc2 = preprocess_text(extracred_text2)

   # Calculate similarity scores
    similarity_score = check_similarity(cleaned_doc1, cleaned_doc2)
    semantic_score = semantic_similarity(cleaned_doc1, cleaned_doc2)
   # Generate the plagiarism report
    Result=generate_report(similarity_score, semantic_score, document1, document2)
    return semantic_score,similarity_score,Result
# Centered submit button
if st.button("Submit"):
    semantic_score,similarity_score,Result=submit_documents(document1, document2)  # Call the function on button click
    # Spacer
    st.write("\n")

    # Horizontal layout for similarity score labels
    st.markdown(f"<div class='score-label'>Similarity Score (TF-IDF):<div class='score'>{similarity_score}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='score-label'>Semantic Similarity Score (BERT):<div class='score'>{semantic_score}</div></div>", unsafe_allow_html=True)

    # Placeholder for similarity scores
    st.markdown(f"<h2 class='result-text'>{Result}</h2>", unsafe_allow_html=True)




































































