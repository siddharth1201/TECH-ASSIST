import boto3
import json
import streamlit as st
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import pdfplumber

# Load environment variables
load_dotenv()

# Initialize Bedrock Runtime client
bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")

# Model ID for Llama
model_id = "meta.llama3-1-8b-instruct-v1:0"

# PDF and FAISS configuration
PDF_FOLDER = "pdfs"
FAISS_INDEX_PATH = "faiss_index"

# Initialize chat session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Function to extract text and formatting from PDF files
def get_pdf_text_with_formatting():
    text = []
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    total_files = len(pdf_files)
    
    progress_bar = st.progress(0)
    percentage_text = st.empty()

    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = ""
                for element in page.extract_words():
                    fontname = element.get('fontname', '').lower()
                    
                    if fontname.startswith('bold'):
                        page_text += f"**{element['text']}** "
                    elif fontname.startswith('italic'):
                        page_text += f"*{element['text']}* "
                    else:
                        page_text += element['text'] + " "
                text.append(page_text.strip())
        
        progress_percentage = (i + 1) / total_files
        progress_bar.progress(progress_percentage)
        percentage_text.text(f"Loading PDFs... {int(progress_percentage * 100)}% completed")
    
    progress_bar.empty()
    percentage_text.empty()
    
    return "\n\n".join(text)


# Function to call the Llama model on Bedrock
def call_llama_bedrock(prompt):
    formatted_prompt = f"""
        If the information related to the user's question is not present in the documents, respond with: "question out of context " .
        If it is in the documents summarise the answer.
        Context: {prompt['context']}
        Question: {prompt['question']}

        Your response must be solely based on the documents. If the answer is not found in the documents, respond with "question out of context: no response. \n\n"
    """
    
    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": 500,
        "temperature": 0.1,
        "stop": ["no response"]
    }
    
    request_body = json.dumps(native_request)

    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=request_body,
            contentType="application/json",
        )
        
        model_response = json.loads(response["body"].read())
        response_text = model_response.get("generation", "No response generated")
        return response_text
    
    except (ClientError, Exception) as e:
        st.error(f"ERROR: Unable to invoke model. Reason: {e}")
        return None


# Function to split the PDF text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


# Function to generate a FAISS vector store from the text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = None
    
    total_chunks = len(text_chunks)
    progress_bar = st.progress(0)
    percentage_text = st.empty()

    for i, chunk in enumerate(text_chunks):
        if vector_store is None:
            vector_store = FAISS.from_texts([chunk], embedding=embeddings)
        else:
            vector_store.add_texts([chunk])

        progress_percentage = (i + 1) / total_chunks
        progress_bar.progress(progress_percentage)
        percentage_text.text(f"Processing... {int(progress_percentage * 100)}% completed")

    vector_store.save_local(FAISS_INDEX_PATH)
    st.success("Embedding completed!")
    progress_bar.empty()
    percentage_text.empty()


# Function to handle user input and respond
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    context = " ".join([doc.page_content for doc in docs])
    
    prompt = {
        "context": context,
        "question": user_question
    }

    response = call_llama_bedrock(prompt)
    
    if response:
        return response
    else:
        return "No valid response received from the model."


# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Tech Assist!",
        page_icon=":brain:",  # Favicon emoji
        layout="centered",  # Page layout option
    )
    st.header("AWS | DRUPAL | GENERATIVE AI | LMS Moodle | TECH ASSIST 🤖")

    
    # Input field for user's message
    user_question = st.chat_input("Ask a Question from the PDF Files")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "text": user_question})

        response_text = user_input(user_question)
        st.session_state.chat_history.append({"role": "assistant", "text": response_text})
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])


    with st.sidebar:
        st.title("Menu:")
        
        if st.button("Start embedding"):
            if not os.path.exists(FAISS_INDEX_PATH):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text_with_formatting()
                    text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
            else:
                st.success("FAISS index already exists, skipping embedding process.")

if __name__ == "__main__":
    main()