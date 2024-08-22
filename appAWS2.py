import boto3
import json
import streamlit as st
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import pdfplumber  # Use pdfplumber for extracting PDF with formatting

# Load environment variables
load_dotenv()

# Initialize Bedrock Runtime client
bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")

# Model ID for Llama
model_id = "meta.llama3-1-8b-instruct-v1:0"

# PDF and FAISS configuration
PDF_FOLDER = "pdfs"
FAISS_INDEX_PATH = "faiss_index"

# Function to extract text and formatting from PDF files
def get_pdf_text_with_formatting():
    text = ""
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    total_files = len(pdf_files)
    
    # Initialize the progress bar and placeholder for percentage text
    progress_bar = st.progress(0)
    percentage_text = st.empty()

    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text with markdown-like formatting
                for element in page.extract_words():
                    # Check if the 'fontname' attribute exists
                    fontname = element.get('fontname', '').lower()
                    
                    if fontname.startswith('bold'):
                        text += f"**{element['text']}** "
                    elif fontname.startswith('italic'):
                        text += f"*{element['text']}* "
                    else:
                        text += element['text'] + " "
                text += "\n\n"  # Add paragraph spacing between pages

        # Calculate and display percentage completed
        progress_percentage = (i + 1) / total_files
        progress_bar.progress(progress_percentage)
        percentage_text.text(f"Loading PDFs... {int(progress_percentage * 100)}% completed")
    
    print("loaded")
    # Clear the progress bar and percentage text after completion
    progress_bar.empty()
    percentage_text.empty()
    
    return text


# Function to call the Llama model on Bedrock
def call_llama_bedrock(prompt):
    # Format the prompt for Llama
    formatted_prompt = f"""
    You are an AI assistant that answers questions based on provided document context. 
    If the information is not in the context, reply with "answer is not available in the context."
    Ensure to keep the formatting intact as provided in the document. limit output to 50 lines, but complete the last line.
    
    Context: {prompt['context']}
    
    Question: {prompt['question']}
    
    assistant:
    """
    
    # Native request payload structure
    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
    }
    
    # Convert the request to JSON
    request_body = json.dumps(native_request)

    try:
        # Invoke the Llama model
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=request_body,
            contentType="application/json",  # Specify content type as JSON
        )
        
        # Parse the response from the Bedrock API
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
        st.markdown(response)
    else:
        st.write("No valid response received from the model.")


# Main Streamlit app
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Llama 3.1 8B 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        
        if st.button("Process PDFs"):
            if not os.path.exists(FAISS_INDEX_PATH):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text_with_formatting()
                    text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
            else:
                st.success("FAISS index already exists, skipping embedding process.")


if __name__ == "__main__":
    main()
