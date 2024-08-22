import boto3
import json
import streamlit as st
from botocore.exceptions import ClientError
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Bedrock Runtime client
bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")

# Model ID for Llama
model_id = "meta.llama3-1-8b-instruct-v1:0"

# PDF and FAISS configuration
PDF_FOLDER = "pdfs"
FAISS_INDEX_PATH = "faiss_index"

# Function to extract text from PDF files
def get_pdf_text():
    text = ""
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    total_files = len(pdf_files)
    
    # Initialize the progress bar and placeholder for percentage text
    progress_bar = st.progress(0)
    percentage_text = st.empty()

    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Calculate and display percentage completed
        progress_percentage = (i + 1) / total_files
        progress_bar.progress(progress_percentage)
        percentage_text.text(f"Loading PDFs... {int(progress_percentage * 100)}% completed")
    
    # Clear the progress bar and percentage text after completion
    progress_bar.empty()
    percentage_text.empty()
    
    return text


# Function to split the PDF text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    
    # Split the text and keep track of progress
    total_length = len(text)
    text_chunks = []
    
    # Initialize the progress bar and placeholder for percentage text
    progress_bar = st.progress(0)
    percentage_text = st.empty()
    
    # Simulate chunk splitting with progress tracking
    current_position = 0
    while current_position < total_length:
        end_position = min(current_position + 10000, total_length)
        text_chunk = text[current_position:end_position]
        text_chunks.append(text_chunk)
        
        # Calculate and display percentage completed
        progress_percentage = (end_position / total_length)
        progress_bar.progress(progress_percentage)
        percentage_text.text(f"Splitting text... {int(progress_percentage * 100)}% completed")
        
        current_position += 9000  # Move by chunk size minus overlap
    
    # Clear the progress bar and percentage text after completion
    progress_bar.empty()
    percentage_text.empty()

    return text_chunks


# Function to generate a FAISS vector store from the text chunks and display percentage completed
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = None
    
    # Total number of chunks
    total_chunks = len(text_chunks)
    
    # Initialize the progress bar and placeholder for percentage text
    progress_bar = st.progress(0)
    percentage_text = st.empty()

    # Embed each chunk and update the percentage
    for i, chunk in enumerate(text_chunks):
        if vector_store is None:
            vector_store = FAISS.from_texts([chunk], embedding=embeddings)
        else:
            vector_store.add_texts([chunk])

        # Calculate and display percentage completed
        progress_percentage = (i + 1) / total_chunks
        progress_bar.progress(progress_percentage)
        percentage_text.text(f"Processing... {int(progress_percentage * 100)}% completed")

    # Save the vector store after processing all chunks
    vector_store.save_local(FAISS_INDEX_PATH)

    # Display completion message
    st.success("Embedding completed!")
    percentage_text.empty()  # Remove the percentage text
    progress_bar.empty()  # Clear the progress bar


# Function to call the Llama model on Bedrock
def call_llama_bedrock(prompt):
    # Format the prompt for Llama
    formatted_prompt = f"""
    You are an AI assistant that answers questions based on provided document context. 
    If the information is not in the context, reply with "answer is not available in the context."
    Give response in markdown format. limit output to 50 lines, but complete the last line.
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
    

def user_input(user_question):
    # Load the FAISS index from the storage
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Search the vector store for similar documents
    docs = new_db.similarity_search(user_question)
    
    # Combine the context from the documents to form the prompt for Llama
    context = " ".join([doc.page_content for doc in docs])
    
    # Create a dictionary for the prompt
    prompt = {
        "context": context,
        "question": user_question
    }

    # Call the Llama Bedrock model
    response = call_llama_bedrock(prompt)
    
    if response:
        st.write("Reply: ", response)
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
        
        # Check if the FAISS index already exists
        if st.button("Process PDFs"):
            if not os.path.exists(FAISS_INDEX_PATH):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text()
                    text_chunks = get_text_chunks(raw_text)
                    
                # Place the percentage display inside the spinner
                get_vector_store(text_chunks)
            else:
                st.success("FAISS index already exists, skipping embedding process.")

if __name__ == "__main__":
    main()
