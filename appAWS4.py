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
    text = ""
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    total_files = len(pdf_files)
    
    progress_bar = st.progress(0)
    percentage_text = st.empty()

    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                for element in page.extract_words():
                    fontname = element.get('fontname', '').lower()
                    
                    if fontname.startswith('bold'):
                        text += f"**{element['text']}** "
                    elif fontname.startswith('italic'):
                        text += f"*{element['text']}* "
                    else:
                        text += element['text'] + " "
                text += "\n\n"

        progress_percentage = (i + 1) / total_files
        progress_bar.progress(progress_percentage)
        percentage_text.text(f"Loading PDFs... {int(progress_percentage * 100)}% completed")
    
    progress_bar.empty()
    percentage_text.empty()
    
    return text

# Function to call the Llama model on Bedrock
def call_llama_bedrock(prompt):
    formatted_prompt = f"""
    You are an AI assistant that answers questions based on provided document context. 
    If the information is not in the context, reply with "answer is not available in the context."
    Ensure to keep the formatting intact as provided in the document. limit output to 50 lines, but complete the last line.
    
    Context: {prompt['context']}
    
    Question: {prompt['question']}
    
    assistant:
    """
    
    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
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
    st.set_page_config("Chat PDF")
    st.header("AWS | DRUPAL | GENERATIVE AI TECH ASSIST💁")

    # Inject custom CSS for scrollable chat
    st.markdown(
        """
        <style>
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create a single column layout
    with st.container():
        # Display chat history in a scrollable container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["text"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Input field for user's message
        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "text": user_question})
            # Rerun the app to show the user's message immediately
            st.experimental_rerun()

    # Process the user's question and get the response
    if "user_question" in st.session_state:
        # Call the user_input function to get the response
        response_text = user_input(st.session_state.user_question)
        # Add the response to the chat history
        st.session_state.chat_history.append({"role": "assistant", "text": response_text})

        # Clear the stored user question after processing
        del st.session_state.user_question
        # Rerun the app to show the response
        st.experimental_rerun()

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
