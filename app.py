import streamlit as st
import os
from dotenv import load_dotenv
import time
from io import StringIO # To handle uploaded file in memory

# Langchain specific imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage

# --- Configuration and Initialization ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RESUME_FILENAME = "my_resume.pdf" # Default, can be overridden by upload

# --- Helper Functions ---
def load_resume_from_bytes(pdf_bytes, filename="uploaded_resume.pdf"):
    """Loads resume text content from bytes (uploaded file)."""
    print(f"Attempting to load resume from uploaded bytes: '{filename}'")
    # Save bytes to a temporary file to use PyPDFLoader
    temp_pdf_path = f"./temp_{filename}"
    try:
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_bytes)

        loader = PyPDFLoader(temp_pdf_path)
        pages = loader.load_and_split()
        if not pages:
            st.error(f"Error: No content loaded from PDF '{filename}'. It might be empty, password-protected, or unreadable.")
            return None
        print(f"Resume loaded successfully from '{filename}' ({len(pages)} pages).")
        full_resume_text = "\n".join([page.page_content for page in pages])
        return full_resume_text
    except FileNotFoundError:
         st.error(f"Error: Could not create temporary file for PDF '{filename}'.")
         return None
    except ImportError:
         st.error("Error: pypdf library not found. Please install it: pip install pypdf")
         return None
    except Exception as e:
        st.error(f"Error loading or reading PDF resume file '{filename}': {e}")
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            print(f"Removed temporary file: {temp_pdf_path}")


# --- LangChain Prompts --- (Can be customized further)
FIT_SCORE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant analyzing job fit based *only* on the provided resume and job description.
     Provide:
     1.  A 'Fit Score' (0-10, 10=perfect fit).
     2.  A brief justification (2-4 sentences).
     3.  Key matching skills/experience (bullet points).
     4.  Noticeable gaps (bullet points).
     Format your response clearly using Markdown."""),
    ("human", """Analyze the fit between this resume and job description:

--- RESUME START ---
{resume}
--- RESUME END ---

--- JOB DESCRIPTION START ---
{job_description}
--- JOB DESCRIPTION END ---""")
])

COVER_LETTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant writing a professional, concise cover letter based *strictly* on the provided Resume and Job Description.
    - Highlight skills/experiences from the resume that match job requirements.
    - Aim for three short paragraphs: Intro, Qualification Highlights (3-4 sentences), Closing.
    - Maintain a professional and enthusiastic tone.
    - Address it to 'Dear Hiring Manager,' and end formally with letter format.
    - Output *only* the cover letter text."""),
    MessagesPlaceholder(variable_name="history"), # Include chat history for context if needed, though maybe not for strict generation
    ("human", """Based on our conversation and the following documents, please draft the cover letter:

--- RESUME START ---
{resume}
--- RESUME END ---

--- JOB DESCRIPTION START ---
{job_description}
--- JOB DESCRIPTION END ---

{user_request}""") # Allow specific instructions from the user prompt
])

GENERAL_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are AICareerNavigator, a helpful assistant knowledgeable about the provided resume and job description. Answer questions concisely based on that context."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}"),
])

# --- Streamlit App UI ---
st.set_page_config(page_title="AI Career Navigator", layout="wide")
st.title("AI Career Navigator Chatbot")
st.caption("Chat about your resume and a job description and generate a cover letter.")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # Store chat history {role: "user/assistant", content: "..."}
if "chat_history_langchain" not in st.session_state:
     # Use ChatMessageHistory for LangChain Memory compatibility
    st.session_state.chat_history_langchain = ChatMessageHistory()
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        chat_memory=st.session_state.chat_history_langchain,
        return_messages=True 
    )
if "resume_content" not in st.session_state:
    st.session_state.resume_content = None
if "job_desc_content" not in st.session_state:
    st.session_state.job_desc_content = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "llm_initialized" not in st.session_state:
     st.session_state.llm_initialized = False


# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")

    # API Key Input (Consider using st.secrets for deployment)
    # api_key_input = st.text_input("Google API Key", type="password", value=GOOGLE_API_KEY or "")
    if not GOOGLE_API_KEY:
         st.warning("API Key not found in .env. Please provide it below.")
         api_key_input = st.text_input("Google API Key", type="password")
    else:
         api_key_input = GOOGLE_API_KEY # Use key from .env if found
         st.success("API Key loaded from .env")

    # Resume Upload
    uploaded_file = st.file_uploader("Upload Resume PDF", type="pdf")
    if uploaded_file is not None:
        if st.button("Load Uploaded Resume"):
            with st.spinner("Processing resume..."):
                pdf_bytes = uploaded_file.getvalue()
                st.session_state.resume_content = load_resume_from_bytes(pdf_bytes, uploaded_file.name)
                if st.session_state.resume_content:
                    st.success("Resume loaded successfully!")
                    st.session_state.messages.append({"role": "system", "content": "Resume loaded."}) # Add system message to chat
                    st.session_state.chat_history_langchain.add_message(HumanMessage(content="I have uploaded my resume."))
                    st.session_state.chat_history_langchain.add_message(AIMessage(content="Okay, I have loaded your resume content."))


    # Job Description Input
    st.session_state.job_desc_content = st.text_area("Paste Job Description Here", height=300, value=st.session_state.job_desc_content or "")
    if st.session_state.job_desc_content:
         st.success("Job description provided.")
         # Optionally add a system message to history when JD is pasted/changed

    # Initialize LLM Button (Only if not already initialized)
    if not st.session_state.llm_initialized and api_key_input:
        if st.button("Initialize LLM"):
            with st.spinner("Initializing LLM model..."):
                try:
                    st.session_state.llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash-preview-04-17", # choose a suitable model
                        google_api_key=api_key_input,
                        temperature=0.7 # Adjust as needed
                        #convert_system_message_to_human=True # do not need for current model/Langchain version
                    )
                    # Simple test call
                    st.session_state.llm.invoke("Hello!")
                    st.session_state.llm_initialized = True
                    st.success(f"LLM ({st.session_state.llm.model}) Initialized!")
                    st.session_state.messages.append({"role": "system", "content": "LLM Initialized."})
                    st.rerun() # Rerun to update UI state
                except Exception as e:
                    st.error(f"Error initializing LLM: {e}. Check API key and permissions.")
                    st.session_state.llm = None
                    st.session_state.llm_initialized = False
    elif st.session_state.llm_initialized:
         st.info(f"LLM ({st.session_state.llm.model}) is initialized.")

# --- Main Chat Interface ---

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if ready for chat commands
ready_for_commands = st.session_state.resume_content and st.session_state.job_desc_content and st.session_state.llm_initialized

if not ready_for_commands:
     st.warning("Please upload a resume, paste a job description, and initialize the LLM using the sidebar.")

# Chat input
if prompt := st.chat_input("Ask about the fit score, cover letter, or anything else..." if ready_for_commands else "Provide inputs in sidebar first...", disabled=not ready_for_commands):
    # Add user message to UI chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare context for LangChain runnables
    context = {
        "resume": st.session_state.resume_content or "Not Provided",
        "job_description": st.session_state.job_desc_content or "Not Provided",
        "user_input": prompt,
        "user_request": prompt # Specific for cover letter prompt if needed
    }

    # Determine which action to take based on input (simple keyword matching)
    output_parser = StrOutputParser()
    chosen_runnable = None
    prompt_lower = prompt.lower()

    with st.spinner("Thinking..."):
        try:
            if "fit score" in prompt_lower or "score fit" in prompt_lower or "analyze fit" in prompt_lower:
                st.info("Generating Fit Score...")
                fit_chain = (
                    RunnablePassthrough() # Pass context directly
                    | FIT_SCORE_PROMPT
                    | st.session_state.llm
                    | output_parser
                )
                response_content = fit_chain.invoke(context)
                chosen_runnable = "Fit Score"

            elif "cover letter" in prompt_lower or "write letter" in prompt_lower or "draft letter" in prompt_lower:
                st.info("Generating Cover Letter...")
                # We need to load memory variables for the cover letter chain
                cover_letter_chain = (
                    RunnablePassthrough.assign(
                         history=RunnableLambda(st.session_state.memory.load_memory_variables) | RunnableLambda(lambda mem: mem['history'])
                    )
                    | COVER_LETTER_PROMPT
                    | st.session_state.llm
                    | output_parser
                )

                response_content = cover_letter_chain.invoke(context)
                chosen_runnable = "Cover Letter"

            else:
                # General chat - use memory
                st.info("Handling general query...")
                 # Runnable that loads memory, passes inputs, runs chain, saves memory
                general_chain = (
                    RunnablePassthrough.assign(
                         history=RunnableLambda(st.session_state.memory.load_memory_variables) | RunnableLambda(lambda mem: mem['history'])
                     )
                    | GENERAL_CHAT_PROMPT
                    | st.session_state.llm
                    | output_parser
                )
                response_content = general_chain.invoke(context)
                chosen_runnable = "General Chat"

            # Add AI response to UI chat history
            st.session_state.messages.append({"role": "assistant", "content": response_content})

            # Update LangChain memory (important!)
            # For simplicity now, add all interactions. Can choose to add less

            st.session_state.memory.save_context({"user_input": prompt}, {"output": response_content})
            # Note: save_context expects dicts. Check LangChain docs for exact format if issues arise.
            # Alternative: Manually add messages if save_context is tricky
            # st.session_state.chat_history_langchain.add_user_message(prompt)
            # st.session_state.chat_history_langchain.add_ai_message(response_content)


            # Display AI response
            with st.chat_message("assistant"):
                st.markdown(response_content)

        except Exception as e:
             error_message = f"An error occurred: {e}"
             st.error(error_message)
             st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {e}"})
             with st.chat_message("assistant"):
                  st.markdown(f"Sorry, I encountered an error processing that request ({chosen_runnable or 'Unknown'}).")