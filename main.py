import streamlit as st
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
import os

from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptionTool, ObjectDetectionTool
from dotenv import load_dotenv


# load key-value pairs from the .env file located in the parent directory
load_dotenv(".env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
#print(f"OPENAI_API_KEY = {OPENAI_API_KEY}")

# initialize agent
tools = [ImageCaptionTool(), ObjectDetectionTool()]

# k = 5 ==> the number of previous messages to remember
conversational_memory = ConversationBufferWindowMemory(
    memory_key = "chat_history", 
    k = 5, 
    return_messages = True
)

# temperature = 0 ==> deterministic/less creative
llm = ChatOpenAI(
    openai_api_key = str(OPENAI_API_KEY),
    temperature = 0, 
    model_name = "gpt-3.5-turbo"
)

agent = initialize_agent(
    agent = "chat-conversational-react-description", 
    tools = tools, 
    llm = llm, 
    max_iterations = 5, 
    verbose = True, 
    memory = conversational_memory, 
    early_stopping_method = "generate"
)

# set title
st.title("Snap Quest")

# set header
st.header("Please Upload an image(jpg, jpeg, png)")

# upload file
file = st.file_uploader("", type = ["jpg", "jpeg","png"])

if file:
    print(f"file = {file}")
    # display image
    st.image(file, use_column_width = True)
    
    # text input
    user_question = st.text_input("Ask a question about the image")
    
    # compute agent response
    with NamedTemporaryFile(dir = '.', mode ="w+b") as f:
        print(f"inside temporary file")
        # save image to temporary file
        f.write(file.getbuffer())
        image_path = f.name
        
        
        # write agennt response
        if user_question and user_question != "":
            print(f"User question: {user_question}")
            print(f"Image path: {image_path}")
            with st.spinner(text = "Processing..."):
                response = agent.run("{}, this is the image path: {}".format(user_question, image_path))
                st.write(response)