import streamlit as st
from tempfile import NamedTemporaryFile

from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptionTool, ObjectDetectionTool

# initialize agent
tools = [ImageCaptionTool(), ObjectDetectionTool()]

# k = 5 ==> the number of previous messages to remember
conversational_memory = ConversationBufferWindowMemory(memory_key = "chat_history",
                                                       k = 5, return_messages = True)

# temperature = 0 ==> deterministic/less creative
llm = ChatOpenAI(openai_api_key = "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",
                 temperature = 0, model_name = "gpt-3.5-turbo")

agent = initialize_agent(agent = "chat-conversational-react-description", tools = tools, llm = llm, max_iterations = 5,
                         verbose = True, memory = conversational_memory, early_stopping_method = "generate")

# set title
st.title("Snap Quest")

# set header
st.header("Please Upload an image(jpg, jpeg, png)")

# upload file
file = st.file_uploader("", type = ["jpg", "jpeg","png"])

if file:
    # display image
    st.image(file, use_column_width = True)
    
    # text input
    user_question = st.text_input("Ask a question about the image")
    
    # compute agent response
    with NamedTemporaryFile(dir = '.') as f:
        # save image to temporary file
        f.write(file.getbuffer())
        image_path = f.name
        
        
        # write agennt response
        if user_question and user_question != "":
            with st.spinner(text = "Processing..."):
                response = agent.run("{}, this is the image path: {}".format(user_question, image_path))
                st.write(response)