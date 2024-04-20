import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

load_dotenv()

APP_NAME = "langsmith-demo"


def create_agent_chain(history):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

    tools = [TavilySearchResults()]

    prompt = hub.pull("hwchase17/openai-tools-agent")

    memory = ConversationBufferMemory(
        chat_memory=history, memory_key="chat_history", return_messages=True
    )

    agent = create_tool_calling_agent(chat, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

    return agent_executor.with_config({"run_name": APP_NAME})


st.title(APP_NAME)

history = StreamlitChatMessageHistory()

for message in history.messages:
    st.chat_message(message.type).write(message.content)

prompt = st.chat_input("What's up?")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())

        agent_chain = create_agent_chain(history)
        response = agent_chain.invoke(
            {"input": prompt},
            {"callbacks": [callback]},
        )

        st.markdown(response["output"])
