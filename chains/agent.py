from typing import Any

import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI


def create_agent_chain(
    history: BaseChatMessageHistory,
) -> Runnable[dict[str, Any], dict[str, Any]]:
    model = ChatOpenAI(name="gpt-3.5-turbo-0125", temperature=0)
    tools = [TavilySearchResults()]
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_tool_calling_agent(model, tools, prompt)

    memory = ConversationBufferMemory(
        chat_memory=history,
        memory_key="chat_history",
        return_messages=True,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)  # type: ignore[arg-type]

    return agent_executor.with_config(
        {
            "run_name": "agent",
            "metadata": {"conversation_id": st.session_state.conversation_id},
        },
    )
