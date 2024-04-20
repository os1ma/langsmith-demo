from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.callbacks import collect_runs
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langsmith import Client
from streamlit_feedback import streamlit_feedback  # type: ignore[import-untyped]

APP_NAME = "langsmith-demo"

load_dotenv()


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

    return agent_executor.with_config({"run_name": "agent"})


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

        with collect_runs() as cb:
            response = agent_chain.invoke(
                {"input": prompt},
                {"callbacks": [callback]},
            )

            run_id = cb.traced_runs[0].id
            st.session_state.latest_run_id = run_id

        st.markdown(response["output"])

if st.session_state.get("latest_run_id"):
    run_id = st.session_state.latest_run_id

    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional] Please provide an explanation",
    )

    if feedback:
        scores = {"👍": 1, "👎": 0}
        score_key = feedback["score"]
        score = scores[score_key]
        comment = feedback.get("text")

        client = Client()
        client.create_feedback(
            run_id=run_id,
            key="thumbs",
            score=score,
            comment=comment,
        )
