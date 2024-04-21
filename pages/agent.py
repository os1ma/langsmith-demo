from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import collect_runs
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langsmith import Client
from streamlit_feedback import streamlit_feedback  # type: ignore[import-untyped]

from chains.agent import create_agent_chain

load_dotenv()


st.title("Agent")

if not st.session_state.get("conversation_id"):
    st.session_state.conversation_id = uuid4()

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
