import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import collect_runs
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langsmith import Client
from streamlit_feedback import streamlit_feedback  # type: ignore[import-untyped]

from chains.retrieval_qa import create_retrieval_qa_chain

APP_NAME = "langsmith-demo"

load_dotenv()


st.title("Retrieval QA")

history = StreamlitChatMessageHistory()

for message in history.messages:
    st.chat_message(message.type).write(message.content)

question = st.chat_input("Ask me a question!")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        chain = create_retrieval_qa_chain()

        with collect_runs() as cb:
            stream = chain.stream(question)
            response = st.write_stream(stream)

            run_id = cb.traced_runs[0].id
            st.session_state.latest_run_id = run_id

        history.add_user_message(question)
        history.add_ai_message(response)  # type: ignore[arg-type]

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
