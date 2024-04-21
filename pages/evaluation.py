from typing import Any

import streamlit as st
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run

from chains.retrieval_qa import create_retrieval_qa_chain


def predict(inputs: dict[str, str]) -> dict[str, str]:
    chain = create_retrieval_qa_chain()
    response = chain.invoke(inputs["input"])
    return {"output": response}


def keyword_match(run: Run, example: Example) -> dict[str, Any]:
    run_output = run.outputs["output"]
    expected_keyword = example.outputs["keyword"]

    if expected_keyword in run_output:
        score = 1
    else:
        score = 0

    return {"key": "keyword_match", "score": score}


st.title("Evaluation")

submit = st.button("Run evaluation")

if submit:
    with st.spinner("Running evaluation..."):
        client = Client()
        evaluate(
            predict,
            data="langsmith-demo",
            evaluators=[keyword_match],
        )
    st.success("Evaluation complete!")
