from typing import Any

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator, evaluate
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


def prepare_data(run: Run, example: Example) -> dict[str, str]:
    return {
        "prediction": run.outputs["output"],
        "reference": example.outputs["output"],
    }


st.title("Evaluation")

submit = st.button("Run evaluation")

if submit:
    with st.spinner("Running evaluation..."):
        embedding_distance_evaluator = LangChainStringEvaluator(
            evaluator="embedding_distance",
            prepare_data=prepare_data,
            config={
                "embeddings": OpenAIEmbeddings(model="text-embedding-3-small"),
                "distance_metric": "cosine",
            },
        )

        client = Client()
        evaluate(
            predict,
            data="langsmith-demo",
            evaluators=[
                embedding_distance_evaluator,
                keyword_match,
            ],
        )
    st.success("Evaluation complete!")
