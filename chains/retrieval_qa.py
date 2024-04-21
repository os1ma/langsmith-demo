from langchain import hub
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def create_retrieval_qa_chain() -> Runnable[str, str]:
    retriever = TavilySearchAPIRetriever(k=5)
    model = ChatOpenAI(name="gpt-3.5-turbo-0125", temperature=0)
    prompt = hub.pull("rlm/rag-prompt")

    chain: Runnable[str, str] = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.with_config({"run_name": "retrieval_qa"})
