from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import SeleniumURLLoader
from pydantic import Field, BaseModel


def read_url(url: str) -> List[Document]:
    loader = SeleniumURLLoader(urls=[url])
    docs = loader.load()
    return docs

def read_webpage(
        url: str,
        query: str,
) -> str:
    """用于从一个网页中读取文本内容"""

    raw_docs = read_url(url)
    if len(raw_docs) == 0:
        return "Sorry, I can't read the webpage."
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_docs)
    if documents is None or len(documents) == 0:
        return "Sorry, I can't read the webpage."
    db = Chroma.from_documents(documents, OpenAIEmbeddings(model="text-embedding-ada-002"))
    docs = db.max_marginal_relevance_search(query, k=1)
    if len(docs) == 0:
        return "Sorry, I can't read the webpage."
    return docs[0].page_content
