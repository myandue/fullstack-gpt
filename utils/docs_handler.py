import os

from utils import common

from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS


class DocsHandler:
    def __init__(self, splitter):
        # value 객체가 "split_documents"라는 메서드 또는 속성을 갖고 있는지 체크
        # splitter는 langchain에서 제공하는 RecursiveCharacterTextSplitter, CharacterTextSplitter 같은 객체를 받아야함
        # 이들은 반드시 split_documents()라는 메서드를 갖기 때문에, 아래 if문으로 적절한 splitter를 받았는지 체크 할 수 있다.
        if not hasattr(splitter, "split_documents"):
            raise ValueError("Invalid splitter")
        self.splitter = splitter

    def split_n_return_docs(self, file_path: str):
        loader = TextLoader(file_path)
        docs = loader.load_and_split(text_splitter=self.splitter)

        return docs

    def embedding_n_return_retriever(self, file_path: str):
        cache_dir = "./.cache/embeddings"
        common.check_dir(cache_dir)

        embeddings = OpenAIEmbeddings()

        cache_path = LocalFileStore(
            f"{cache_dir}/{os.path.basename(file_path)}"
        )
        loader = TextLoader(file_path)
        docs = loader.load_and_split(text_splitter=self.splitter)
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_path
        )
        vector_store = FAISS.from_documents(docs, cached_embeddings)
        retriever = vector_store.as_retriever()

        return retriever

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
