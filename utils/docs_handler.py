import os
import streamlit as st

from utils import common

from langchain.storage import LocalFileStore
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    TextLoader,
)
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS


class DocsHandler:
    def __init__(self):
        self._splitter = None

    @property
    def splitter(self):
        if self._splitter is None:
            raise ValueError("Splitter is not set.")
        return self._splitter

    @splitter.setter
    def splitter(self, value):
        # value 객체가 "split_documents"라는 메서드 또는 속성을 갖고 있는지 체크
        # splitter는 langchain에서 제공하는 RecursiveCharacterTextSplitter, CharacterTextSplitter 같은 객체를 받아야함
        # 이들은 반드시 split_documents()라는 메서드를 갖기 때문에, 아래 if문으로 적절한 splitter를 받았는지 체크 할 수 있다.
        if not hasattr(value, "split_documents"):
            raise ValueError("Invalid splitter")
        self._splitter = value

    # 업로드한 파일이 이미 존재하는 경우 해당 함수를 실행하지 않음
    @st.cache_resource(show_spinner="Saving file...")
    def save_text_file(self, file):
        # setting path
        file_folder = "./.cache/files"
        common.check_dir(file_folder)

        # save uploaded file
        file_content = file.read()
        with open(f"{file_folder}/{file.name}", "wb") as f:
            f.write(file_content)

        return f"{file_folder}/{file.name}"

    @st.cache_resource(show_spinner="Loading file...")
    def load_file(self, file_path: str):
        loader = UnstructuredFileLoader(file_path)

        return loader.load()

    def split_n_return_docs(self, file_path: str):
        loader = TextLoader(file_path)

        return loader.load_and_split(text_splitter=self._splitter)

    @st.cache_resource(show_spinner="Embedding file...")
    def embedding_n_return_retriever(self, file_path: str):
        cache_dir = "./.cache/embeddings"
        common.check_dir(cache_dir)

        cache_path = LocalFileStore(
            f"{cache_dir}/{os.path.basename(file_path)}"
        )

        embeddings = OpenAIEmbeddings()

        docs = self.split_n_return_docs(file_path)
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_path
        )
        vector_store = FAISS.from_documents(docs, cached_embeddings)

        return vector_store.as_retriever()

    # document list -> string
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
