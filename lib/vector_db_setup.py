import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import DataFrameLoader, TextLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import string

import os
import datetime as dt
import logging
from dotenv import load_dotenv
load_dotenv()
PROJECT_DIRECTORY = os.getenv('PROJECT_DIRECTORY')
SOURCE_DOCUMENTS_FOLDER = os.getenv('SOURCE_DOCUMENTS_FOLDER')

logging.basicConfig(
    # filename='lll.log',
    # filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f"{PROJECT_DIRECTORY}/logs/{dt.datetime.now().strftime('%Y_%m_%d')}.log"),
        logging.StreamHandler()
    ],
    level=logging.INFO
)


def get_texts(
        file_name:str,
        collection_name:str,
        chunk_size=100,
        chunk_overlap=10
    ):

    """
    This function is used to prepare the data for the vector store.
    It is used to load the dataset and split the documents into smaller chunks.
    
    file_name: str - full file's name in $SOURCE_DOCUMENTS_FOLDER
    chunk_size: int, default=500 - the size of the chunks to split the documents into
    chunk_overlap: int, default=200 - the overlap between the chunks
    """
    dataset_path = SOURCE_DOCUMENTS_FOLDER + "/" + file_name
    logging.info(f'Выбраны данные из файла: {dataset_path}')

    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path, dtype='object')
        loader = DataFrameLoader(df, page_content_column="question")
    elif dataset_path.endswith('.txt'):
        loader = TextLoader(dataset_path)
    elif dataset_path.endswith('.pdf'):
        loader = PDFMinerLoader(dataset_path)
    
    documents = loader.load()  

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        length_function=len
    )

    # text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    #     AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca"),
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    #     separators=["\n\n", "\n", " ", ""]
    # )

    texts = text_splitter.split_documents(documents)


    def upload_to_vectorstore(texts, collection_name):

        """
        This function is used to upload the data to the vector store.
        It is used to upload the data to the vector store and create a collection.

        texts: list - the list of texts to upload to the vector store
        collection_name: str, default=None - the name of the collection to create in the vector store    
        """

        flag = True
        try:
            # embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            logging.info("Загрузка модели для эмбеддингов: SUCCESS")
        except Exception as e:
            logging.info("Ошибка при загрузке модели эмбеддингов:", e)
            flag = False
            
        if flag:
            try:
                chroma_client = chromadb.HttpClient(settings=Settings(
                    allow_reset=True,
                    chroma_api_impl='chromadb.api.fastapi.FastAPI',
                    chroma_server_host='localhost',
                    chroma_server_http_port='8000',
                    anonymized_telemetry=False)
                )
                logging.info("Подключение к CHROMADB: SUCCESS")
            except Exception as e:
                logging.info("Ошибка при подключении к CHROMADB:", e)
                flag = False
        
        if flag:
            for i in chroma_client.list_collections():
                if i.name == collection_name:
                    chroma_client.delete_collection(collection_name)
                    logging.info(f'Была обнаружена и удалена существующая коллекция с именем "{collection_name}"')
            try:
                collection = chroma_client.get_or_create_collection(name=collection_name,
                                                metadata={"hnsw:space": "cosine"},
                                                embedding_function=embedding_function
                                                )
                logging.info(f"Создание коллекции {collection_name}: SUCCESS")
            except Exception as e:
                logging.info("Ошибка при создании коллекции:", e)
                flag = False
        
        if flag:
            try:
                counter = 0
                for doc in texts:
                    counter += 1
                    # if counter % 100 == 0:
                    #     logging.info(f"uploaded {counter} of {len(texts)}")
                    collection.add(
                        documents=doc.page_content,
                        metadatas=doc.metadata,
                        ids=['id'+str(counter)]
                        # ids=doc.metadata['id']
                    )
                logging.info("Загрузка данных в CHROMADB: SUCCESS")
            except Exception as e:
                logging.info("Ошибка при загрузке данных в CHROMADB:", e)
    
    upload_to_vectorstore(
        texts=texts,
        collection_name=collection_name
    )
    return texts


def get_chroma_client():
    """
    This function is used to get the chroma client.
    It is used to get the chroma client to connect to the vector store.
    
    Returns: chroma_client - the chroma client to connect to the vector store"""
    try:
        chroma_client = chromadb.HttpClient(settings=Settings(
            allow_reset=True,
            chroma_api_impl='chromadb.api.fastapi.FastAPI',
            chroma_server_host='localhost',
            chroma_server_http_port='8000',
            anonymized_telemetry=False)
        )
        logging.info("Подключение к CHROMADB: SUCCESS")
        logging.info("Доступны следующие коллекции:")
        for i in chroma_client.list_collections():
            logging.info(f"- {i.name}")
        return chroma_client
    except Exception as e:
        logging.info("Ошибка при подключении к CHROMADB:", e)
        return None

def vectorstore_query(collection, source_file_type, question, n_results):
    """
    This function is used to query the vector store.
    It is used to query the vector store to get the response to a question.
    
    collection: collection - the collection to query in the vector store
    source_file_type: str - type of file which was used for collection creating
    question: str - the question to query in the vector store
    n_results: int - the number of results to return
    """

    response = collection.query(
            query_texts=question,
            n_results=n_results
        )
    
    if source_file_type.lower() == 'csv':
        # убираем одинаковые ответы
        response_list = []
        for repl in response['metadatas'][0]:
            response_list.append(repl['answer'])   
        response_list = list(set(response_list))

        # собираем итоговый ответ
        vector_db_response = " ".join(response_list)
        
    elif source_file_type.lower() in ['txt']:
        vector_db_response = " ".join(response["documents"][0])
    
    elif source_file_type.lower() in ['pdf']:
        vector_db_response = ''
        for doc in response['documents'][0]:
            for i in doc:
                if i in string.punctuation or i in "«»": 
                    doc = doc.replace(i, '').replace("\n","")
            vector_db_response += doc.capitalize() + ". "

    return vector_db_response
