import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 





def get_texts(chunk_size=500, chunk_overlap=500):
    """
    This function is used to prepare the data for the vector store.
    It is used to load the dataset and split the documents into smaller chunks.
    
    chunk_size: int, default=500 - the size of the chunks to split the documents into
    chunk_overlap: int, default=500 - the overlap between the chunks
    """

    dataset_path = input("Введите относительный путь до файла в формате CSV: ")
    if len(dataset_path) > 0:
        try:
            df = pd.read_csv(dataset_path, dtype='object')
            loader = DataFrameLoader(df, page_content_column='question')
            documents = loader.load()

            rec_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap,
                                                    length_function=len
                                                    )
            texts = rec_text_splitter.split_documents(documents)
        except:
            print("Некорректный путь к датасету или произошла ошибка при обработке")
        else:
            print('Датасет успешно загружен')
    else:
        print('Применяю датасет по умолчаюнию: ./SOURCE_DOCUMENTS/questions_stolot.csv')
        try:
            df = pd.read_csv("./SOURCE_DOCUMENTS/questions_stolot.csv", dtype='object')
            loader = DataFrameLoader(df, page_content_column='question')
            documents = loader.load()

            rec_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap,
                                                    length_function=len
                                                    )
            texts = rec_text_splitter.split_documents(documents)
        except:
            print('Датасет не найден или произошла ошибка при обработке')
        else:
            print('Датасет успешно загружен')
    
    return texts



def upload_to_vectorstore(texts, collection_name=None):
    """
    This function is used to upload the data to the vector store.
    It is used to upload the data to the vector store and create a collection.

    texts: list - the list of texts to upload to the vector store
    collection_name: str, default=None - the name of the collection to create in the vector store    
    """

    flag = True
    try:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Загрузка модели для эмбеддингов: SUCCESS")
    except Exception as e:
        print("Ошибка при загрузке модели эмбеддингов:", e)
        flag = False
        
    if flag:
        try:
            chroma_client = chromadb.HttpClient(settings=Settings(
                allow_reset=True,
                chroma_api_impl='chromadb.api.fastapi.FastAPI',
                chroma_server_host='localhost',
                chroma_server_http_port='8000')
            )
            print("Подключение к векторной БД: SUCCESS")
        except Exception as e:
            print("Ошибка при подключении к векторной БД:", e)
            flag = False
    
    if flag:
        try:
            if collection_name is not None:
                collection = chroma_client.get_or_create_collection(name=collection_name,
                                                    metadata={"hnsw:space": "cosine"},
                                                    embedding_function=embedding_function
                                                    )
            else:
                collection_name = 'book'
                collection = chroma_client.get_or_create_collection(name=collection_name,
                                                    metadata={"hnsw:space": "cosine"},
                                                    embedding_function=embedding_function
                                                    )
            print(f"Создание коллекции {collection_name}: SUCCESS")
        except Exception as e:
            print("Ошибка при создании коллекции:", e)
            flag = False
    
    if flag:
        try:
            print("Загружаю данные в векторную БД...")
            for doc in texts:
                collection.add(
                    documents=doc.page_content,
                    metadatas=doc.metadata,
                    ids=doc.metadata['id']
                )
            print("Загрузка данных в векторную БД: SUCCESS")
        except Exception as e:
            print("Ошибка при загрузке данных в векторную БД:", e)


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
            chroma_server_http_port='8000')
        )
        print("Подключение к векторной БД: SUCCESS")
        return chroma_client
    except Exception as e:
        print("Ошибка при подключении к векторной БД:", e)
        return None

def vectorstore_query(collection, question, n_results):
    """
    This function is used to query the vector store.
    It is used to query the vector store to get the response to a question.
    
    collection: collection - the collection to query in the vector store
    question: str - the question to query in the vector store
    n_results: int - the number of results to return
    """

    response = collection.query(
    # query_embeddings=embedding_function(question),
    query_texts=question,
    n_results=n_results,
    # include=["documents"],
    # where={"metadata_field":"answer"}, # где искать
    # where_document={"$contains":"$search_string"}
    )

    # убираем одинаковые ответы
    response_list = []
    for repl in response['metadatas'][0]:
        response_list.append(repl['answer'])

    response_list = set(response_list)

    # собираем итоговый ответ
    vector_db_response = ""
    for repl in response_list:
        vector_db_response += repl
        # if repl['url'] is not False:
        #     full_response += f" Ссылка на подробную информацию: {repl['url']}\n"
        # else:
        #     full_response += "\n"
    
    return vector_db_response






# def main():
#     texts = data_preparation()
#     collection = vector_store_upload(texts, collection_name='book')



# if __name__ == "__main__":
#     main()