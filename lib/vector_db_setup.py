import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import DataFrameLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 





def get_texts(
        file_name,
        content_column=None,
        chunk_size=300,
        chunk_overlap=300
    ):

    """
    This function is used to prepare the data for the vector store.
    It is used to load the dataset and split the documents into smaller chunks.
    
    chunk_size: int, default=500 - the size of the chunks to split the documents into
    chunk_overlap: int, default=500 - the overlap between the chunks
    """
    dataset_path = "./SOURCE_DOCUMENTS/" + file_name

    if dataset_path.endswith('.csv'):
        print(f'Выбраны данные из файла: {dataset_path}')
        try:
            df = pd.read_csv(dataset_path, dtype='object')
            loader = DataFrameLoader(df, page_content_column=content_column)
            documents = loader.load()

            rec_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap,
                                                    length_function=len
                                                    )
            texts = rec_text_splitter.split_documents(documents)
        except Exception as e:
            print(e)
        else:
            print('Загрузка датасета: SUCCESS')
        return texts
    elif dataset_path.endswith('.txt'):
        print(f'Выбраны данные из файла: {dataset_path}')
        try:
            loader = TextLoader(dataset_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=300)
            texts = text_splitter.split_documents(documents)
            # with open(dataset_path, 'r') as file:
            #     text = file.readlines()
            #     text = ''.join(text)
            # rec_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
            #                                                 chunk_overlap=chunk_overlap,
            #                                                 length_function=len
            #                                                 )
            # texts = rec_text_splitter.split_text(text)
            # print(f'Total number of chunks: {len(texts)}\n')
        except Exception as e:
            print(e)
        else:
            print('Загрузка датасета: SUCCESS')            
        return texts

        



def upload_to_vectorstore(texts, collection_name):

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
            print("Подключение к CHROMADB: SUCCESS")
        except Exception as e:
            print("Ошибка при подключении к CHROMADB:", e)
            flag = False
    
    if flag:
        for i in chroma_client.list_collections():
            if i.name == collection_name:
                chroma_client.delete_collection(collection_name)
                print(f'Была обнаружена и удалена существующая коллекция с именем "{collection_name}"')
        try:
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
            counter = 0
            for doc in texts:
                counter += 1
                collection.add(
                    documents=doc.page_content,
                    metadatas=doc.metadata,
                    ids=['id'+str(counter)]
                    # ids=doc.metadata['id']
                )
            print("Загрузка данных в CHROMADB: SUCCESS")
        except Exception as e:
            print("Ошибка при загрузке данных в CHROMADB:", e)


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
        print("Подключение к CHROMADB: SUCCESS")
        return chroma_client
    except Exception as e:
        print("Ошибка при подключении к CHROMADB:", e)
        return None

def vectorstore_query(collection, source_file_type, question, n_results):
    """
    This function is used to query the vector store.
    It is used to query the vector store to get the response to a question.
    
    collection: collection - the collection to query in the vector store
    question: str - the question to query in the vector store
    n_results: int - the number of results to return
    """
    if source_file_type == 'table':
        response = collection.query(
            query_texts=question,
            n_results=n_results
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
    
    elif source_file_type == 'text':
        response = collection.query(
        # query_embeddings=embedding_function(question),
        query_texts=question,
        n_results=n_results,
        # include=["documents"],
        # where={"metadata_field":"is_equal_to_this"}, # где искать
        # where_document={"$contains":"$search_string"}
    )
        vector_db_response = " ".join(response["documents"][0])
        return vector_db_response




# def main():
#     texts = data_preparation()
#     collection = vector_store_upload(texts, collection_name='book')



# if __name__ == "__main__":
#     main()