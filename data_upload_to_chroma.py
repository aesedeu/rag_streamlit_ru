from lib.vector_db_setup import get_texts
import warnings
warnings.simplefilter("ignore", UserWarning)
import warnings
import click

import os
import datetime as dt
import logging
from dotenv import load_dotenv
load_dotenv()
PROJECT_DIRECTORY = os.getenv('PROJECT_DIRECTORY')

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

@click.command()
@click.option("-dp", "--data_path", help='Path to your data which will be uploaded to the verctorstore. CSV/TXT formats are supported')
@click.option("-cn", "--collection_name", help='Collection name in vectorstore')
@click.option("-cs", "--chunk_size", default=100, help='Size of splitting chunks')
@click.option("-co", "--chunk_overlap", default=50, help='Size of chunks overlap')
def main(
    data_path:str,
    collection_name:str,
    chunk_size:int,
    chunk_overlap:int
    ):
    assert data_path is not None and collection_name is not None, "Please provide correct 'data_path' and 'collection_name' or type --help for FAQ"
    # data_path = input(f"Введите путь к файлу в директории SOURCE_DOCUMENTS: ")
    # collection_name = input(f"Введите название коллекции: ")

    texts = get_texts(
            file_name=data_path,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    # upload_to_vectorstore(
    #     texts,
    #     collection_name=collection_name
    # )
    
if __name__ == "__main__":
    main()