from lib.vector_db_setup import get_texts, upload_to_vectorstore
import warnings
warnings.simplefilter("ignore", UserWarning)
import warnings
import click

# @click.command()
# @click.option("-dp", "--data_path", help='Path to your data which will be uploaded to the verctorstore. CSV/TXT formats are supported')
# @click.option("-cn", "--collection_name", help='Collection name in vectorstore')
def main(
    # data_path:str,
    # collection_name:str,
    ):
    data_path = input(f"Введите путь к файлу в директории SOURCE_DOCUMENTS: ")
    collection_name = input(f"Введите название коллекции: ")

    if data_path.endswith('.csv'):
        texts = get_texts(
            file_name=data_path,
            content_column='question'
        )
    elif data_path.endswith('.txt'):
        texts = get_texts(
            file_name=data_path,
            chunk_size=150,
            chunk_overlap=150
        )

    upload_to_vectorstore(
        texts,
        collection_name=collection_name
    )

if __name__ == "__main__":
    main()