from lib.vector_db_setup import get_texts, upload_to_vectorstore
import warnings
warnings.simplefilter("ignore", UserWarning)
import warnings

def main():
    file_name = input(f"Введите путь к файлу в директории SOURCE_DOCUMENTS: ")
    collection_name = input(f"Введите название коллекции: ")

    if file_name.endswith('.csv'):
        texts = get_texts(
            file_name=file_name,
            content_column='question'
        )
    elif file_name.endswith('.txt'):
        texts = get_texts(
            file_name=file_name,
            chunk_size=150,
            chunk_overlap=150
        )

    upload_to_vectorstore(
        texts,
        collection_name=collection_name
    )

if __name__ == "__main__":
    main()