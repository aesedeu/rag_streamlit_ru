. .env
. .venv/bin/activate

printf 'Do you want to upload data to vectorstore? (y/n)'
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    printf "WARNING: CURRENTLY SUPPORTED FORMATS: CSV/TXT/PDF"
    printf "\nEnter the full filename in '$SOURCE_DOCUMENTS_FOLDER' folder: "
    read answer_file_name
    printf "Enter the name of the collection: "
    read answer_collection_name

    printf 'Do you want to change the chunk size and chunk overlap? Default: 100/50 (y/n)'
    read answer
    if [ "$answer" != "${answer#[Yy]}" ] ;then
        printf "Enter 'chunk_size': "
        read chunk_size
        printf "Enter 'chunk_overlap': "
        read chunk_overlap

        echo "Uploading data from '$SOURCE_DOCUMENTS_FOLDER/$answer_file_name' to '$answer_collection_name' collection..."
        python data_upload_to_chroma.py \
            --data_path=$answer_file_name \
            --collection_name=$answer_collection_name \
            --chunk_size=$chunk_size \
            --chunk_overlap=$chunk_overlap
    else
        echo "Uploading data from '$SOURCE_DOCUMENTS_FOLDER/$answer_file_name' to '$answer_collection_name' collection..."
        python data_upload_to_chroma.py \
            --data_path=$answer_file_name \
            --collection_name=$answer_collection_name
    fi
fi

printf 'Do you want to run test app without LLM (y/n)?'
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Running app in test mode"
    streamlit run $PROJECT_DIRECTORY/patterns/no_llm/app_streamlit_no_llm.py
else
    echo "done"
fi