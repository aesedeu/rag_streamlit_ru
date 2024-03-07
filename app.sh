. .env
. .venv/bin/activate

printf 'Do you want to upload data to vectorstore (y/n)?'
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    printf "Enter full filename in '$SOURCE_DOCUMENTS_FOLDER' folder: "
    read answer_file_name
    printf "Enter the name of collection: "
    read answer_collection_name
    echo "Uploading data from '$SOURCE_DOCUMENTS_FOLDER/$answer_file_name' to '$answer_collection_name' collection..."

    python data_upload_to_chroma.py --data_path=$answer_file_name --collection_name=$answer_collection_name
fi

printf 'Do you want to run test app without LLM (y/n)?'
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Running app in test mode"
    streamlit run $PROJECT_DIRECTORY/patterns/no_llm/app_streamlit_no_llm.py
else
    echo "done"
fi