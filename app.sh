. .env
. .venv/bin/activate

printf 'Do you want to upload data to vectorstore (y/n)? '
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Uploading data from '$PROJECT_DIRECTORY/$DATA_PATH' to '$COLLECTION_NAME' collection..."
    python data_upload_to_chroma.py --data_path=$DATA_PATH --collection_name=$COLLECTION_NAME
fi

printf 'Do you want to run test app without LLM (y/n)? '
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Running app in test mode..."
    sleep 1
    streamlit run $PROJECT_DIRECTORY/patterns/no_llm/app_streamlit_no_llm.py
else
    echo "done"
fi