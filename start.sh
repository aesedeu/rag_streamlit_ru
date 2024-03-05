. .venv/bin/activate

printf 'Do you want to run test app without LLM (y/n)? '
read answer

if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "running app in test mode"
    sleep 1
    streamlit run patterns/api+no_llm/app_streamlit_no_llm.py
else
    echo "running app with RAG"
    sleep 1
    streamlit run rag_streamlit.py
fi