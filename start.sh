touch .env
. .env
echo PROJECT_DIRECTORY=$PWD >> .env
echo SOURCE_DOCUMENTS_FOLDER=SOURCE_DOCUMENTS >> .env

mkdir $SOURCE_DOCUMENTS_FOLDER
mkdir logs
mkdir dialogs

echo "========================================="
echo "Creating vectorstore ChromaDB..."
echo "========================================="
docker-compose up -d

echo "========================================="
echo "Installing virtual environment with dependencies..."
echo "========================================="
python3.10 -m venv .venv
echo "$(pwd)" > .venv/lib/python3.10/site-packages/my_project.pth 
. .venv/bin/activate
pip install -r requirements.txt

# Running main app
sh app.sh