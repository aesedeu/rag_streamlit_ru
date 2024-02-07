from fastapi import FastAPI, Request, HTTPException, status, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from datetime import datetime
from pydantic import BaseModel
import json
import datetime as dt
import time
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from lib.vector_db_setup import get_texts, upload_to_vectorstore, vectorstore_query, get_chroma_client
import chromadb
from chromadb.config import Settings
from peft import PeftConfig
from transformers import AutoTokenizer
from lib.llm_setup import initialize_model, generate_llm_response
from lib.postgres_setup import upload_to_postgres

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

app = FastAPI()
# Ключ для проверки API
load_dotenv()
load_dotenv(".env")
api_keys = os.environ.get("API_KEYS").split(',')
api_key_header = APIKeyHeader(name="API")

# Подключаемся к векторному хранилищу
chroma_client = get_chroma_client()
# Получаем коллекцию
collection = chroma_client.get_collection('book')

lora_adapter = "IlyaGusev/saiga_mistral_7b"
config = PeftConfig.from_pretrained(lora_adapter)
base_model = config.base_model_name_or_path

# Настройка токенизатора
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right' # ???

try:
    model = initialize_model(
        base_model=base_model,
        lora_adapter=lora_adapter
    )
except Exception as e:
    print(e)

class IncomeMessage(BaseModel):
    user_id: int
    question_string: str


@app.get('/')
def read_root():
    return {"Hello": "World"}


@app.post('/llm')
async def add_numbers(
        request: Request,
        income_message: IncomeMessage,
        api_key: str = Security(api_key_header)
        ):
    # Проверяем API ключ
    if api_key not in api_keys:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")

    start = time.time()
    start_time = dt.datetime.now().strftime("%H:%M:%S")
    client_host = request.client.host

    # Генерация ответа LLM
    ai_response = generate_llm_response(
        question=income_message.question_string,
        model=model,
        collection=collection,
        tokenizer=tokenizer
    )

    end = time.time()
    time_spent = round((end-start), 2)
    
    result = {
        "user_id": income_message.user_id,
        "date": dt.datetime.now().strftime("%Y-%m-%d"),
        "time": start_time,
        "client_ip": client_host,
        "user_question": income_message.question_string,
        "ai_response": ai_response,
        "response_time": time_spent
    }
    
    # # Добавляем ответ в postgres
    # upload_to_postgres(result)


    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, # "app_api_llm:app"
                host="0.0.0.0",
                port=8008,
                # workers=4
                )