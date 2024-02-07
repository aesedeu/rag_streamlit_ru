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
from lib.postgres_setup import upload_to_postgres

app = FastAPI()
# Ключ для проверки API
load_dotenv()
load_dotenv(".env")
api_keys = os.environ.get("API_KEYS").split(',')
# Создаем экземпляр APIKeyHeader с именем "API"
api_key_header = APIKeyHeader(name="API")

class IncomeMessage(BaseModel):
    user_id: int
    question_string: str


@app.get('/')
async def read_root():
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
    ai_response = income_message.question_string + ' GOT!!!'
    time.sleep(10)
    end = time.time()
    time_spent = round((end-start), 2)
    time_spent
    result = {
        "user_id": income_message.user_id,
        "date": dt.datetime.now().strftime("%Y-%m-%d"),
        "time": start_time,
        "client_ip": client_host,
        "user_question": income_message.question_string,
        "ai_response": ai_response,
        "response_time": time_spent
    }
    # Добавляем ответ в postgres
    upload_to_postgres(result)

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,
                host="0.0.0.0",
                port=8008,
                # workers=4
                )