import pytest
import asyncio
import logging
from httpx import AsyncClient
from app_api_no_llm import app

@pytest.mark.asyncio
async def test_concurrent_requests():  

    async with AsyncClient(app=app, base_url="http://localhost:8008") as client:
        # Список для хранения всех отправленных запросов
        requests = []

        # Отправляем 5 запросов асинхронно
        for i in range(10):
            # Создаем корутину для отправки запроса и добавляем ее в список
            request_coroutine = client.post(
                "/llm",
                json={"user_id": 123, "question_string": "Test question"},
                headers={"API":"131e12aa46252f4da6920dd2feccc94978688eab3a96337ba4b67a945eac1308"}
            )
            requests.append(request_coroutine)

        # Ожидаем завершения всех запросов
        responses = await asyncio.gather(*requests)

        # Проверяем статус-коды всех ответов
        for response in responses:
            assert response.status_code == 200

# pytest --log-cli-level=DEBUG