import requests

URL = "http://localhost:8008/llm"

def get_llm_api_response(question, api_key, user_id):
    POST_JSON = {
        "user_id": user_id,
        "question_string": question
    }
    r = requests.post(
        url=URL,
        headers={
            "Content-Type": "application/json",
            "API": api_key
        },
        json=POST_JSON
    )
    
    return r.json()