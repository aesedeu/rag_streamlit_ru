import psycopg2
import json

conn = psycopg2.connect(
    dbname="postgres",
    user="aesedeu",
    password="aesedeu",
    host="localhost",
    port="5432"
)

def upload_to_postgres(api_response):
    cur = conn.cursor()

    # SQL-запрос для вставки данных из JSON-файла в таблицу
    sql = """INSERT INTO chat_logs (user_id, date, time, client_ip, user_question, ai_response, response_time) 
         VALUES (%s, %s, %s, %s, %s, %s, %s)"""

    cur.execute(sql, (
        api_response['user_id'],
        api_response['date'],
        api_response['time'],
        api_response['client_ip'],
        api_response['user_question'],
        api_response['ai_response'],
        api_response['response_time']
    ))
    conn.commit()
    cur.close()
    conn.close()
