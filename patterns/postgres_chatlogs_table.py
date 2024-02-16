import psycopg2

conn = psycopg2.connect(
    dbname="postgres",
    user="aesedeu",
    password="aesedeu",
    host="localhost",
    port="5432"
)

def main():
    cur = conn.cursor()

    

    try:
        sql = """
        SELECT * FROM chat_logs LIMIT 1
        """
        cur.execute(sql)
        conn.commit()

        print("Таблица chat_logs уже существует")
    except:
        sql = """
        CREATE TABLE IF NOT EXISTS chat_logs (
        id SERIAL PRIMARY KEY,
        user_id INT,
        date DATE,
        time TIME,
        client_ip VARCHAR(15),
        user_question TEXT,
        ai_response TEXT,
        response_time FLOAT
        );
        """
        cur.execute(sql)
        conn.commit()
        
        print("Таблица chat_logs успешно создана")
    finally:
        # Закрываем курсор и соединение
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()