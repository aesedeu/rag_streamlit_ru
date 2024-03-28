import streamlit as st
from streamlit.web import cli as stcli
from streamlit import runtime
runtime.exists()
import sys
# from lib.llm_api_response import get_llm_api_response
# from lib.postgres_setup import upload_to_postgres
import time
import click


st.markdown("<h1 style='text-align: center; color: orange;'>Тестовый чат-бот</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: right; color: grey;'>Built by <a href='https://synchro.pro/'>Synchro</a></h6>", unsafe_allow_html=True)

st.markdown("<div style='text-align: left; color:red;'>Версия: v0.1 </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>Алгоритм: RAG </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>Векторное хранилище: --HIDDEN-- </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>Базовая модель: --HIDDEN-- </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>Finetuning: LoRA + PEFT </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>Функция создания эмбеддингов: --HIDDEN-- </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>GPU: nvidia-A100-40GB </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>Температура генерации: 0.1 </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>Ограничение тематики диалога: НЕТ </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>ПД пользователя в промпте: НЕТ </div>", unsafe_allow_html=True)

# @click.command()
# @click.option("-lr", "--lora", is_flag=True, default=True,  help='---')
def main(
        # lora:bool=False
    ):
    
    def foo():
        return 100500
    
    @st.cache_resource
    def test_data():
        x = foo()
        time.sleep(2)
        result = 10*3 + x
        return result

    test = test_data()

    # Инициализируем историю сообщений
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Отображаем историю сообщений
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar=message.get("avatar")):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Получаем сообщение от пользователя
    prompt = st.chat_input("Введите ваше сообщение")
    if prompt:    
        with st.chat_message(name="user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Отвечаем пользователю
        with st.chat_message(name="assistant", avatar="./icons/assistant_icon.jpg"):
            with st.spinner('Собираю информацию по Вашему вопросу...⏳'):
                response = prompt + f"\nВремя генерации ответа: {test} сек"
                
                # Добавляем ответ в postgres
                # upload_to_postgres(api_response)

                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "./icons/assistant_icon.jpg"})
        
    else:
        response = f"Здравствуйте, меня зовут Степан. Я - искусственный интеллект. Какой у Вас вопрос?"
        with st.chat_message(name="assistant", avatar="./icons/assistant_icon.jpg"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "./icons/assistant_icon.jpg"})

if __name__ == '__main__':
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())