import streamlit as st
# from lib.llm_api_response import get_llm_api_response
# from lib.postgres_setup import upload_to_postgres
import time

st.markdown("<h1 style='text-align: center; color: orange;'>Тестовый чат-бот проекта Столото </h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: right; color: grey;'>Built by Synchro </a></h6>", unsafe_allow_html=True)

st.markdown("<div style='text-align: left; color:red;'>Версия: v0.1 </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>Алгоритм: RAG </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>Векторное хранилище: chromadb </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>Базовая модель: OpenOrca-7B </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>Finetuning: LoRA + PEFT </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>Функция создания эмбеддингов: all-MiniLM-L6-v2 </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>GPU: nvidia-A100-40GB </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>Температура генерации: 0.1 </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>Ограничение тематики диалога: НЕТ </div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: left; color:red;'>ПД пользователя в промпте: НЕТ </div>", unsafe_allow_html=True)



@st.cache_resource
def test_data():
    print('TEST_TEST_TEST')
    time.sleep(5)
    return 10*3

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
            # api_response = get_llm_api_response(
            #     question=prompt,
            #     api_key="131e12aa46252f4da6920dd2feccc94978688eab3a96337ba4b67a945eac1308",
            #     user_id=906
            # )
            # response = api_response['ai_response'] + f"\nВремя генерации ответа: {api_response['response_time']} сек"
            test = test_data()
            response = prompt + f"\nВремя генерации ответа: {test} сек"
            
            # Добавляем ответ в postgres
            # upload_to_postgres(api_response)

            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "./icons/assistant_icon.jpg"})
    
else:
    response = f"Здравствуйте, меня зовут Степан. Я - искусственный интеллект, созданный для помощи Вам по вопросам, связанными со Столото. Какой у Вас вопрос?"
    with st.chat_message(name="assistant", avatar="./icons/assistant_icon.jpg"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "./icons/assistant_icon.jpg"})
