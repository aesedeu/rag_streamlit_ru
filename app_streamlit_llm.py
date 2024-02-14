import streamlit as st
# from lib.llm_api_response import get_llm_api_response
# from lib.postgres_setup import upload_to_postgres
from lib.vector_db_setup import get_texts, upload_to_vectorstore, vectorstore_query, get_chroma_client
from lib.llm_setup import initialize_model, generate_llm_response
from peft import PeftConfig
from transformers import AutoTokenizer

import time

st.markdown("<h1 style='text-align: center; color: orange;'>Тестовый чат-бот проекта Столото </h1>", unsafe_allow_html=True)
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
st.markdown("<div style='text-align: left; color:orange;'> БОТ НЕ ЗАПОМИНАЕТ ПРЕДЫДУЩИЕ СООБЩЕНИЯ! </div>", unsafe_allow_html=True)

# Настройка токенизатора
@st.cache_resource
def connection_tokenizer():
    # lora_adapter="IlyaGusev/saiga_mistral_7b"
    lora_adapter = "IlyaGusev/saiga2_13b_lora"
    config = PeftConfig.from_pretrained(lora_adapter)
    base_model = config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right' # ???
    return tokenizer

@st.cache_resource
def connection_get_chroma_client():
    chroma_client = get_chroma_client()
    return chroma_client

@st.cache_resource
def connection_initialize_model():
    # lora_adapter="IlyaGusev/saiga_mistral_7b"
    lora_adapter = "IlyaGusev/saiga2_13b_lora"
    config = PeftConfig.from_pretrained(lora_adapter)
    base_model = config.base_model_name_or_path
    model = initialize_model(
        base_model=base_model,
        lora_adapter=lora_adapter,
        # bnb=False
    )
    return model

tokenizer = connection_tokenizer()
collection = connection_get_chroma_client().get_collection('book')
model = connection_initialize_model()


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
            response = generate_llm_response(
                question=prompt,
                model=model,
                collection=collection,
                tokenizer=tokenizer,
                source_file_type='table'
            )
            
            # Добавляем ответ в postgres
            # upload_to_postgres(api_response)

            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "./icons/assistant_icon.jpg"})
    
else:
    response = f"Здравствуйте, меня зовут Степан. Я - искусственный интеллект, созданный для помощи Вам по вопросам, связанными со Столото. Какой у Вас вопрос?"
    with st.chat_message(name="assistant", avatar="./icons/assistant_icon.jpg"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "./icons/assistant_icon.jpg"})
