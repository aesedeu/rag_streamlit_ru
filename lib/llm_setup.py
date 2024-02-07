import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from lib.vector_db_setup import vectorstore_query

def create_input_message(question, tokenizer, collection):
    """
    Создание входного сообщения для модели
    
    Args:
    question: str - вопрос пользователя
    tokenizer: AutoTokenizer - токенизатор
    collection: str - название коллекции векторов"""

    vector_db_response = vectorstore_query(
        collection,
        question = question,
        n_results=3
    )
    SYSTEM_PROMPT = f"Ты - доброжелательный русскоязычный ассистент Степан. Отвечаешь только на вопросы о лотереях. Тебя создали в компании Синхро. Ты отвечаешь на вопрос, используя только следующую данную тебе информацию.\nИнформация: {vector_db_response}"
    QUESTION = question
    chat = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}"},
        {"role": "user", "content": f"{QUESTION}"},
        {"role": "assistant", "content":""}
    ]

    input_message = ""
    for i in chat:
        input_message += tokenizer.bos_token + i['role'] + '\n' + i['content'] + tokenizer.eos_token + '\n'
    input_message = input_message[:-5].strip() + "\n"

    return input_message

def initialize_model(base_model, lora_adapter):
    """
    Инициализация модели
    
    Args:
    base_model: str - название базовой модели
    lora_adapter: str - название адаптера LoRA"""
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True,
        # load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    model = PeftModel.from_pretrained(
        model,
        lora_adapter,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    model.eval()
    model = model.merge_and_unload() #  ОБЯЗАТЕЛЬНО!!! После загрузки модели в память, нужно вызвать метод merge_and_unload() для слияния адаптеров и выгрузки модели из памяти

    return model

def generate_llm_response(question, model, collection, tokenizer):
    """
    Генерация ответа на вопрос с помощью модели
    
    Args:
    question: str - вопрос пользователя
    model: PeftModel - модель
    collection: str - название коллекции векторов
    tokenizer: AutoTokenizer - токенизатор
    """
    input_message = create_input_message(question, tokenizer, collection)
    input_data = tokenizer(input_message, return_tensors="pt", add_special_tokens=False)
    input_data = {k: v.to("cuda:0") for k, v in input_data.items()}

    generation_config = GenerationConfig(
        bos_token_id = 1,
        do_sample = True,
        eos_token_id = 2,
        max_length = 1024,
        repetition_penalty=1.1,
        # length_penalty=0.1,
        no_repeat_ngram_size=15,
        pad_token_id = 2,
        temperature = 0.1,
        # top_p = 0.9,
        # top_k= 40,
        # low_memory=True
    )

    output_data = model.generate(
        **input_data,
        generation_config = generation_config
    )[0]
    
    output_data = output_data[len(input_data["input_ids"][0]):]
    output_message = tokenizer.decode(output_data, skip_special_tokens=True)

    return output_message