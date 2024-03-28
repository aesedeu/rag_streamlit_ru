import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from lib.vector_db_setup import vectorstore_query
import json
import datetime
import time
import yaml

config = yaml.safe_load(open('./config.yaml'))

def create_input_message(
        question:str,
        tokenizer: AutoTokenizer
    ):
    """
    Создание входного сообщения для модели.
    
    Args:
    question: str - вопрос пользователя
    tokenizer: AutoTokenizer - токенизатор
    """

    SYSTEM_PROMPT = config['llm']['system_prompt']

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


def create_input_rag_message(question, tokenizer, collection, source_file_type, n_results):
    """
    Создание входного сообщения для модели с использованием векторной БД chromadb.
    
    Args:
    question: str - вопрос пользователя
    tokenizer: AutoTokenizer - токенизатор
    collection: str - название коллекции векторов
    source_file_type: str - тип исходного файла по которому создавалась коллекция
    n_results: int - количество результатов из векторного поиска
    """

    vector_db_response = vectorstore_query(
        collection=collection,
        source_file_type=source_file_type,
        question=question,
        n_results=n_results
    )
    SYSTEM_PROMPT = config['llm']['system_prompt_rag'].format(vector_db_response)
    # SYSTEM_PROMPT = f"""Ты - пьяный пират, который ищет свой корабль. Разговаривай как пьяный пират."""

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

def initialize_model(
        base_model:str,
        bnb:bool
    ):
    """
    Инициализация модели
    
    Args:
    base_model: str - название базовой модели
    bnb: bool - применение квантизации модели с помощью BitsAndBytes
    """

    if bnb:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=False, # Загружает модель в 4-битном формате для уменьшения использования памяти.
                load_in_8bit=True,
                bnb_4bit_quant_type="fp4", # Указывает тип квантования, в данном случае "nf4" (nf4/dfq/qat/ptq/fp4)
                bnb_4bit_compute_dtype="float16", # Устанавливает тип данных для вычислений в 4-битном формате как float16.
                bnb_4bit_use_double_quant=False # Указывает, что не используется двойное квантование.
            ),
            device_map="cuda"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # load_in_8bit=True,
            # load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model.eval()
    
    return model, tokenizer

def initialize_lora_model(
        base_model:str,
        lora_adapter:str,
        bnb:bool=False
    ):
    """
    Инициализация модели c LoRA-адаптером
    
    Args:
    base_model: str - название базовой модели
    lora_adapter: str - название адаптера LoRA
    bnb: bool - применение квантизации модели с помощью BitsAndBytes
    """
    if bnb:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=False, # Загружает модель в 4-битном формате для уменьшения использования памяти.
                load_in_8bit=True,
                bnb_4bit_quant_type="fp4", # Указывает тип квантования, в данном случае "nf4" (nf4/dfq/qat/ptq/fp4)
                bnb_4bit_compute_dtype="float16", # Устанавливает тип данных для вычислений в 4-битном формате как float16.
                bnb_4bit_use_double_quant=False # Указывает, что не используется двойное квантование.
            ),
            device_map="cuda"
        )
        model = PeftModel.from_pretrained(
            model,
            lora_adapter,
            torch_dtype="auto",
            device_map="cuda"
        )
    else:
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

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model.eval()
    model = model.merge_and_unload() #  ОБЯЗАТЕЛЬНО!!! После загрузки модели в память, нужно вызвать метод merge_and_unload() для слияния адаптеров и выгрузки модели из памяти

    return model, tokenizer

def generate_llm_response(
        question:str,
        model,
        tokenizer,
        collection=None,
        source_file_type:str=None,
        n_results:int=None
    ):
    """
    Генерация ответа на вопрос с помощью модели
    
    Args:
    question: str - вопрос пользователя
    model: PeftModel - модель
    collection: str - название коллекции векторов
    tokenizer: AutoTokenizer - токенизатор
    source_file_type: str - 'csv' или 'txt'
    """

    start = time.time()
    
    if collection==None and source_file_type==None and n_results==None:
        input_message = create_input_message(question, tokenizer)
    else:
        input_message = create_input_rag_message(question, tokenizer, collection, source_file_type, n_results)

    input_data = tokenizer(input_message, return_tensors="pt", add_special_tokens=False)
    input_data = {k: v.to("cuda:0") for k, v in input_data.items()}

    generation_config = GenerationConfig(
        bos_token_id = config['llm']['generation_config']['bos_token_id'],
        do_sample = config['llm']['generation_config']['do_sample'],
        eos_token_id = config['llm']['generation_config']['eos_token_id'],
        max_length = config['llm']['generation_config']['max_length'],
        max_new_tokens = config['llm']['generation_config']['max_new_tokens'],
        repetition_penalty= config['llm']['generation_config']['repetition_penalty'],
        length_penalty = config['llm']['generation_config']['length_penalty'],
        no_repeat_ngram_size = config['llm']['generation_config']['no_repeat_ngram_size'],
        pad_token_id = config['llm']['generation_config']['pad_token_id'],
        temperature = config['llm']['generation_config']['temperature'],
        top_p = config['llm']['generation_config']['top_p'],
        top_k = config['llm']['generation_config']['top_k'],
        low_memory = config['llm']['generation_config']['low_memory'],
    )

    output_data = model.generate(
        **input_data,
        generation_config = generation_config
    )[0]
    
    output_data = output_data[len(input_data["input_ids"][0]):]
    output_message = tokenizer.decode(output_data, skip_special_tokens=True)

    end = time.time()
    time_spent = round((end-start), 2)

    filename = "./dialogs/q_a.json"
    # Читаем существующие данные в моем джейсоне
    try:
        with open(filename, "r") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}  # Используем пустой словарь

    # Формирую новую запись в джейсоне
    timestamp = datetime.datetime.now().timestamp()
    existing_data[str(timestamp)] = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "time_utc": datetime.datetime.now().strftime("%H:%M:%S"),
        "question": question,  # Добавляем отправленные данные
        "response": output_message,
        "time_spent": time_spent
    }

    # Сохраняю обновленные данные
    with open(filename, "w") as file:
        json.dump(existing_data, file, indent=4)

    return output_message