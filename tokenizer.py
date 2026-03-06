# ==========================================
# TOKENIZER.PY
# ==========================================
#
# НАЗНАЧЕНИЕ:
# обучает BPE токенизатор (SentencePiece)
# на всех текстах из папки datasets/
#
# Токенизатор переводит текст → числа
# которыми уже обучается нейросеть.
#
# Запускается ОДИН РАЗ перед обучением модели.
#
# После работы появятся файлы:
#
# tokenizer/tokenizer.model
# tokenizer/tokenizer.vocab
#
# ==========================================


import sentencepiece as spm
import os


# ==========================================
# БАЗОВАЯ НАСТРОЙКА ПУТЕЙ
# ==========================================

DATASET_FOLDER = "datasets"
TOKENIZER_FOLDER = "tokenizer"

TEMP_DATA_FILE = "tokenizer_training_text.txt"


# ==========================================
# СОБИРАЕМ ВСЕ ФАЙЛЫ ИЗ DATASETS
# ==========================================
# БАЗОВАЯ ЛОГИКА
#
# читаем ВСЕ .txt файлы
# и объединяем их во временный файл
#
# Это нужно потому что sentencepiece
# принимает один файл на вход
# ==========================================

def collect_dataset():

    texts = []

    print("Scanning datasets folder...")

    for root, dirs, files in os.walk(DATASET_FOLDER):

        for file in files:

            if file.endswith(".txt"):

                path = os.path.join(root, file)

                print("Loading:", path)

                with open(path, "r", encoding="utf-8") as f:

                    texts.append(f.read())

    print("Total files loaded:", len(texts))

    full_text = "\n".join(texts)

    print("Total characters:", len(full_text))

    return full_text


# ==========================================
# ОБУЧЕНИЕ TOKENIZER
# ==========================================
# БАЗОВАЯ ЛОГИКА
#
# создаёт BPE токены
#
# vocab_size = размер словаря
#
# 2000 — хороший размер
# для маленькой LLM
# ==========================================

def train_tokenizer():

    os.makedirs(TOKENIZER_FOLDER, exist_ok=True)

    text = collect_dataset()

    print("Writing temporary dataset file...")

    with open(TEMP_DATA_FILE, "w", encoding="utf-8") as f:

        f.write(text)

    print("Training tokenizer...")

    spm.SentencePieceTrainer.train(

        input=TEMP_DATA_FILE,

        model_prefix="tokenizer/tokenizer",

        vocab_size=2000,

        # ==================================
        # СПЕЦИАЛЬНЫЕ ТОКЕНЫ (ВАЖНО)
        # ==================================

        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3

    )

    print("Tokenizer trained successfully")

    print("Saved to tokenizer/tokenizer.model")

    os.remove(TEMP_DATA_FILE)

    print("Temporary dataset file removed")


# ==========================================
# ЗАГРУЗКА TOKENIZER
# ==========================================
# используется train.py и chat.py
# ==========================================

def load_tokenizer():

    sp = spm.SentencePieceProcessor()

    sp.load("tokenizer/tokenizer.model")

    return sp


# ==========================================
# ТОЧКА ВХОДА СКРИПТА
# ==========================================
# запускается если выполнить
#
# python tokenizer.py
# ==========================================

if __name__ == "__main__":

    train_tokenizer()