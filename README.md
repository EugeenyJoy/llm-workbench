Локально:

- python3 -m venv venv - создать виртуальное окружение
- source venv/bin/activate - войти в виртуальное окружение
- pip install -r requirements.txt - установвка зависимостей: numpy tqdm(показывает прогресс обучения) sentencepiece (токенизация как в GPT / LLaMA)
- python train.py - Запуск обучения

Использование в Colab:

1. !git clone YOUR_REPO

2. %cd mini_llm

3. !pip install -r requirements.txt

4. !python train.py

Для загрузки обученной модели
После обучения в Colab добавить ячейку:

Python

from google.colab import files
files.download("model.pt")
Файл загрузится на компьютер.
