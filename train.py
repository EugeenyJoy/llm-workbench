import torch
import torch.nn.functional as F
import os

from model import GPT
from config import Config
from tokenizer import load_tokenizer

from torch.cuda.amp import autocast, GradScaler


# ==========================================
# ОПРЕДЕЛЯЕМ УСТРОЙСТВО
# ==========================================
# если есть GPU — используем его
# если нет — работаем на CPU
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


config = Config()


# ==========================================
# ЗАГРУЗКА ТОКЕНИЗАТОРА
# ==========================================

sp = load_tokenizer()

# ======================================
# ЧИТАЕМ ВСЕ ФАЙЛЫ ИЗ datasets
# ======================================

texts = []

dataset_folder = "datasets"

for file in os.listdir(dataset_folder):

    path = os.path.join(dataset_folder, file)

    if file.endswith(".txt"):

        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())

text = "\n".join(texts)

print("Loaded files:", len(texts))
print("Dataset size:", len(text))

tokens = sp.encode(text)

data = torch.tensor(tokens)


# ==========================================
# ФУНКЦИЯ СОЗДАНИЯ БАТЧА
# ==========================================
# используется packing (непрерывный поток)
# ==========================================

def get_batch():

    ix = torch.randint(
        len(data) - config.block_size,
        (config.batch_size,)
    )

    x = torch.stack([
        data[i:i+config.block_size] for i in ix
    ])

    y = torch.stack([
        data[i+1:i+config.block_size+1] for i in ix
    ])

    return x.to(device), y.to(device)


# ==========================================
# СОЗДАЁМ ПАПКУ ДЛЯ CHECKPOINT
# ==========================================

os.makedirs("checkpoints", exist_ok=True)


# ==========================================
# СОЗДАЁМ МОДЕЛЬ
# ==========================================

model = GPT(config).to(device)


# ==========================================
# УСКОРЕНИЕ: torch.compile
# ==========================================

if config.use_compile:
    model = torch.compile(model)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate
)

scaler = GradScaler()


# ==========================================
# ПРОВЕРЯЕМ ЕСТЬ ЛИ CHECKPOINT
# ==========================================

start_step = 0

checkpoint_path = "checkpoints/latest.pt"

if os.path.exists(checkpoint_path):

    print("Loading checkpoint...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    scaler.load_state_dict(checkpoint["scaler"])

    start_step = checkpoint["step"]

    print("Resuming from step:", start_step)


# ==========================================
# ФУНКЦИЯ СОХРАНЕНИЯ CHECKPOINT
# ==========================================

def save_checkpoint(step):

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step
    }

    torch.save(checkpoint, checkpoint_path)

    print("Checkpoint saved at step", step)


# ==========================================
# ЦИКЛ ОБУЧЕНИЯ
# ==========================================

for step in range(start_step, config.max_steps):

    xb, yb = get_batch()

    optimizer.zero_grad()

    with autocast():

        logits = model(xb)

        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            yb.view(-1)
        )

    scaler.scale(loss).backward()

    scaler.step(optimizer)

    scaler.update()


    if step % 200 == 0:
        print("step:", step, "loss:", loss.item())


    # ======================================
    # СОХРАНЯЕМ CHECKPOINT КАЖДЫЕ 1000 ШАГОВ
    # ======================================

    if step % 1000 == 0 and step > 0:
        save_checkpoint(step)


# ==========================================
# ФИНАЛЬНОЕ СОХРАНЕНИЕ МОДЕЛИ
# ==========================================

torch.save(model.state_dict(), "model.pt")

print("Training finished")