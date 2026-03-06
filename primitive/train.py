import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== ЧИТАЕМ ДАННЫЕ =====

with open("data.txt", "r") as f:
    text = f.read()

# список уникальных символов
chars = sorted(list(set(text)))

vocab_size = len(chars)

# словарь: символ -> число
stoi = {ch:i for i,ch in enumerate(chars)}

# обратный словарь: число -> символ
itos = {i:ch for i,ch in enumerate(chars)}

# функция кодирования текста в числа
def encode(s):
    return [stoi[c] for c in s]

# функция декодирования чисел в текст
def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# ===== ПАРАМЕТРЫ МОДЕЛИ =====

block_size = 8      # сколько символов модель видит за раз
batch_size = 16
embed_size = 32     # размер вектора символа

# ===== СОЗДАНИЕ БАТЧА ДАННЫХ =====

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x, y

# ===== МОДЕЛЬ =====

class MiniGPT(nn.Module):

    def __init__(self):
        super().__init__()

        # превращает символы в векторы
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # слой внимания
        self.attention = nn.MultiheadAttention(embed_size, num_heads=2, batch_first=True)

        # линейный слой для предсказания следующего символа
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):

        x = self.embedding(x)

        attn_output, _ = self.attention(x, x, x)

        logits = self.fc(attn_output)

        return logits


model = MiniGPT()

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# ===== ОБУЧЕНИЕ =====

for step in range(2000):

    xb, yb = get_batch()

    logits = model(xb)

    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        yb.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print("step:", step, "loss:", loss.item())

# ===== ГЕНЕРАЦИЯ ТЕКСТА =====

context = torch.zeros((1,1), dtype=torch.long)

generated = []

for _ in range(100):

    logits = model(context)

    logits = logits[:,-1,:]

    probs = F.softmax(logits, dim=-1)

    next_token = torch.multinomial(probs, num_samples=1)

    generated.append(next_token.item())

    context = torch.cat([context, next_token], dim=1)

print("\nGenerated text:\n")
print(decode(generated))