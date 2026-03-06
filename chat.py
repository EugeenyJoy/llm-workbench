import torch
import torch.nn.functional as F

from model import GPT
from config import Config
from tokenizer import load_tokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

sp = load_tokenizer()

model = GPT(config).to(device)

model.load_state_dict(torch.load("model.pt",map_location=device))

model.eval()

print("chat started")

context = ""

while True:

    user = input("\nYou: ")

    context += user + "\nAI: "

    tokens = sp.encode(context)

    x = torch.tensor([tokens]).to(device)

    for _ in range(200):

        x_cond = x[:,-config.block_size:]

        logits = model(x_cond)

        logits = logits[:,-1,:]

        probs = F.softmax(logits,dim=-1)

        next_token = torch.multinomial(probs,1)

        x = torch.cat((x,next_token),dim=1)

    output = sp.decode(x[0].tolist())

    response = output.split("AI:")[-1]

    print("\nAI:",response)