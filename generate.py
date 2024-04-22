import os
import torch
import pickle

from model import GPT, GPTConfig

init_from = 'resume'
out_dir = 'out'
device = 'cpu'
start = "\n"
max_new_tokens = 500


checkpoint_path = os.path.join(out_dir, 'checkpoint.pt')
checkpoint = torch.load(checkpoint_path, map_location=device)
gptconfig = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconfig)
stoi = checkpoint['tokenizer']['encoding']
itos = checkpoint['tokenizer']['decoding']
state_dict = checkpoint['model']
checkpoint = None
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

start_tokens = encode(start * model.config.block_size)
x = torch.tensor(start_tokens, dtype=torch.long, device=device)[None, ...]

with torch.no_grad():
    y = model.generate(x, max_new_tokens)
    print()
    print(decode(y[0][model.config.block_size:].tolist()))