import os
import numpy as np
import torch
import pickle
import time
from model import GPT, GPTConfig
import math
from contextlib import nullcontext


dataset = 'he_an'
device = 'cuda'
init_from = 'scratch'
out_dir = 'out'

eval_interval = 10000
eval_iters = 200
logging_interval = 1000

num_layers = 4
num_heads = 6
n_embd = 96
bias = False
dropout = 0.2
block_size = 32
batch_size = 12
# gradient accumulation
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# AMP
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

os.makedirs(out_dir, exist_ok=True)

data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1 : i+block_size+1].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

iter_num = 0
best_val_loss = 1e9

meta_data_path = os.path.join(data_dir, 'meta.pkl')
vocab_size = None
if os.path.exists(meta_data_path):
    with open(meta_data_path, 'rb') as f:
        meta_data = pickle.load(f)
    # vocab_size = 3136 # pad up to nearest multiple of 64
    vocab_size = meta_data['vocab_size']
    print(f"found vocab_size = {vocab_size} (inside {meta_data_path})")

model_args = dict(batch_size=batch_size, num_layers=num_layers, num_heads=num_heads, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = vocab_size if vocab_size is not None else 50304
    gptconfig = GPTConfig(**model_args)
    model = GPT(gptconfig)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    checkpoint_path = os.path.join(out_dir, 'checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['batch_size', 'num_layers', 'num_heads', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconfig = GPTConfig(**model_args)
    model = GPT(gptconfig)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

X, Y = get_batch('train')
logs = []
t0 = time.time()
while True:
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'tokenizer': {
                        'encoding': meta_data['stoi'],
                        'decoding': meta_data['itos']
                        }
                }
                torch.save(checkpoint, os.path.join(out_dir, 'checkpoint.pt'))
                print(f"saved checkpoint to {out_dir}")
    for accumulation_step in range(gradient_accumulation_steps):
        # with ctx:
        #     logits, loss = model(X, Y)
        #     loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # X, Y = get_batch('train')
        # # backward pass, with gradient scaling if training in fp16
        # scaler.scale(loss).backward()
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    logs.append(dt*1000)
    if iter_num % logging_interval == 0:
        print(f'Average time for {logging_interval} batches: {torch.tensor(logs).mean():.2f}ms')
        # print(f"iter {iter_num}: time {dt*1000:.2f}ms")

    iter_num += 1
    if iter_num > max_iters:
        break