from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect

class CausalSelfAttention(nn.Module):
   def __init__(self, config):
      super().__init__()
      assert config.n_embd % config.num_heads == 0
      self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
      self.num_heads = config.num_heads
      self.dropout = config.dropout
      self.attn_dropout = nn.Dropout(config.dropout)
      self.resid_dropout = nn.Dropout(config.dropout)
      self.projection = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
      T = config.block_size
      self.flash = False
      self.flash = hasattr(F, 'scaled_dot_product_attention')
      if self.flash:
         print('Using flash attention.')
      if not self.flash:
        print("Not using flash attention because of old PyTorch version.")
        self.register_buffer('tril', torch.tril(torch.ones(T, T)).view(1, 1, T, T))

   def forward(self, x):
      B, T, C = x.size()
      nh = self.num_heads

      q, k, v = self.c_attn(x).split(C, dim=2) # (B, T, C) @ (C, 3C) = (B, T, 3C) => 3 of (B, T, C)
      q = q.view(B, T, nh, C // nh).transpose(1, 2) # (B, T, C) => (B, T, nh, hs) => (B, nh, T, hs)
      k = k.view(B, T, nh, C // nh).transpose(1, 2) # (B, T, C) => (B, T, nh, hs) => (B, nh, T, hs)
      v = v.view(B, T, nh, C // nh).transpose(1, 2) # (B, T, C) => (B, T, nh, hs) => (B, nh, T, hs)
      if self.flash:
         y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
      else:
        attention = q @ k.transpose(-1, -2) * (k.size(-1)**-0.5) # (B, nh, T, hs) @ (B, nh, hs, T) = (B, nh, T, T) => multiply by 1/sqrt(hs) (which is fan_in)
        attention = attention.masked_fill(self.tril==0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)
        y = attention @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
      y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

      # output projection
      y = self.resid_dropout(self.projection(y))
      return y

class Head(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.head_size = config.n_embd // config.num_heads
    self.key = nn.Linear(config.n_embd, self.head_size, bias=False)
    self.query = nn.Linear(config.n_embd, self.head_size, bias=False)
    self.value = nn.Linear(config.n_embd, self.head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)

    wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
    wei = wei.masked_fill(self.tril==0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)

    v = self.value(x)
    out = wei @ v
    return out
  
class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.heads = nn.ModuleList([Head(config) for _ in range(config.num_heads)])
    self.projection = nn.Linear(config.n_embd, config.n_embd)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.projection(out))
    return out
  
class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
        nn.GELU(),
        nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias), # projection layer
        nn.Dropout(config.dropout)
        )

  def forward(self, x):
    return self.net(x)
  
class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.layernorm1 = nn.LayerNorm(config.n_embd, bias=False)
    # self.attention = MultiHeadAttention(config)
    self.attention = CausalSelfAttention(config)
    self.layernorm2 = nn.LayerNorm(config.n_embd, bias=False)
    self.mlp = MLP(config)

  def forward(self, x):
    x = x + self.attention(self.layernorm1(x))
    x = x + self.mlp(self.layernorm2(x))
    return x
  
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_layers)])
        self.layernorm_final = nn.LayerNorm(config.n_embd, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.token_embedding_table.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
           if pn.endswith('projection.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # batch * block_size * num_embed
        pos_emb = self.position_embedding_table(torch.arange(self.config.block_size, device=device))
        x = tok_emb + pos_emb # (B, T, C) + (T, C)
        x = self.blocks(x)
        x = self.layernorm_final(x)

        if targets == None:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        else:
            logits = self.lm_head(x)
            logits = logits.view(self.config.batch_size * self.config.block_size, self.config.vocab_size)
            targets = targets.view(self.config.batch_size * self.config.block_size)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that are 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -self.config.block_size:]
            logits, loss = self(idx_cropped)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
    
@dataclass
class GPTConfig:
   batch_size: int = 32
   block_size: int = 1024
   vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
   num_layers: int = 12
   num_heads: int = 12
   n_embd: int = 768
   dropout: float = 0.0
   bias: bool = True