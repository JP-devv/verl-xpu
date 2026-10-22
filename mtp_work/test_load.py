import sys, torch
sys.path.insert(0, "/home/haxor/AI/verl/mtp_work")
from mtp_head import MTPHead
from transformers import AutoConfig
from safetensors import safe_open

MODEL = "/home/haxor/models/swift-27b"
cfg = AutoConfig.from_pretrained(MODEL)
text_config = cfg.text_config if hasattr(cfg, "text_config") else cfg
print("hidden", text_config.hidden_size, "vocab", text_config.vocab_size,
      "heads", text_config.num_attention_heads, "kv", text_config.num_key_value_heads,
      "head_dim", text_config.head_dim, "inter", text_config.intermediate_size)

# load mtp.* weights from shard 18
f = f"{MODEL}/model-00018-of-00018.safetensors"
mtp_state = {}
with safe_open(f, "pt") as s:
    for k in s.keys():
        if k.startswith("mtp."):
            mtp_state[k] = s.get_tensor(k)
print(f"loaded {len(mtp_state)} mtp tensors from disk")

# dummy embed/lm_head (real ones come from the main model in the harness)
embed = torch.nn.Embedding(text_config.vocab_size, text_config.hidden_size)
lm_head = torch.nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

head = MTPHead(text_config, embed, lm_head, device="cpu")
report = head.load_mtp_state(mtp_state)
print("loaded:", report["loaded"])
print("missing:", report["missing"])
print("unexpected:", report["unexpected"])

# sanity: a forward pass produces finite logits
h = torch.randn(4, text_config.hidden_size)
tok = torch.tensor([1, 2, 3, 4])
with torch.no_grad():
    logits = head(h, tok)
print("logits shape:", tuple(logits.shape), "finite:", torch.isfinite(logits).all().item())
