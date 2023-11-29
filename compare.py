import torch
from lit_gpt.model import GPT
from model_te import GPT as TEGPT

device = torch.device("cuda")
name = "pythia-70m"

with device:
    original_model = GPT.from_name(name)
    original_model.apply(original_model._init_weights)
    x = torch.randint(0, 10, (2, original_model.max_seq_length))
expected = original_model(x)
state_dict = original_model.state_dict()

with device:
    te_model = TEGPT.from_name(name)
keys = te_model.load_state_dict(state_dict, strict=False)
assert not keys.unexpected_keys
actual = te_model(x)

torch.testing.assert_close(actual, expected)