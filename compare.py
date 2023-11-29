import torch

from lit_gpt.model import GPT

device = torch.device("cuda")
name = "pythia-70m"

with device:
    original_model = GPT.from_name(name)
    x = torch.randint(0, 10, (2, original_model.max_seq_length))

expected = original_model(x)
state_dict = original_model.state_dict()

with device:
    te_model = GPT.from_name(name)
    te_model.load_state_dict(state_dict)

actual = te_model(x)

torch.testing.assert_close(actual, expected)