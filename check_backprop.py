import torch
import torch.nn.functional as F
print('START: imports done')
from SATVN import SATVN
print('IMPORT SATVN OK')

# Small model/data for a quick backprop check
device = torch.device('cpu')
batch = 8
in_seq_length = 4
out_seq_length = 2
input_dim = 1
output_dim = 1

d_model = 4
hidden_dim = 8
h = 2
M = 1

model = SATVN(input_dim, d_model, hidden_dim, h, M, output_dim, in_seq_length, out_seq_length, device).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# random input/target
input = torch.randn(batch, in_seq_length, input_dim, device=device)
target = torch.randn(out_seq_length, batch, output_dim, device=device)

# save a copy of parameters
params_before = [p.detach().clone() for p in model.parameters()]

outputs = model(input, target, is_training=True)
loss = F.mse_loss(outputs, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# check whether any parameter changed
changed = False
for p_before, p_after in zip(params_before, model.parameters()):
    if not torch.allclose(p_before, p_after):
        changed = True
        break

print('Loss:', loss.item())
print('Parameters changed:', changed)
