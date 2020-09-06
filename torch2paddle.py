import torch
import paddle.fluid as fluid
from collections import OrderedDict
from models import TSN

place = fluid.CPUPlace()
with fluid.dygraph.guard(place):
    model = TSN(101, 16, 'finetune', 'RGB', 'ECOfull')
    model_dict = model.state_dict()

load_path = ''
save_path = ''
torch_weight = torch.load(load_path)
torch_dict = torch_weight['state_dict']
paddle_dict = OrderedDict()

for k,v in torch_dict.items():
    k = k[7:]  # remove 'module.'
    if 'tracked' in k or 'fc' in k:
        continue
    if 'running_mean' in k:
        k = k.replace('running_mean', '_mean')
    if 'running_var' in k:
        k = k.replace('running_var', '_variance')
    paddle_dict[k] = v.detach().numpy()

keys1 = set(list(paddle_dict.keys()))
keys2 = set(list(model_dict.keys()))

print('extra keys: {}'.format(keys1 - keys2))
print('missing keys: {}'.format(keys2 - keys1))

if len(keys1 - keys2) == 0:
    model_dict.update(paddle_dict)
    model.set_dict(model_dict)
    fluid.dygraph.save_dygraph(model.state_dict(), save_path)
else:
    print('Error! existing extra keys!')
