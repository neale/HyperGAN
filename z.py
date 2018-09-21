# coding: utf-8
import foolbox 
import logging
from foolbox.criteria import Misclassification
foolbox_logger = logging.getLogger('foolbox')
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import torch
import matplotlib.pyplot as plt
import adversarial_test as adv
import adversarial_test as adv
args = adv.load_args()
arch = adv.get_network(args)
get_ipython().run_line_magic('clear', '')
import utils; hypernet = utils.load_hypernet('hypermnist_0_0.984390625.pt')
import netdef; args.stat = netdef.nets()[args.net]
model_base, fmodel_base = adv.sample_fmodel(args, hypernet, arch)
criterion = Misclassification()
fgs = foolbox.attacks.BIM(fmodel_base, criterion)
import datagen
_, test = datagen.load_mnist(args)
batch = []
y = []
for (data, target) in test:
    batch.append(data)
    y.append(target)
    if len(batch) == 32:
        break
    
stack_data = torch.stack(batch)
stack_y = torch.stack(y)
adv_batch, target_batch, _ = adv.sample_adv_batch(stack_data[0], stack_y[0], fmodel_base, 0.03, fgs)
import torch.nn.functional as F
soft_out = []
pred_out = []
logits = []
paths = ['mnist_model_small2_0.pt', 'mnist_model_small2_1.pt', 'mnist_model_small2_2.pt', 'mnist_model_small2_3.pt', 'mnist_model_small2_4.pt', 'mnist_model_small2_5.pt', 'mnist_model_small2_6.pt', 'mnist_model_small2_7.pt', 'mnist_model_small2_8.pt', 'mnist_model_small2_9.pt']
models = []
for path in paths:
    model = adv.get_network(args)
    model.load_state_dict(torch.load(path))
    models.append(model.eval())
    
model = adv.FusedNet(models)
import attacks
fmodel = attacks.load_model(model)
fgs = foolbox.attacks.BIM(fmodel, criterion)
adv_batch, target_batch, _ = adv.sample_adv_batch(stack_data[0], stack_y[0], fmodel, 0.08, fgs)
soft_out = []
pred_out = []
logits = []
for i in range(5):
    output = models[i](adv_batch)
    soft_out.append(F.softmax(output, dim=1))
    pred_out.append(output.data.max(1, keepdim=True)[1])
    logits.append(output)
    

