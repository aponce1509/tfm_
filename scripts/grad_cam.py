# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
from skimage.transform import resize

class GradCamModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        
        #PRETRAINED MODEL
        self.pretrained = model
        self.layerhook.append(self.pretrained.features.register_forward_hook(self.forward_hook()))
        
        for p in self.pretrained.parameters():
            p.requires_grad = True
    
    def activations_hook(self,grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self,x):
        out = self.pretrained(x)
        return out, self.selected_out

def get_cam(img, label, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gcmodel = GradCamModel(model).to(device)
    img = img.reshape(1, 1, 128, 62)
    img = torch.Tensor(img).to(device)
    out, acts = gcmodel(img)
    acts = acts.detach().cpu()
    criterion = nn.NLLLoss()
    loss = criterion(out, torch.LongTensor([label]).to(device).reshape(1))
    loss.backward()
    grads = gcmodel.get_act_grads().detach().cpu()
    pooled_grads = torch.mean(grads, dim=[0, 2, 3]).detach().cpu()
    for i in range(acts.shape[1]):
        acts[:, i, :, :] += pooled_grads[i]
    heatmap_j = torch.mean(acts, dim = 1).squeeze()
    heatmap_j_max = heatmap_j.max(axis = 0)[0]
    heatmap_j /= heatmap_j_max
    heatmap_j = heatmap_j.to("cpu").numpy()
    heatmap_j = resize(heatmap_j, (128, 62), preserve_range=True)
    cmap = mpl.cm.get_cmap('jet', 256)
    heatmap_j2 = cmap(heatmap_j, alpha=0.2)
    return heatmap_j2
# %%
if __name__ == "__main__":
    run_id = "ea7ab94303af439fa9c8a5d364b84a5a"
    from utils_study_res import get_model, get_data
    params, model = get_model(run_id)
    train_loader, val_loader, _, test_loader = get_data(params)
    batch_img, labels, energy = iter(train_loader).next()


    # %%
    n = 10
    print(labels[n])
    gcmodel = GradCamModel(model).to("cuda")
    img = batch_img[n].reshape(1, 1, 128, 62).type(torch.FloatTensor).cpu()
    out, acts = gcmodel(img.to("cuda"))

    acts = acts.detach().cpu()
    criterion = nn.NLLLoss()
    loss = criterion(out, labels[n].to('cuda').reshape(1))
    loss.backward()
    grads = gcmodel.get_act_grads().detach().cpu()
    pooled_grads = torch.mean(grads, dim=[0,2,3]).detach().cpu()
    for i in range(acts.shape[1]):
        acts[:,i,:,:] += pooled_grads[i]
    heatmap_j = torch.mean(acts, dim = 1).squeeze()
    heatmap_j_max = heatmap_j.max(axis = 0)[0]
    heatmap_j /= heatmap_j_max
    heatmap_j = heatmap_j.to("cpu").numpy()
    heatmap_j = resize(heatmap_j, (128, 62), preserve_range=True)
    cmap = mpl.cm.get_cmap('jet', 256)
    heatmap_j2 = cmap(heatmap_j, alpha=0.2)

    img_numpy = img.to("cpu").numpy()[0, 0, :, :]
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.imshow(img_numpy)
    axs.imshow(heatmap_j2)
    plt.show()
    # %%

    run_id = "ea7ab94303af439fa9c8a5d364b84a5a"
    from utils_study_res import get_model, get_data
    params, model = get_model(run_id)
    # %%
    train_loader, val_loader, _, test_loader = get_data(params)
    batch_img, labels, energy = iter(train_loader).next()
    img = batch_img[0].reshape(1, 1, 128, 62).type(torch.FloatTensor).cpu()
    img_numpy = img.to("cpu").numpy()[0, 0, :, :].reshape(128, 62)
    import matplotlib.pyplot as plt
    from torchsummary import summary
    img_numpy = img.to("cpu").numpy()[0, 0, :, :]
    plt.imshow(img_numpy, "gray")
    # %%
