# # %% 
# %load_ext autoreload
# %autoreload 2
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils_study_res import get_model
from model_run import basic_pipeline, fix_seed
from params import params
import torchvision.transforms as T

def main():
    fix_seed(params)
    basic_pipeline(params, experiment_name=exp_name)

if __name__ == "__main__":
    params['runs_id'] = [
        'f26ea35d05f9452fb1061e2a89f85d1c',  # x
        '1f7b43573e0a4556945381b2e4933c07',  # y
        '0b49acce2a024bb09056b292126ad4d5',  # z
    ]
    params_list = [[get_model(run_id)[0] for run_id in params['runs_id']]]
    params['params_multi'] = params_list[0]
    params['transform_multi'] = T.Compose([
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    params['clf_neurons'] = (128, 2)
    params['optim_lr'] = 0.001
    exp_name = "Concat Eff"
    main()

# %%
# params_list = [[get_model(run_id)[0] for run_id in params['runs_id']]]

# params['params_multi'] = params_list[0]
# params['transform_multi'] = T.Compose([
#     T.Lambda(lambda x: x.repeat(3, 1, 1)),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# params['clf_neurons'] = (128, 2)
# params['optim_lr'] = 0.001
# exp_name = "Concat Eff"
# # %%
# %load_ext autoreload
# %autoreload 2
# import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from utils_study_res import get_model
# from model_run import basic_pipeline, fix_seed
# from params import params
# import torchvision.transforms as T
# from concat_eff.params import params
# import torchvision.transforms as T
# from torchsummary import summary
# from _models import ConcatEffModels
# import torch
# from paths import cuda_device
# params['runs_id'] = [
#     'f26ea35d05f9452fb1061e2a89f85d1c',  # x
#     '1f7b43573e0a4556945381b2e4933c07',  # y
#     '0b49acce2a024bb09056b292126ad4d5',  # z
# ]
# model = ConcatEffModels(params)
# device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
# model.to(device)
# summary(model, [(3, 224,  224), (3, 224,  224), (3, 224,  224)])
# # %%
# x = torch.rand([1, 3, 224, 244]).to('cuda')
# y = torch.rand([1, 3, 224, 244]).to('cuda')
# z = torch.rand([1, 3, 224, 244]).to('cuda')
# model(x, y, z)
# # %%
# from model_run import get_data_validation
# train_data, val_data = get_data_validation(params)