import json
import os
from math import cos, pi

import torch
import torch.optim as optim
from tqdm import tqdm

from datasets.build_dataset import build_dataset
from models.build_model import build_model
from utils.configs import Parser
from utils.losses import build_loss


def checkpoint(model, epoch, save_path,arg_dict):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_out_path = f"{save_path}/{arg_dict['traindataset']}_model_iters_{epoch}.pth"
    torch.save({'state_dict': model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

class CosineRestartLr(object):
    def __init__(self,
                 base_lr,
                 periods,
                 restart_weights = [1],
                 min_lr = None,
                 min_lr_ratio = None):
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        self.base_lr = base_lr

    def annealing_cos(self, start: float,
                    end: float,
                    factor: float,
                    weight: float = 1.) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(f'Current iteration {iteration} exceeds '
                        f'cumulative_periods {cumulative_periods}')

    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups,
                                        lr_groups):
                param_group['lr'] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]  # iters

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault('initial_lr', group['lr'])
            self.base_lr = [group['initial_lr'] for group in optimizer.param_groups  # type: ignore
        ]

def train():
    argp = Parser()
    arg = argp.get_args()
    arg_dict = vars(arg)

    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    arg_dict['save_path_lingshi'] = os.path.join(arg_dict['save_path'], arg_dict['trainsave'])
    arg_dict['save_path'] = os.path.join(arg_dict['save_path'], arg_dict['traindataset'])
    arg_dict['dataroot'] = os.path.join(arg_dict['dataroot'], arg_dict['traindataset'])
    arg_dict['ann_file_train'] = os.path.join(arg_dict['ann_file_train'], f"{arg_dict['traindataset']}.csv")

    print("         **************************         ")
    print("===> save_path_lingshi:",arg_dict['save_path_lingshi'])
    print("===> save_path:",arg_dict['save_path'])
    print("===> dataroot:",arg_dict['dataroot'])
    print("===> ann_file_train:",arg_dict['ann_file_train'])
    print("         **************************         ")

    if not os.path.exists(arg_dict['save_path']):
        os.makedirs(arg_dict['save_path'])
    with open(os.path.join(arg_dict['save_path'], 'arg.json'), 'wt') as f:
        json.dump(arg_dict, f, indent=4)

    arg_dict['ann_file'] = arg_dict['ann_file_train']
    arg_dict['test_mode'] = False

    print('===> Loading target datasets')
    target_dataset = build_dataset(arg_dict)

    print('===> Loading pretrained model')
    model = build_model(arg_dict)

    model = model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Current device: {device}")

    for name, param in model.named_parameters():
        if 'uNet.encoder' in name:
            param.requires_grad = False

    if arg_dict['mode'] == 'pretrain':
        arg_dict['max_iters'] = 200000
    elif arg_dict['mode'] == 'transfer':
        arg_dict['max_iters'] = 200

    print("===> max_iters=",arg_dict['max_iters'])

    print("\n\n-------------------------Trainable parameters:-------------------------")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(name)    

    loss = build_loss(arg_dict)

    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_parameters, lr=arg_dict['lr'], betas=(0.9, 0.999), weight_decay=arg_dict['weight_decay'])

    cosine_lr = CosineRestartLr(arg_dict['lr'], [arg_dict['max_iters']], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    epoch_loss = 0
    iter_num = 0
    print_freq = 200
    save_freq = 200

    while iter_num < arg_dict['max_iters']:
        eepoch = iter_num+1
        print(f"********* epoch{eepoch}: *********")
        with tqdm(total=print_freq) as bar:
            for feature, label, _ in target_dataset:        
                if arg_dict['cpu']:
                    input, target = feature, label
                else:
                    input, target = feature.cuda(), label.cuda()

                regular_lr = cosine_lr.get_regular_lr(iter_num)
                cosine_lr._set_lr(optimizer, regular_lr)

                prediction = model(input)

                optimizer.zero_grad()
                pixel_loss = loss(prediction, target)

                epoch_loss += pixel_loss.item()
                pixel_loss.backward()
                optimizer.step()

                iter_num += 1
                
                bar.update(1)

                if iter_num % print_freq == 0:
                    break

        print("===> Iters[{}]({}/{}): Loss: {:.4f}".format(iter_num, iter_num, arg_dict['max_iters'], epoch_loss / print_freq))
        if iter_num % save_freq == 0:
            checkpoint(model, iter_num, arg_dict['save_path_lingshi'],arg_dict)
        print("\n")
        epoch_loss = 0


if __name__ == "__main__":
    train()