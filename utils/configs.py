import argparse
import os
import sys

sys.path.append(os.getcwd())

def str2bool(str):
    return True if str.lower() == 'true' else False

class Parser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--save_path', type=str)
        self.parser.add_argument('--pretrained', type=str)
        self.parser.add_argument('--max_iters', default=200)
        self.parser.add_argument('--plot_roc', action='store_true')
        self.parser.add_argument('--arg_file', default=None)
        self.parser.add_argument('--cpu', default=False)
        self.parser.add_argument('--dataroot', type=str)
        self.parser.add_argument('--ann_file_train', type=str)
        self.parser.add_argument('--ann_file_test', type=str)
        self.parser.add_argument('--dataset_type', default='CongestionDataset')
        self.parser.add_argument('--batch_size', default=15)
        self.parser.add_argument('--aug_pipeline', default=['Flip'])
        self.parser.add_argument('--model_type', default='Congestion_Prediction_Net')
        self.parser.add_argument('--in_channels', default=3)
        self.parser.add_argument('--out_channels', default=1)
        self.parser.add_argument('--lr', default=2e-4)
        self.parser.add_argument('--weight_decay', default=0)
        self.parser.add_argument('--loss_type', default='MSELoss')
        self.parser.add_argument('--eval-metric', default=['NRMS', 'SSIM'])
        self.parser.add_argument('--traindataset', default=None)
        self.parser.add_argument('--trainsave', default=None)
        self.parser.add_argument('--testdataset', default=None)
        self.parser.add_argument('--mode', type=str, choices=['pretrain', 'transfer'], default='pretrain')

    def get_args(self):
        return self.parser.parse_args()

