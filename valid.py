import torch

import os
from utils import Adder, Timer, check_lr, ELBO_loss, Generative_loss
# from torch.utils.tensorboard import SummaryWriter
from data_load import valid_dataloader
import torch.nn.functional as F


def _valid(model, args, ep):
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = valid_dataloader(args.data_path, args.file_name, args.class_num, batch_size=batch_size, num_workers=0)

    model.eval()
    loss_value = Adder()

    with torch.no_grad():
        print('Start Evaluation')
        for idx, data in enumerate(dataloader):
            record, mask, priori = data
            record = record.to(device)
            mask = mask.to(device)
            priori = priori.to(device)

            y_hat, x_hat = model(record)

            expectation_mean, kl_div, L1 = loss(args, model, record, mask, priori, y_hat, x_hat)
            loss_content = -1 * (
                        expectation_mean - args.lambda_kl * kl_div) + args.lambda_l1 / args.batch_size / args.class_num / args.class_num * L1

            loss_value(loss_content)
            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()
    return loss_value.average()
