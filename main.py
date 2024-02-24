import os
import torch
import argparse
from torch.backends import cudnn

from train import _train
from test import _test
from SideAggregator import build_net
from utils import init_weights


def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.log_save_dir):
        os.makedirs(args.log_save_dir)

    model = build_net(args.model_name, args.batch_size, args.ques_num, args.worker_num, args.class_num, args.latent_dimension, args.side_signal)
    model.apply(init_weights)
    # print(model)
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _test(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='LAA', choices=['LAA'], type=str)
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--data_path', type=str, default='D:\LocalProject\datasets')
    parser.add_argument('--dataset_name', type=str, default='RTE')
    parser.add_argument('--file_name', type=str, default='answer.npz')
    parser.add_argument('--ques_num', type=int, default=800)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--worker_num', type=int, default=164)

    # side information
    parser.add_argument('--side_signal', type=bool, default=True)
    parser.add_argument('--latent_dimension', type=int, default=64)
    parser.add_argument('--replay_test_num', type=int, default=10)
    parser.add_argument('--replay_threshold', type=float, default=1)


    # Train
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=64)
    parser.add_argument('--num_worker', type=int, default=0)

    # decoder optimizer
    parser.add_argument('--decoder_learning_rate', type=float, default=0.01)
    parser.add_argument('--decoder_weight_decay', type=float, default=0)
    # decoder scheduler
    parser.add_argument('--decoder_gamma', type=float, default=0.5)
    parser.add_argument('--decoder_lr_steps', type=list, default=[20,40,60,80])
    # decoder hyper para
    parser.add_argument('--inner_decoder_iter_num', type=int, default=3000)
    parser.add_argument('--decoder_lambda_kl', type=float, default=0.0001)
    parser.add_argument('--decoder_lambda_l1', type=float, default=0.005)


    # optimizer
    parser.add_argument('--laa_learning_rate', type=float, default=0.0005)
    parser.add_argument('--laa_weight_decay', type=float, default=0)
    # scheduler
    parser.add_argument('--laa_gamma', type=float, default=0.5)
    parser.add_argument('--laa_lr_steps', type=list, default=[100,200,300,400])
    # hyper para
    parser.add_argument('--inner_lla_iter_num', type=int, default=100)
    parser.add_argument('--laa_lambda_kl', type=float, default=0.0001)
    parser.add_argument('--laa_lambda_l1', type=float, default=0.005)
    parser.add_argument('--laa_lambda_side', type=float, default=0.005)



    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--valid_freq', type=int, default=1)
    parser.add_argument('--resume', type=str, default='')

    args = parser.parse_args()
    args.log_save_dir = os.path.join('results/', args.dataset_name + '_' + str(args.batch_size) , 'logs/')
    args.log_text_dir = os.path.join('results/', args.dataset_name + '_' + str(args.batch_size), 'logs/', 'log.txt')
    args.model_save_dir = os.path.join('results/', args.dataset_name + '_' + str(args.batch_size), 'weights/')
    print(args)
    main(args)
