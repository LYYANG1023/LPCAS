import csv
import os
import numpy as np
import torch
from torch.optim import optimizer
import torch.nn.functional as F

from SideAggregator import build_net
from data_load import test_dataloader
from utils import mask_for_1stworker


def getTruth(file_path):
    e2truth = {}

    with open(os.path.join(file_path, 'transformed.truth.csv'), encoding='UTF-8-sig') as f_truth:

        reader = csv.reader(f_truth)

        header = next(reader)

        for line in reader:

            example, truth = line

            e2truth[int(example)] = int(truth)

    return e2truth


def _test(args):

    truth = getTruth(args.data_path)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_path, args.file_name, args.class_num, batch_size=1, num_workers=0)

    state = torch.load('./weights/Best.pkl')
    model = build_net('LPA', batch_size=1, input_size=args.input_size, class_num=args.class_num)
    try:
        model.load_state_dict(state['model'])
    except Exception as e:
        print(e)

    print('Resumed from Best.pkl')
    correct_count = 0
    total_count = len(truth)
    model.eval()
    with torch.no_grad():
        print('Start Evaluation')
        for idx, data in enumerate(dataloader):
            answer, mask, priori = data
            answer = answer.to(device)

            y_hat, v_hat = model(answer)
            y_pred = torch.argmax(y_hat)
            print(y_pred)
            if y_pred == truth[idx]:
                correct_count += 1


    print('Acc Rate: {}         Err Rate: {}'.format(correct_count/total_count, 1 - correct_count/total_count))
    model.train()

def _test_immediately(args, model, iter_idx, gen_answer, ques_id_seen):

    truth = getTruth(args.data_path)
    pred_label = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    correct_count = 0
    model.eval()
    false_answer = []


    with torch.no_grad():
        dataset_size = gen_answer.shape[0]
        for i in range(dataset_size):

            real_question_id = ques_id_seen[i]
            real_answer = gen_answer[i].unsqueeze(dim=0)
            real_answer = real_answer.to(device)

            y_hat, v_hat = model(None, None, real_answer, 'LAA')
            y_pred = torch.argmax(y_hat)
            pred_label[int(real_question_id)] = y_pred

            
        for q_id, p_label in pred_label.items():
            if p_label == truth[int(q_id)]:
                correct_count += 1
            else:
                false_answer.append(str(int(q_id)))

    model.train()
    return false_answer, correct_count