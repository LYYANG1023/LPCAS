from re import X
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy_of_replay(real_record_dict, question_id, q_worker_dict, replay_record, class_num):
    total_count = 0
    correct_count = 0
    for idx, q_id in enumerate(question_id):
        
        for worker_id in q_worker_dict[q_id]:
            total_count += 1
            if replay_record[idx][worker_id * class_num] == 1.:
                
                if real_record_dict[q_id][worker_id] == 0:
                    correct_count += 1
                else:
                    pass
            
            elif replay_record[idx][worker_id * class_num + 1] == 1.:
                if real_record_dict[q_id][worker_id] == 1:
                    correct_count += 1
                else:
                    pass
    return correct_count, total_count



def write_print_log(log_dir, log_content):
    print(log_content)
    with open(log_dir, "a+") as text_log:
        text_log.write(log_content + "\n")

def replay_acc_rate(previous_record, bin_gen_record, class_num, worker_num):

    batch_size = previous_record.shape[0]
    real = previous_record.view(batch_size, -1, class_num)
    gen = bin_gen_record.view(batch_size, -1, class_num)

    total_count = 0
    acc_count = 0
    for i in range(batch_size):
        for j in range(worker_num):
            
            if real[i][j][0] == 0. and real[i][j][1] == 0.:
                pass
            else:
                total_count += 1
                if real[i][j][0] == gen[i][j][0] and real[i][j][1] == gen[i][j][1]:
                    acc_count += 1

    return acc_count, total_count

def replay_acc_rate_2(previous_record, bin_gen_record, class_num):
    
    batch_size = previous_record.shape[0]
    real = previous_record.view(batch_size, -1, class_num)
    gen = bin_gen_record.view(batch_size, -1, class_num)

    acc_count = 0

    sum_res = torch.sum(real, dim=2)
    non_zero_zero_count =torch.count_nonzero(sum_res)

    sign_matrix = real == gen
    sign_matrix = torch.sum(sign_matrix, dim=2)
    sign_matrix = sign_matrix + sum_res
    sign_matrix = torch.where(sign_matrix == 3,1,0)
    acc_count = torch.count_nonzero(sign_matrix)
    return acc_count.item(), non_zero_zero_count.item()

def binary_record(class_num, input, mask):
    record = input.view(input.shape[0], -1, class_num)

    second_dim = torch.max(record, dim=2).indices
    fisrt_dim = torch.ones_like(second_dim) ^ second_dim
    record = torch.cat((fisrt_dim.unsqueeze(2), second_dim.unsqueeze(2)), 2)

    record = record.view(input.shape[0],-1)
    record = torch.mul(record, mask)

    return record


def mask_for_1stworker(zero_mask, q_id, w_id):
    for idx, q in enumerate(q_id):

        zero_mask[idx][w_id[idx] * 2] = 1.
        zero_mask[idx][w_id[idx] * 2 + 1] = 1.
    return zero_mask.requires_grad_(True)

def mask_update(old_mask, exist_idx, w_id):
    mask = old_mask.detach()
    if exist_idx == -1:
        mask_for_new = torch.zeros((1,mask.shape[1]), dtype=torch.float32).to(mask.device)
        mask_for_new[0][w_id * 2] = 1.
        mask_for_new[0][w_id * 2 + 1] = 1.
        mask = torch.cat((mask, mask_for_new), dim=0)
    else:
        mask[exist_idx][w_id * 2] = 1.
        mask[exist_idx][w_id * 2 + 1] = 1.
        

    return mask.requires_grad_(True)

def predicted_priori(concated_record):

    complete_record = torch.sum(concated_record, dim=0, keepdim=True)
    complete_record[complete_record>1.] = 1.
    pred_priori = torch.sum(complete_record.view(1,-1,2),dim=1)
    return pred_priori

def cumulative_real_priori(ques_seen_ids, class_num, priori_dict):

    cumu_real_priori = torch.zeros((len(ques_seen_ids), class_num), dtype=torch.float32)
    for idx, q_id in enumerate(ques_seen_ids):
        for i in range(class_num):
            cumu_real_priori[idx][i] = priori_dict[q_id][0][i]

    return cumu_real_priori

def init_weights(m):

    if type(m) == nn.Linear:
        torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-0.02, b=0.02)
        m.bias.data.fill_(0.)



def ELBO_loss(args, model, input, priori, y_hat, x_hat):

    if args.side_signal == True:
        input_side = input[:,0:args.latent_dimension]
        input_record = input[:,args.latent_dimension:]

        rec_side = x_hat[:,:,0:args.latent_dimension]
        rec_record = x_hat[:,:,args.latent_dimension:]  

        cross_entropy = []
        for i in range(args.class_num):
            log_p_v_y = torch.mul(input_side, torch.log(1e-10 + rec_side[i]))
            log_p_v_y = torch.mean(log_p_v_y, 1)
            qyv_logpvy = torch.mul(y_hat[:,i], log_p_v_y)
            cross_entropy.append(qyv_logpvy)

        cross_entropy = torch.stack(cross_entropy,dim=1)
        cross_entropy_sum = torch.mean(torch.sum(cross_entropy,1))


        expectation = []
        for i in range(args.class_num):
            log_p_v_y = torch.mul(input_record, torch.log(1e-10 + rec_record[i]))
            log_p_v_y = torch.mean(log_p_v_y, 1)
            qyv_logpvy = torch.mul(y_hat[:,i], log_p_v_y)
            expectation.append(qyv_logpvy)

        expectation = torch.stack(expectation,dim=1)
        expectation_sum = torch.mean(torch.sum(expectation,1))

        kl_div = F.kl_div(torch.log(y_hat), priori, reduction='batchmean')

        L1 = 0
        for para in model.named_parameters():
            if para[0] == 'ed.decoder.weight':
                L1 += torch.sum(torch.abs(para[1]))


        return expectation_sum, kl_div, L1, cross_entropy_sum

    else:

        expectation = []
        for i in range(args.class_num):
            lpg_p_v_y = torch.mul(input, torch.log(1e-10 + x_hat[i]))
            lpg_p_v_y = torch.mean(lpg_p_v_y, 1)
            qyv_logpvy = torch.mul(y_hat[:,i], lpg_p_v_y)
            expectation.append(qyv_logpvy)

        expectation = torch.stack(expectation,dim=1)
        expectation_sum = torch.mean(torch.sum(expectation,1))

        kl_div = F.kl_div(torch.log(y_hat), priori, reduction='batchmean')

        L1 = 0
        for para in model.named_parameters():
            if para[0] == 'ed.decoder.weight':
                L1 += torch.sum(torch.abs(para[1]))


        return expectation_sum, kl_div, L1, None



def Generative_loss(model, label_record, gen_record, real_priori):

    cross_entropy = torch.mul(label_record, torch.log(1e-10 + gen_record))
    cross_entropy = torch.mean(torch.mean(cross_entropy, 1), dim=0)

    pred_priori = torch.sum(gen_record.view(gen_record.shape[0],-1,2),dim=1)
    generator_kl_div = F.kl_div(torch.log(1e-10 + pred_priori), real_priori, reduction='batchmean')

    L1 = 0
    for para in model.named_parameters():
        if para[0] == 'gen.generator.weight':
            L1 += torch.sum(torch.abs(para[1]))

    return cross_entropy, generator_kl_div, L1

class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr

