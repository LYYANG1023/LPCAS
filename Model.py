from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

def paired_softmax(batch_size, class_num, input, binary=False):
    v_hat = input.view(batch_size, -1, class_num)
    v_hat = F.softmax(v_hat, dim=2)

    if binary == True:
        second_dim = torch.max(v_hat, dim=2).indices
        fisrt_dim = torch.ones_like(second_dim) ^ second_dim
        v_hat = torch.cat((fisrt_dim.unsqueeze(2), second_dim.unsqueeze(2)), 2)

    v_hat = v_hat.view(batch_size,-1)

    return v_hat

def compact_answer(concated_answer):

    complete_answer = torch.sum(concated_answer, dim=0, keepdim=True)
    complete_answer[complete_answer>1.] = 1.
    a = torch.sum(complete_answer.view(1,-1,2),dim=1)
    return complete_answer

class decoder(nn.Module):
    def __init__(self, record_num, user_num, class_num, latent_dimension):
        super(decoder, self).__init__()
        self.record_num = record_num
        self.user_num = user_num
        self.class_num = class_num
        self.latent_dimension = latent_dimension
        self.encoder = nn.Linear(record_num, latent_dimension)
        self.decoder = nn.Linear(latent_dimension, user_num * class_num)


    def forward(self, record_embed):
        side_info = self.encoder(record_embed)
        side_info = F.sigmoid(side_info)
        replay_num = record_embed.shape[0]
        gen_answer = self.decoder(side_info)
        gen_answer = paired_softmax(replay_num, self.class_num, gen_answer, binary=False)
        return side_info, gen_answer


class Sideencoder(nn.Module):

    def __init__(self, batch_size, user_num, class_num, latent_dimension, side_signal):
        super(Sideencoder, self).__init__()
        self.batch_size = batch_size
        self.input_size = user_num * class_num
        self.class_num = class_num
        self.latent_dimension = latent_dimension
        self.side_signal = side_signal
        if side_signal == True:
            self.encoder = nn.Linear(latent_dimension + user_num * class_num, class_num)
            self.decoder = nn.Linear(class_num, latent_dimension + user_num * class_num)
        else:
            self.encoder = nn.Linear(user_num * class_num, class_num)
            self.decoder = nn.Linear(class_num, user_num * class_num)


    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        y_hat = F.softmax(self.encoder(x),dim=1)


        y = torch.zeros((self.class_num, x.shape[0], self.class_num), dtype=torch.float32).to(device)
        for i in range(self.class_num):
            y[i,:,i] = 1.
        y.requires_grad_()

        x_hat = []
        for i in range(self.class_num):
            x_i = self.decoder(y[i])

            if self.side_signal == True:
                rec_side = F.sigmoid(x_i[:,0:self.latent_dimension])
                rec_answ = x_i[:,self.latent_dimension:]  
            else:
                rec_answ = x_i

            x_i = paired_softmax(x.shape[0], self.class_num, rec_answ)

            if self.side_signal == True:
                x_i = torch.cat((rec_side, x_i), dim=1)  

            x_hat.append(x_i)

        x_hat = torch.stack(x_hat,dim=0)
        return y_hat, x_hat # y_hat [batch_size, class_num], x_hat [calss_num, batch_size, input_size]



class Extractor(nn.Module):
    def __init__(self, batch_size, record_num, user_num, class_num, latent_dimension, side_signal):
        super(Incremental_TI, self).__init__()
        self.batch_size = batch_size
        self.record_num = record_num
        self.user_num = user_num
        self.class_num = class_num
        self.side_signal = side_signal
        self.latent_dimension = latent_dimension
        self.gen = decoder(record_num, user_num, class_num, latent_dimension)
        self.ed = Sideencoder(batch_size, user_num, class_num, latent_dimension, side_signal)


    def forward(self, recordtion_embeds, mask, best_gen_answer, mode):

        if mode == 'decoder':
            side_info, gen_answer = self.gen(recordtion_embeds)
            gen_answer = torch.mul(gen_answer, mask)
            return side_info, gen_answer

        if mode == 'encoder':
            rec_y_hat, rec_x_hat = self.ed(best_gen_answer)
            return rec_y_hat, rec_x_hat



def build_net(model_name, batch_size, record_num, user_num, class_num, latent_dimension, side_signal):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "Extractor":
        return Incremental_TI(batch_size, record_num, user_num, class_num, latent_dimension, side_signal)
    raise ModelError('Wrong Model!\nYou should choose Extractor.')

