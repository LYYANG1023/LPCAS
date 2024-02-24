import os
from collections import defaultdict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from data_load import train_dataloader
from test import _test_immediately
from utils import *
from valid import _valid
from record2Vector import read_answer

def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    dataloader = train_dataloader(args.data_path, args.file_name, args.class_num, args.batch_size, args.num_user)
    max_iter = len(dataloader)


    writer = SummaryWriter(args.log_save_dir)
    early_stop_count = 0
    replay_timer = Timer('s')
    decoder_training_timer = Timer('s')

    record_seen = list()

    for iter_idx, batch_data in enumerate(dataloader):

        real_record_id, real_user_id, real_answer, real_mask, real_priori_dict = batch_data
        real_answer = real_answer.to(device).requires_grad_() # [batch_size, class_num * user_num]
        real_mask = real_mask.to(device).requires_grad_()
        

        real_record_id_list = real_record_id.tolist()
        real_user_id_list = real_user_id.tolist()

        # decoder
        if iter_idx == 0:
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.decoder_learning_rate,
                                         weight_decay=args.decoder_weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.decoder_lr_steps, args.decoder_gamma)


            for idx, q_id in enumerate(real_record_id_list):

                if q_id in record_seen:
                    index = record_seen.index(q_id)
                    gen_answer_seen[index] = gen_answer_seen[index] + real_answer[idx]
                    mask = mask_update(mask, index, real_user_id_list[idx])
     
                else:
                    record_seen.append(q_id)
                    real_record_embeds = F.one_hot(torch.tensor(q_id).unsqueeze(dim=0), num_classes=args.ques_num).to(dtype=torch.float32).to(device)
                    if iter_idx == 0 and idx == 0:
                        record_seen_embeds = real_record_embeds
                        gen_answer_seen = real_answer[idx].unsqueeze(0)
                        mask = mask_for_1stuser(torch.zeros_like(real_answer[idx].unsqueeze(0)), [q_id], [real_user_id_list[idx]]).to(device)
                    else:
                        record_seen_embeds = torch.cat((record_seen_embeds, real_record_embeds), dim=0)
                        gen_answer_seen = torch.cat((gen_answer_seen, real_answer[idx].unsqueeze(0)),dim=0)
                        mask = mask_update(mask, -1, real_user_id_list[idx])
                
            
            real_priori = cumulative_real_priori(record_seen, args.class_num, real_priori_dict).to(device).requires_grad_()
            previous_answer = gen_answer_seen.detach().to(device).requires_grad_(True)
            gen_answer_seen = gen_answer_seen.detach().to(device).requires_grad_(True)
            record_seen_embeds = record_seen_embeds.detach().to(device).requires_grad_(True)
            mask = mask.detach().to(device).requires_grad_(True)
            for i in range(args.inner_decoder_iter_num):
                
                if i % args.replay_test_num == 0:
                    model.eval()
                    with torch.no_grad():
                        side_info, gen_answer = model(record_seen_embeds, mask, None, 'decoder')
                        bin_gen_answer = binary_answer(args.class_num, gen_answer, mask)
                        acc_count, total_count = replay_acc_rate_2(previous_answer, bin_gen_answer, args.class_num)
                    model.train()
 
                    # write_print_log(args.log_text_dir, "Instantaneous Replay: Iter index : {}    Inner_decoder : {}          Correct Count : {}    Total Count : {}     ".format(iter_idx, i, acc_count, total_count))
                    writer.add_scalar('Instantaneous Replay Acc rate', acc_count / total_count, iter_idx * args.inner_decoder_iter_num + i)
                    if acc_count / total_count >= args.replay_threshold and i != 0:

                        write_print_log(args.log_text_dir, "****decoder inner-training early stopping****\tIter index : {}\tInner_decoder : {}\tCorrect Count : {}\tTotal Count : {}".format(iter_idx, i, acc_count, total_count))
                    
                        early_stop_count += 1
                        write_print_log(args.log_text_dir, "Iter index : {}    Inner_decoder : {}     Early stopping".format(iter_idx, i))
                        writer.add_scalar('Early Stop Rate', early_stop_count / (iter_idx + 1), iter_idx + 1)
                        break

                
                _, gen_answer = model(record_seen_embeds, mask, None, 'decoder')
                cross_entropy, decoder_kl_div, decoder_L1 = Generative_loss(model, gen_answer_seen, gen_answer, real_priori)
                loss_decoder = -1 * cross_entropy # + args.decoder_lambda_l1 / args.ques_num / args.user_num / args.class_num * decoder_L1 # args.decoder_lambda_kl *10000* decoder_kl_div
                optimizer.zero_grad()
                loss_decoder.backward()
                optimizer.step()
                scheduler.step()
                
                writer.add_scalar('decoder inner-training Cross Entropy', -1 * cross_entropy, iter_idx * args.inner_decoder_iter_num + i)
                writer.add_scalar('decoder inner-training Loss', loss_decoder, iter_idx * args.inner_decoder_iter_num + i)


                if i == (args.inner_decoder_iter_num - 1):
                    model.eval()
                    with torch.no_grad():
                        side_info, gen_answer = model(record_seen_embeds, mask, None, 'decoder')
                        bin_gen_answer = binary_answer(args.class_num, gen_answer, mask)
                        acc_count, total_count = replay_acc_rate_2(previous_answer, bin_gen_answer, args.class_num)    
                    model.train()
                    
                    # write_print_log(args.log_text_dir, "Instantaneous Replay: Iter index : {}    Inner_decoder : {}          Correct Count : {}    Total Count : {}     ".format(iter_idx, i, acc_count, total_count))
                    writer.add_scalar('Instantaneous Replay Acc rate', acc_count / total_count, iter_idx * args.inner_decoder_iter_num + i)

            write_print_log(args.log_text_dir, "****decoder Iteration Res ****  Iter index : {}         Cross Entropy : {}     Loss:{}".format(iter_idx, -1 * cross_entropy, loss_decoder))

            if args.side_signal:
                encoder_input = torch.cat((side_info, bin_gen_answer), dim=1).clone().detach().requires_grad_(True)
            else:
                encoder_input = bin_gen_answer.clone().detach().requires_grad_(True)

        else:
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.decoder_learning_rate,
                                         weight_decay=args.decoder_weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.decoder_lr_steps, args.decoder_gamma)

            
            record_seen_ids = torch.tensor(record_seen).to(device)
            record_seen_embeds = F.one_hot(record_seen_ids, num_classes=args.ques_num).to(dtype=torch.float32).to(device)

            replay_timer.tic()
            model.eval()
            with torch.no_grad():
                _, gen_answer_seen = model(record_seen_embeds, mask, None, 'decoder')
                gen_answer_seen = binary_answer(args.class_num, gen_answer, mask)
            model.train()
            writer.add_scalar('replay_time', replay_timer.toc(), iter_idx)

            
            for idx, q_id in enumerate(real_record_id_list):
                if q_id in record_seen:
                    index = record_seen.index(q_id)
                    gen_answer_seen[index] = gen_answer_seen[index] + real_answer[idx]
                    mask = mask_update(mask, index, real_user_id_list[idx])
     
                else:
                    record_seen.append(q_id)
                    real_record_embeds = F.one_hot(torch.tensor(q_id).unsqueeze(dim=0), num_classes=args.ques_num).to(dtype=torch.float32).to(device)
                    record_seen_embeds = torch.cat((record_seen_embeds, real_record_embeds), dim=0)
                    gen_answer_seen = torch.cat((gen_answer_seen, real_answer[idx].unsqueeze(0)),dim=0)
                    mask = mask_update(mask, -1, real_user_id_list[idx])

            
            decoder_training_timer.tic()
            previous_answer = gen_answer_seen.detach().to(device).requires_grad_(True)
            gen_answer_seen = gen_answer_seen.detach().to(device).requires_grad_(True)
            mask = mask.detach().to(device).requires_grad_(True)
            real_priori = cumulative_real_priori(record_seen, args.class_num, real_priori_dict).to(device).requires_grad_()
            for i in range(args.inner_decoder_iter_num):

                if i % args.replay_test_num == 0:
                    model.eval()
                    with torch.no_grad():
                        side_info, gen_answer = model(record_seen_embeds, mask, None, 'decoder')
                        bin_gen_answer = binary_answer(args.class_num, gen_answer, mask)
                        acc_count, total_count = replay_acc_rate_2(previous_answer, bin_gen_answer, args.class_num)
                    model.train()
                    
                    writer.add_scalar('Instantaneous Replay Acc rate', acc_count / total_count, iter_idx * args.inner_decoder_iter_num + i)
                    if acc_count / total_count >= args.replay_threshold:
                        write_print_log(args.log_text_dir, "****decoder inner-training early stopping****     Iter index : {}    Inner_decoder : {}          Correct Count : {}    Total Count : {}     ".format(iter_idx, i, acc_count, total_count))
                    
                        early_stop_count += 1
                        writer.add_scalar('Early Stop Rate', early_stop_count / (iter_idx + 1), iter_idx + 1)
                        break
                
                _, gen_answer = model(record_seen_embeds, mask, None, 'decoder')
                cross_entropy, decoder_kl_div, decoder_L1 = Generative_loss(model, gen_answer_seen, gen_answer,
                                                                                    real_priori)
                loss_decoder = -1 * cross_entropy
                optimizer.zero_grad()
                loss_decoder.backward()
                optimizer.step()
                scheduler.step()
                writer.add_scalar('decoder inner-training Cross Entropy', -1 * cross_entropy, iter_idx * args.inner_decoder_iter_num + i)
                writer.add_scalar('decoder inner-training Loss', loss_decoder, iter_idx * args.inner_decoder_iter_num + i)

                if i == (args.inner_decoder_iter_num - 1):
                    model.eval()
                    with torch.no_grad():
                        side_info, gen_answer = model(record_seen_embeds, mask, None, 'decoder')
                        bin_gen_answer = binary_answer(args.class_num, gen_answer, mask)
                        acc_count, total_count = replay_acc_rate_2(previous_answer, bin_gen_answer, args.class_num)
                            
                    model.train()
                    
                    writer.add_scalar('Instantaneous Replay Acc rate', acc_count / total_count, iter_idx * args.inner_decoder_iter_num + i)
                    write_print_log(args.log_text_dir, "****decoder inner-training early stopping****     Iter index : {}    Inner_decoder : {}          Correct Count : {}    Total Count : {}     ".format(iter_idx, i, acc_count, total_count))
                    

            writer.add_scalar('decoder_training_time', decoder_training_timer.toc(), iter_idx)
            write_print_log(args.log_text_dir, "****decoder Iteration Res ****  Iter index : {}         Cross Entropy : {}     Loss:{}".format(iter_idx, -1 * cross_entropy, loss_decoder))
            if args.side_signal:
                writer.add_histogram('side_information', side_info, iter_idx)
                encoder_input = torch.cat((side_info, bin_gen_answer), dim=1).clone().detach().requires_grad_(True)
            else:
                encoder_input = bin_gen_answer.clone().detach().requires_grad_(True)


        write_print_log(args.log_text_dir, "****decoder Iteration Res****     Iter index : {}      Correct Count : {}    Total Count : {}     Replay Correct Rate : {}".format(iter_idx, acc_count, total_count, acc_count / total_count))
        writer.add_scalar('decoder Iteration Res - Replay Correct Rate', acc_count / total_count, iter_idx)

        optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.encoder_learning_rate,
                                         weight_decay=args.encoder_weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.encoder_lr_steps, args.encoder_gamma)
        if iter_idx == (max_iter - 1):
            args.inner_lla_iter_num = 200
        for i in range(args.inner_lla_iter_num):
            
            dataset_size = encoder_input.shape[0]
            rec_y_hat, rec_x_hat = model(None, None, encoder_input, 'encoder')
            if args.side_signal == True:
                expectation_mean, encoder_kl_div, encoder_L1, side_ce = ELBO_loss(args, model, encoder_input, real_priori, rec_y_hat, rec_x_hat)
                loss_encoder = -1 * (expectation_mean - args.encoder_lambda_kl * encoder_kl_div) + args.encoder_lambda_l1 / args.user_num / args.class_num / args.class_num * encoder_L1 + (-1 * args.encoder_lambda_side * side_ce)
            else:
                expectation_mean, encoder_kl_div, encoder_L1, _ = ELBO_loss(args, model, encoder_input, real_priori, rec_y_hat, rec_x_hat)
                loss_encoder = -1 * (expectation_mean - args.encoder_lambda_kl * encoder_kl_div) + args.encoder_lambda_l1 / args.user_num / args.class_num / args.class_num * encoder_L1
            optimizer.zero_grad()
            loss_encoder.backward()
            optimizer.step()
            scheduler.step()
            
            writer.add_scalar('encoder Inner-training Cross Entropy', -1 * expectation_mean, iter_idx * args.inner_lla_iter_num + i)
            writer.add_scalar('encoder Inner-training Loss', loss_encoder, iter_idx * args.inner_lla_iter_num + i)

            incorrect_qid, correct_count = _test_immediately(args, model, iter_idx, encoder_input, record_seen)
            writer.add_scalar('Instantaneous Cumulative Prediction Accuracy Rate', correct_count/dataset_size, iter_idx * args.inner_lla_iter_num + i)

        write_print_log(args.log_text_dir, "****encoder Iteration Res****   Iter index : {}       Cross Entropy : {}     Loss:{}".format(iter_idx, -1 * expectation_mean, loss_encoder))

        incorrect_qid, correct_count = _test_immediately(args, model, iter_idx, encoder_input, record_seen)
        write_print_log(args.log_text_dir, '****encoder Iteration Res****   Iter_idx: {}       Cumulative record Number: {}      Correct predicted Number: {}   Acc Rate: {}'.format(iter_idx, dataset_size, correct_count, correct_count/dataset_size))
        
        writer.add_scalar('Iteration Cumulative Prediction Accuracy Rate', correct_count/dataset_size, iter_idx)
    
        write_print_log(args.log_text_dir, '')
    
    save_name = os.path.join(args.model_save_dir, 'model_{}.pkl'.format(iter_idx))
    torch.save({'model': model.state_dict(),'iter_idx': iter_idx}, save_name)

