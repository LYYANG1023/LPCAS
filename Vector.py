import csv
import os
import random
from collections import defaultdict

import numpy as np
random.seed(1023)

def transformIndex(path, filename):

    records = []

    record_list = []
    user_list = []
    with open(os.path.join(path, filename), encoding='UTF-8-sig', newline='') as f:
        f_csv = csv.reader(f)
        header = next(f_csv)

        for line in f_csv:
            example, user, label = line
            records.append([int(example), user, int(label)])
            record_list.append(int(example))
            user_list.append(user)

    record_list = list(set(record_list))
    record_list.sort()
    user_list = list(set(user_list))
    user_list.sort()

    for record in records:
        record[0] = record_list.index(record[0])
        record[1] = user_list.index(record[1])

    transformed_file = os.path.join(path, 'transformed.answer.csv')
    with open(transformed_file, 'w', encoding='UTF-8-sig', newline='') as f_answer:

        writer = csv.writer(f_answer)

        writer.writerow(['record_id', 'user_id', 'label'])

        for record in records:
            writer.writerow(record)

    transformed_file = os.path.join(path, 'transformed.truth.csv')
    with open(os.path.join(path, 'truth.csv'), encoding='UTF-8-sig', newline='') as r_truth, open(transformed_file, 'w', encoding='UTF-8-sig', newline='') as w_truth:

        writer = csv.writer(w_truth)
        writer.writerow(['record_id', 'truth'])

        reader = csv.reader(r_truth)
        header = next(reader)

        for line in reader:
            example, truth = line
            writer.writerow([record_list.index(int(example)),truth])


    transformed_file = os.path.join(path, 'transformed.user.csv')
    with open(os.path.join(path, 'user.csv'), encoding='UTF-8-sig', newline='') as r_truth, open(transformed_file, 'w', encoding='UTF-8-sig', newline='') as w_truth:

        writer = csv.writer(w_truth)
        writer.writerow(['original id', 'transformed id'])

        reader = csv.reader(r_truth)

        for line in reader:
            user = line
            writer.writerow([user[0],user_list.index(user[0])])



    return records, record_list, user_list


def complete_record2Vector(path, filename, class_num):
    q2qwl = defaultdict(dict)
    records, record_list, user_list = transformIndex(path, filename)
    ques_nums = len(record_list)
    user_nums = len(user_list)

    for record in records:
        q2qwl[record[0]][record[1]] = record[2]

    answer_matrix = np.zeros((ques_nums, user_nums * class_num), dtype=float)
    mask_matrix = np.zeros((ques_nums, user_nums * class_num), dtype=float)
    priori_matrix = np.zeros((ques_nums, class_num), dtype=float)

    for ques, w_l in q2qwl.items():
        for user, label in w_l.items():
            mask_matrix[ques][user * class_num] = 1
            mask_matrix[ques][user * class_num + 1] = 1

            answer_matrix[ques][user * class_num + label] = 1

            priori_matrix[ques][label] += 1


    for ques in priori_matrix:
        sum = np.sum(ques)
        for i in range(class_num):
            ques[i] = ques[i] / sum

    split_index = int(ques_nums * 0.8)
    np.savez(os.path.join(path, 'train.answer.npz'), answer_matrix[0:split_index], mask_matrix[0:split_index], priori_matrix[0:split_index])
    np.savez(os.path.join(path, 'valid.answer.npz'), answer_matrix[split_index:-1], mask_matrix[split_index:-1], priori_matrix[split_index:-1])
    np.savez(os.path.join(path, 'test.answer.npz'), answer_matrix, mask_matrix, priori_matrix)

    return answer_matrix, mask_matrix, priori_matrix


def incomplete_record2Vector(path, filename, class_num):
    records, record_list, user_list = transformIndex(path, filename)
    random.shuffle(records)

    record_nums = len(records)
    ques_nums = len(record_list)
    user_nums = len(user_list)

    record_id_matrix = np.zeros((record_nums), dtype=int)
    user_id_matrix = np.zeros((record_nums), dtype=int)
    answer_matrix = np.zeros((record_nums, user_nums * class_num), dtype=float)
    mask_matrix = np.zeros((record_nums, user_nums * class_num), dtype=float)
    priori_matrix = np.zeros((record_nums, class_num), dtype=float)

    index = 0
    for index, record in enumerate(records):
        ques, user, label = record
        record_id_matrix[index] = ques
        user_id_matrix[index] = user
        mask_matrix[index][user * class_num] = 1
        mask_matrix[index][user * class_num + 1] = 1

        answer_matrix[index][user * class_num + label] = 1

        priori_matrix[index][label] += 1.

    np.set_printoptions(threshold=np.inf)
    print(record_id_matrix)

    split_index = int(ques_nums * 0.8)
    np.savez(os.path.join(path, 'incomplete.train.answer.npz'), record_id_matrix[0:split_index], user_id_matrix[0:split_index], answer_matrix[0:split_index], mask_matrix[0:split_index], priori_matrix[0:split_index])
    np.savez(os.path.join(path, 'incomplete.valid.answer.npz'), record_id_matrix[split_index:-1], user_id_matrix[split_index:-1], answer_matrix[split_index:-1], mask_matrix[split_index:-1], priori_matrix[split_index:-1])
    np.savez(os.path.join(path, 'incomplete.total.answer.npz'), record_id_matrix, user_id_matrix, answer_matrix, mask_matrix, priori_matrix)

    return answer_matrix, mask_matrix, priori_matrix

def default_dict():
    return defaultdict(int)

def read_answer(file_dir):
    q_user_dict = defaultdict(default_dict)
    with open(os.path.join(file_dir, 'transformed.answer.csv'), encoding='UTF-8-sig', newline='') as f:
        f_csv = csv.reader(f)
        header = next(f_csv)

        for line in f_csv:
            record, user, label = line
            q_user_dict[int(record)][int(user)] = int(label)
    return q_user_dict

