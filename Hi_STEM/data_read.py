# encoding = utf-8

from collections import defaultdict

import time

import numpy as np

import random

from gensim.models import word2vec
import gensim
import numpy as np


from Hi_STEM.config import Config

brightkite_data_file = Config.data_path + "brightkite_scopus_tra_1215.dat"

class CheckIn:

    def __init__(self, user_id, time, latitude, longitude, location_id, user_type):
        self.user_id = user_id
        self.time = time
        self.latitude = latitude
        self.longitude = longitude
        self.location_id = location_id
        self.user_type = user_type

        # 这里不能这么写呢
        # self.check_embedding_vectors = get_check_in_embedding_vector()

    # 这里添加check-in embedding 向量的获取函数
    # def get_embedding_vector(self):
    #     return self.check_embedding_vectors[self.location_id]

    def string(self):
        return " ".join([self.user_id, self.time, self.latitude, self.longitude, self.location_id, self.user_type])


def split_train_test_data(trajectory_set):
    test_rate = 0.1
    train_trajectory_set = defaultdict(list)
    test_trajectory_set = defaultdict(list)

    for user in trajectory_set.keys():
        trajectory = trajectory_set[user]
        trajectory_length = len(trajectory)
        train_trajectory_set[user] = trajectory[0: trajectory_length - int(trajectory_length * test_rate)]
        test_trajectory_set[user] = trajectory[trajectory_length - int(trajectory_length * test_rate):]

        # print(trajectory_length)
        # print(int(trajectory_length * test_rate))
        # print(trajectory_length - int(trajectory_length * test_rate))
        # a = input()

    return train_trajectory_set, test_trajectory_set


def get_check_in_embedding_vector():
    check_in_embedding_vectors = dict()

    id_list = []

    model = gensim.models.Word2Vec.load('location2vec')

    location_dict = {}

    user_dict = {}

    with open("training_passenger", 'r') as f:
        for line in f:
            id = line.split()[0]
            id_list.append(id)
            tra = line.split()[1:]
            for location in tra:
                location_dict[location] = 1

    with open("training_fishing", 'r') as f:
        for line in f:
            id = line.split()[0]
            id_list.append(id)
            tra = line.split()[1:]
            for location in tra:
                location_dict[location] = 1

    with open("training_oil", 'r') as f:
        for line in f:
            id = line.split()[0]
            id_list.append(id)
            tra = line.split()[1:]
            for location in tra:
                location_dict[location] = 1

    with open("training_cargo", 'r') as f:
        for line in f:
            id = line.split()[0]
            id_list.append(id)
            tra = line.split()[1:]
            for location in tra:
                location_dict[location] = 1

    with open("test_passenger", 'r') as f:
        for line in f:
            id = line.split()[0]
            id_list.append(id)
            tra = line.split()[1:]
            for location in tra:
                location_dict[location] = 1

    with open("test_fishing", 'r') as f:
        for line in f:
            id = line.split()[0]
            id_list.append(id)
            tra = line.split()[1:]
            for location in tra:
                location_dict[location] = 1

    with open("test_oil", 'r') as f:
        for line in f:
            id = line.split()[0]
            id_list.append(id)
            tra = line.split()[1:]
            for location in tra:
                location_dict[location] = 1

    with open("test_cargo", 'r') as f:
        for line in f:
            id = line.split()[0]
            id_list.append(id)
            tra = line.split()[1:]
            for location in tra:
                location_dict[location] = 1

    # with open(brightkite_data_file, 'r')as f:
    #     #     for line in f:
    #     #         user_id = line.split()[0]
    #     #
    #     #         tra = line.split()[1:]
    #     #
    #     #         user_dict[user_id] = 0
    #     #         if len(user_dict.keys()) > 92:
    #     #             break
    #     #
    #     #         for location in tra:
    #     #             location_dict[location] = 1

    for key in location_dict.keys():
        check_in_embedding_vectors[key] = model[str(key)]

    # with open('gowalla_vector_new250d.dat', 'r') as f:
    #     for line in f:
    #         if (len(line.split()) < 100 or line.split()[0] == '</s>'):
    #             continue
    #         check_in_id = line.split()[0]
    #         embedding_vector = line.split()[1:]
    #         check_in_embedding_vectors[check_in_id] = [float(a) for a in embedding_vector]

    return check_in_embedding_vectors

def trajectory_seg(trajectory_set, time_interval=3600 * 6):
    time_format = "%Y-%m-%dT%H:%M:%SZ"

    # sort by time as increase
    for user in trajectory_set.keys():
        trajectory = trajectory_set[user]
        trajectory = sorted(trajectory, key=lambda check_in: check_in.time, reverse=False)
        trajectory_set[user] = trajectory

    # seg trajectory as time_interval
    for user in trajectory_set.keys():
        trajectory = trajectory_set[user]
        trajectory_set[user] = []
        trajectory_segment = []
        first_time = time.mktime(time.strptime(trajectory[0].time, time_format))  # 初始时刻时间
        for location in trajectory:
            now_time = time.mktime(time.strptime(location.time, time_format))  # 当前时间
            if (now_time - first_time) < time_interval:
                trajectory_segment.append(location)

            else:
                trajectory_set[user].append(trajectory_segment)
                trajectory_segment = []
                # first_time = time.mktime(time.strptime(location.time, time_format))
                trajectory_segment.append(location)

            first_time = time.mktime(time.strptime(location.time, time_format))

    # for user in trajectory_set.keys():
    #     trajectory = trajectory_set[user]
    #
    #     for trajectory_seg in trajectory:
    #         print(len(trajectory_seg))
    #         for location in trajectory_seg:
    #             print(location.string())
    #     a = input()

    return trajectory_set


def read_data_1():
    trajectory_set = defaultdict(list)

    user_dict = {}

    brightkite_data_file = Config.data_path + "brightkite_scopus_tra_1215.dat"

    with open(brightkite_data_file, 'r') as f:
        for line in f:
            user_id = line.split()[0]

            if user_id == "99":
                continue


            tra = line.split()[1:]
            trajectory_new = []
            user_dict[user_id] = 1
            # print(user_id)

            if len(user_dict.keys()) > 92:
                break

            for location_id in tra:
                chech_in = CheckIn(user_id, 0, 0, 0, location_id)
                trajectory_new.append(chech_in)

            trajectory_set[user_id].append(trajectory_new)

        print(trajectory_set.keys())
        print(len(trajectory_set.keys()))
        for key in trajectory_set.keys():
            print(len(trajectory_set[key]))
        input()

    # check_in_embedding_vectors = get_check_in_embedding_vector()

    return trajectory_set


def read_data_2():
    trajectory_set = defaultdict(list)

    user_dict = {}

    with open("training_passenger", 'r') as f:
        for line in f:
            user_id = line.split()[0]

            tra = line.split()[1:]
            trajectory_new = []
            user_dict[user_id] = 1
            # print(user_id)

            for location_id in tra:
                chech_in = CheckIn(user_id, 0, 0, 0, location_id, "0")
                trajectory_new.append(chech_in)

            trajectory_set[user_id].append(trajectory_new)

    with open("training_fishing", 'r') as f:
        for line in f:
            user_id = line.split()[0]

            tra = line.split()[1:]
            trajectory_new = []
            user_dict[user_id] = 1
            # print(user_id)

            for location_id in tra:
                chech_in = CheckIn(user_id, 0, 0, 0, location_id, "1")
                trajectory_new.append(chech_in)

            trajectory_set[user_id].append(trajectory_new)

    with open("training_oil", 'r') as f:
        for line in f:
            user_id = line.split()[0]

            tra = line.split()[1:]
            trajectory_new = []
            user_dict[user_id] = 1
            # print(user_id)

            for location_id in tra:
                chech_in = CheckIn(user_id, 0, 0, 0, location_id, "2")
                trajectory_new.append(chech_in)

            trajectory_set[user_id].append(trajectory_new)

    with open("training_cargo", 'r') as f:
        for line in f:
            user_id = line.split()[0]

            tra = line.split()[1:]
            trajectory_new = []
            user_dict[user_id] = 1
            # print(user_id)

            for location_id in tra:
                chech_in = CheckIn(user_id, 0, 0, 0, location_id, "3")
                trajectory_new.append(chech_in)

            trajectory_set[user_id].append(trajectory_new)

    return trajectory_set


def read_data_3():
    trajectory_set = defaultdict(list)

    user_dict = {}

    with open("test_passenger", 'r') as f:
        for line in f:
            user_id = line.split()[0]

            tra = line.split()[1:]
            trajectory_new = []
            user_dict[user_id] = 1
            # print(user_id)

            for location_id in tra:
                chech_in = CheckIn(user_id, 0, 0, 0, location_id, "0")
                trajectory_new.append(chech_in)

            trajectory_set[user_id].append(trajectory_new)

    with open("test_fishing", 'r') as f:
        for line in f:
            user_id = line.split()[0]

            tra = line.split()[1:]
            trajectory_new = []
            user_dict[user_id] = 1
            # print(user_id)

            for location_id in tra:
                chech_in = CheckIn(user_id, 0, 0, 0, location_id, "1")
                trajectory_new.append(chech_in)

            trajectory_set[user_id].append(trajectory_new)

    with open("test_oil", 'r') as f:
        for line in f:
            user_id = line.split()[0]

            tra = line.split()[1:]
            trajectory_new = []
            user_dict[user_id] = 1
            # print(user_id)

            for location_id in tra:
                chech_in = CheckIn(user_id, 0, 0, 0, location_id, "2")
                trajectory_new.append(chech_in)

            trajectory_set[user_id].append(trajectory_new)

    with open("test_cargo", 'r') as f:
        for line in f:
            user_id = line.split()[0]

            tra = line.split()[1:]
            trajectory_new = []
            user_dict[user_id] = 1
            # print(user_id)

            for location_id in tra:
                chech_in = CheckIn(user_id, 0, 0, 0, location_id, "3")
                trajectory_new.append(chech_in)

            trajectory_set[user_id].append(trajectory_new)

    return trajectory_set



class DataReader:
    def __init__(self):
        # trajectory_set = read_data_2()

        # self.train_trajectory_set, self.test_trajectory_set = split_train_test_data(trajectory_set)

        self.train_trajectory_set = read_data_2()
        self.test_trajectory_set = read_data_3()

        self.train_x = []
        self.train_y = []

        for user in self.train_trajectory_set.keys():
            for trajectory_segment in self.train_trajectory_set[user]:
                self.train_y.append(trajectory_segment[0].user_type)
                self.train_x.append(trajectory_segment)

        self.train_data_number = len(self.train_x)
        self.read_train_data_count = 0
        self.user_list = list(self.train_trajectory_set.keys())
        self.user_dict = dict()

        for i in range(0, len(self.user_list)):
            self.user_dict[self.user_list[i]] = i

        print(self.user_dict)

        self.check_embedding_vectors = get_check_in_embedding_vector()

        self.test_x = []
        self.test_y = []

        for user in self.test_trajectory_set.keys():
            for trajectory_segment in self.test_trajectory_set[user]:
                self.test_y.append(trajectory_segment[0].user_type)
                self.test_x.append(trajectory_segment)

        self.test_data_number = len(self.test_x)
        self.read_test_data_count = 0

        self.user_list = list(self.train_trajectory_set.keys())
        self.user_dict = dict()

        for i in range(0, len(self.user_list)):
            self.user_dict[self.user_list[i]] = i

        print(self.user_dict)

        self.check_embedding_vectors = get_check_in_embedding_vector()

        self.zero_embedding_vectors = [0] * 250

        random.shuffle(self.train_x)

    def get_one_hot(self, user_type):
        y = [0] * 4
        y[int(user_type)] = 1
        return y

    def get_embedding(self, trajectory_segment, max_length):

        trajectory_segment_embedding = []

        for i in range(0, max_length):
            trajectory_segment_embedding.append(self.zero_embedding_vectors)

        for i in range(0, len(trajectory_segment)):
            trajectory_segment_embedding[i] = self.check_embedding_vectors[trajectory_segment[i].location_id]
        return trajectory_segment_embedding

    # def get_embedding_mixup(self, trajectory_segment1, trajectory_segment2, lam):
    #
    #     trajectory_segment_embedding = []
    #
    #     for i in range(0, max_length):
    #         trajectory_segment_embedding.append(self.zero_embedding_vectors)
    #
    #     for i in range(0, len(trajectory_segment)):
    #         trajectory_segment_embedding[i] = self.check_embedding_vectors[trajectory_segment[i].location_id]
    #     return trajectory_segment_embedding

    def read_train_data_mixup(self, batch_size=1):

        lam = np.random.beta(1, 1)

        x, y, seq_len = [], [], []

        if self.read_train_data_count >= self.train_data_number - batch_size:
            self.read_train_data_count = 0
            return None, None, None

        max_length = 0

        for i in range(self.read_train_data_count, self.read_train_data_count + batch_size):
            max_length = max(max_length, len(self.train_x[i]))

        for i in range(self.read_train_data_count, self.read_train_data_count + batch_size):
            x.append(self.get_embedding(self.train_x[i], max_length))
            y.append(self.get_one_hot(self.train_x[i][0].user_type))

            # print("len x", len(x[i-self.read_train_data_count]))

            seq_len.append(len(x[i - self.read_train_data_count]))

        self.read_train_data_count += batch_size

        x = np.array(x)
        y = np.array(y)

        x, y, seq_len = self.mix_up(x, y, seq_len, batch_size, lam)

        return x, y, seq_len

    def mix_up(self, x, y, seq_len, batch_size, lam):

        temp_x = x[0]
        temp_y = y[0]
        temp_len = seq_len[0]

        for i in range(0, batch_size-1):
            x[i] = lam * x[i] + (1-lam) * x[i+1]
            y[i] = lam * y[i] + (1-lam) * y[i+1]
            seq_len[i] = max(seq_len[i], seq_len[i+1])

        x[batch_size-1] = lam * x[batch_size-1] + (1 - lam) * temp_x
        y[batch_size-1] = lam * y[batch_size-1] + (1 - lam) * temp_y
        seq_len[batch_size-1] = max(seq_len[batch_size-1], temp_len)

        return x, y, seq_len

    def read_train_data(self, batch_size=1):

        x, y, seq_len = [], [], []

        if self.read_train_data_count >= self.train_data_number - batch_size:
            self.read_train_data_count = 0
            # random.shuffle(self.train_x)
            return None, None, None

        max_length = 0

        for i in range(self.read_train_data_count, self.read_train_data_count + batch_size):
            max_length = max(max_length, len(self.train_x[i]))

        for i in range(self.read_train_data_count, self.read_train_data_count + batch_size):
            x.append(self.get_embedding(self.train_x[i], max_length))
            # y.append(self.get_one_hot(self.train_y[i]))
            y.append(self.get_one_hot(self.train_x[i][0].user_type))

            # print("len x", len(x[i-self.read_train_data_count]))

            seq_len.append(len(x[i-self.read_train_data_count]))

        self.read_train_data_count += batch_size

        # x = np.concatenate(x, axis=0)
        x = np.array(x)
        return x, y, seq_len

    def read_train_data_old(self, batch_size=1):

        x, y, seq_len = [], [], []

        if self.read_train_data_count >= self.train_data_number - batch_size:
            self.read_train_data_count = 0
            return None, None, None

        for i in range(self.read_train_data_count, self.read_train_data_count + batch_size):
            x.append(self.get_embedding(self.train_x[i]))
            # y.append(self.get_one_hot(self.train_y[i]))
            y.append(self.get_one_hot(self.train_x[i][0].user_id))

            seq_len.append(len(x[i - self.read_train_data_count]))

            # print("len x", len(x[i-self.read_train_data_count]))

        self.read_train_data_count += batch_size

        # x = np.concatenate(x, axis=0)
        x = np.array(x)
        return x, y, seq_len

    def read_test_data(self, batch_size=1):

        x, y, seq_len = [], [], []

        if self.read_test_data_count >= self.test_data_number - batch_size:
            self.read_test_data_count = 0
            return None, None, None

        max_length = 0

        for i in range(self.read_test_data_count, self.read_test_data_count + batch_size):
            max_length = max(max_length, len(self.test_x[i]))

        for i in range(self.read_test_data_count, self.read_test_data_count + batch_size):
            x.append(self.get_embedding(self.test_x[i], max_length))
            y.append(self.get_one_hot(self.test_x[i][0].user_type))

            # print("len x", len(x[i-self.read_train_data_count]))

            seq_len.append(len(x[i-self.read_test_data_count]))

        self.read_test_data_count += batch_size

        # x = np.concatenate(x, axis=0)
        x = np.array(x)
        return x, y, seq_len

    def read_test_data_old(self, batch_size=1):

        x, y, seq_len = [], [], []

        if self.read_test_data_count >= self.test_data_number - batch_size:
            self.read_test_data_count = 0
            return None, None, None

        for i in range(self.read_test_data_count, self.read_test_data_count + batch_size):
            x.append(self.get_embedding(self.test_x[i]))
            y.append(self.get_one_hot(self.test_y[i]))

            seq_len.append(len(x[i - self.read_test_data_count]))

        self.read_test_data_count += batch_size

        return x, y, seq_len


if __name__ == "__main__":
    # read_data()
    # trajectory_set = read_data()
    # trajectory_set = trajectory_seg(trajectory_set)

    reader = DataReader()

    count = 0
    while True:
        x_in, y_in, seq_len = reader.read_test_data(batch_size=1)
        if x_in is None:
            break
        count += 1
        print(y_in[0].index(1))

    print(count)

    # for i in range(0, len(x)):
    # #     print("length", len(x[i]))
    # #     for location in x[i]:
    # #         print(location.string())
    #
    # count = 0
    # while True:
    #     x_in, y_in, seq_len = reader.read_train_data(batch_size=1)
    #     if x_in is None:
    #         break
    #     count += 1
    #
    # print(count)

    #
    # read_data_1()
