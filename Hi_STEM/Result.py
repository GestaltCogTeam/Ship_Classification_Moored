# -*- coding: UTF-8 -*-

import datetime


class Recorder:
    def __init__(self, result_file_name, training_details):
        self.epoch_number = 0

        self.train_start_time = 0
        self.train_end_time = 0
        self.test_start_time = 0
        self.test_end_time = 0

        self.acc_1 = 0
        self.acc_5 = 0
        self.macro_P = 0
        self.macro_R = 0
        self.macro_f1 = 0

        self.results = list()
        self.f_w = open(result_file_name, 'a+')
        self.f_w.write(str(datetime.datetime.now()) + "\n")
        self.f_w.write(training_details + "\n")

    def __del__(self):
        self.f_w.flush()
        self.f_w.close()

    def add_epoch_number(self, epoch_number):
        self.epoch_number = epoch_number

    def add_train_start_time(self, train_start_time):
        self.train_start_time = train_start_time

    def add_train_end_time(self, train_end_time):
        self.train_end_time = train_end_time

    def add_test_start_time(self, test_start_time):
        self.test_start_time = test_start_time

    def add_test_end_time(self, test_end_time):
        self.test_end_time = test_end_time

    def add_result(self, acc_1, acc_5, macro_P, macro_R, macro_f1):
        self.acc_1 = acc_1
        self.acc_5 = acc_5
        self.macro_P = macro_P
        self.macro_R = macro_R
        self.macro_f1 = macro_f1

    def record(self):
        result = Result(self.epoch_number, self.train_start_time, self.train_end_time, self.test_start_time, self.test_end_time, self.acc_1, self.acc_5, self.macro_P, self.macro_R, self.macro_f1)
        self.results.append(result)
        self.f_w.write(result.to_string())

    def show_results(self):
        for result in self.results:
            print(result.to_string(), end="")


class Result:
    def __init__(self, epoch_number, train_start_time, train_end_time, test_start_time, test_end_time, acc_1, acc_5, macro_P, macro_R, macro_f1):
        self.epoch_number = str(epoch_number)
        self.train_start_time = str(train_start_time)
        self.train_end_time = str(train_end_time)
        self.test_start_time = str(test_start_time)
        self.test_end_time = str(test_end_time)
        self.acc_1 = str(acc_1)
        self.acc_5 = str(acc_5)
        self.macro_P = str(macro_P)
        self.macro_R = str(macro_R)
        self.macro_f1 = str(macro_f1)

    def to_string(self):
        return " | ".join([self.epoch_number, self.train_start_time, self.train_end_time, self.test_start_time, self.test_end_time, self.acc_1, self.acc_5, self.macro_P, self.macro_R, self.macro_f1]) + "\n"
