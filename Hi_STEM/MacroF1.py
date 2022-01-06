class MacroF1:
    def __init__(self, class_number):
        self.class_number = class_number
        self.TP = 0.0  # True Positive
        self.FP = 0.0  # False Positive
        self.FN = 0.0  # False Negative
        self.TN = 0.0  # True Negative

        self.P = 0.0
        self.R = 0.0
        self.macro_f1 = 0.0

    def get_P(self):
        if self.TP + self.FP == 0:
            return 0
        self.P = self.TP / (self.TP + self.FP)
        return self.P

    def get_R(self):
        if self.TP + self.FN == 0:
            return 0
        self.R = self.TP / (self.TP + self.FN)
        return self.R

    def get_marcof1(self):
        self.get_P()
        self.get_R()
        if self.P + self.R == 0:
            return 0
        self.macro_f1 = 2 * (self.P * self.R) / (self.P + self.R)
        return self.macro_f1

    def toString(self):
        self.get_marcof1()
        return str(self.class_number) + " P: " + str(self.P) + " R: " + str(self.R) + " macro_f1: " + str(self.macro_f1)

