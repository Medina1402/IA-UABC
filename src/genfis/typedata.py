# x = [XA([XA(5, 7)], 6), XA(5, 6)]
#     print(x[0].value[0].value)
#   FIS.INPUT[ K ].MF[ M ].PARAMS
from typing import List


class RL:
    def __init__(self):
        self.antecedent = []


class PM:
    def __init__(self):
        self.params = []


class MF:
    def __init__(self):
        self.mf = []

    def append(self, data: PM):
        self.mf.append(data)


class FIS:
    def __init__(self):
        self.input = []
        self.output = []
        self.rule = []

    def appendInput(self, data: MF):
        self.input.append(data)

    def appendOutput(self, data: MF):
        self.output.append(data)

    def appendRule(self, data: RL):
        self.input.append(data)
