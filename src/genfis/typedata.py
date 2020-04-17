from typing import List


class RL:
    antecedent = []


class PM:
    params = []


class MF:
    mf: List[PM]
    name = ""
    type = ""


class MFInput:
    input: List[MF] = []
    name = "tskt1fls"
    type = "sugeno"
    andMethod = "prod"
    orMethod = "max"
    defuzzMethod = "wtaver"
    impMethod = "prod"
    aggMethod = "max"
    range = []


class FIS:
    input = MFInput()
    output: List[MF] = []
    rule: List[RL] = []
