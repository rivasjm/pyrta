from model import System


def invslack(system: System) -> float:
    return max([(flow.wcrt-flow.deadline)/flow.deadline for flow in system.flows])