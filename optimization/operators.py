import numpy as np
import biogeme
from biogeme import vns
import biogeme.exceptions as excep
import copy
import biogeme.messaging as msg
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import random





class onesolution(vns.solutionClass):
    """Implements the virtual class. A solution here is a
    configuration.

    """
    def __init__(self, solution):
        super().__init__()
        self.x = solution
        self.objectivesNames = ['sanitary_cost','economic_cost']
        self.objectives = None

    def isDefined(self):
        """Check if the decision variables iarewell defined.

        :return: True if the configuration vector ``x`` is defined,
            and the total deaths and total Cobb-Douglas are both defined.
        :rtype: bool
        """
        if self.x is None:
            return False
        if self.objectives is None:
            return False
        return True

    def __repr__(self):
        return str(self.x)

    def __str__(self):
        return str(self.x)
    
# ------------------ TIME OPERATORS ------------------ #

def increase_policy_end(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    p2=list(solc[2:])
    new_end = solc[1] + size
    proj_new_end = np.clip(new_end,solc[0]+1 , 20)
    if solc[1] == proj_new_end:
        return sol, 0
    xplus=[solc[0], proj_new_end, *p2]   
    print("INCREASE POLICY END ", xplus, flush=True)
    return onesolution(xplus), 1

def decrease_policy_end(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    p2=list(solc[2:])
    new_end = solc[1] - size
    proj_new_end = np.clip(new_end,solc[0]+1 , 20)
    if solc[1] == proj_new_end:
        return sol, 0
    xplus=[solc[0], proj_new_end, *p2]   
    print("DECREASE POLICY END ", xplus, flush=True)
    return onesolution(xplus), 1

def increase_policy_beginning(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    p2=list(solc[2:])
    new_beginning= solc[0] + size
    proj_new_beginning = np.clip(new_beginning,1 , solc[1]-1)
    if solc[0] == proj_new_beginning:
        return sol, 0
    xplus=[proj_new_beginning, solc[1], *p2]   
    print("INCREASE POLICY BEGINNING ", xplus, flush=True)
    return onesolution(xplus), 1

def decrease_policy_beginning(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    p2=list(solc[2:])
    new_beginning= solc[0] - size
    proj_new_beginning = np.clip(new_beginning,1 , solc[1]-1)
    if solc[0] == proj_new_beginning:
        return sol, 0
    xplus=[proj_new_beginning, solc[1], *p2]   
    print("DECREASE POLICY BEGINNING ", xplus, flush=True)
    return onesolution(xplus), 1
   
def shift_later(aSolution, size=1):
    # take both initial and end and change
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    p2=list(solc[2:])
    new_beginning = solc[0] + size
    new_end = solc[1] + size
    new_beginning = np.clip( new_beginning , 1, 20)
    new_end = np.clip(new_end, new_beginning + 1, 20)
    if  new_beginning ==  solc[0] or new_end==solc[1]:
        return sol, 0
    xplus=[new_beginning,new_end,*p2]
    print('SHIFT LATER',xplus)
    return onesolution(xplus), 2

def shift_earlier(aSolution, size=1):
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    p2=list(solc[2:])
    new_beginning = solc[0] - size
    new_end = solc[1] - size
    new_beginning = np.clip( new_beginning , 1, 20)
    new_end = np.clip(new_end, new_beginning + 1, 20)
    if  new_beginning ==  solc[0] or new_end==solc[1]:
        return sol, 0
    xplus=[new_beginning,new_end,*p2]
    print('SHIFT EARLIER',xplus)
    return onesolution(xplus), 2



# ------------------ SCHEDULES OPERATORS ------------------ #

def decrease_scenario_0_9(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[2] - size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[2] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1],proj_policy_new , solc[3], solc[4], solc[5], solc[6] ,solc[7], solc[8], solc[9], solc[10]]   
    print("DECREASE SCHEDULE 0-9 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def increase_scenario_0_9(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[2] + size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[2] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1],proj_policy_new , solc[3], solc[4], solc[5], solc[6] ,solc[7], solc[8], solc[9], solc[10]]   
    print("INCREASE SCHEDULE 0-9 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def decrease_scenario_10_19(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[3] - size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[3] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2],proj_policy_new, solc[4], solc[5], solc[6] ,solc[7], solc[8], solc[9], solc[10]]   
    print("DECREASE SCHEDULE 10-19 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def increase_scenario_10_19(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[3] + size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[3] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2],proj_policy_new, solc[4], solc[5], solc[6] ,solc[7], solc[8], solc[9], solc[10]]   
    print("INCREASE SCHEDULE 10-19 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def decrease_scenario_20_29(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[4] - size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[4] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2], solc[3],proj_policy_new, solc[5], solc[6] ,solc[7], solc[8], solc[9], solc[10]]   
    print("DECREASE SCHEDULE 20-29 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def increase_scenario_20_29(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[4] + size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[4] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2], solc[3],proj_policy_new, solc[5], solc[6] ,solc[7], solc[8], solc[9], solc[10]]   
    print("INCREASE SCHEDULE 20-29 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def decrease_scenario_30_39(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[5] - size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[5] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2], solc[3], solc[4],proj_policy_new, solc[6] ,solc[7], solc[8], solc[9], solc[10]]   
    print("DECREASE SCHEDULE 30-39 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def increase_scenario_30_39(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[5] + size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[5] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2], solc[3], solc[4],proj_policy_new, solc[6] ,solc[7], solc[8], solc[9], solc[10]]   
    print("INCREASE SCHEDULE 30-39 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def decrease_scenario_40_49(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[6] - size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[6] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2], solc[3], solc[4], solc[5],proj_policy_new ,solc[7], solc[8], solc[9], solc[10]]   
    print("DECREASE SCHEDULE 40-49 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def increase_scenario_40_49(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[6] + size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[6] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2], solc[3], solc[4], solc[5],proj_policy_new ,solc[7], solc[8], solc[9], solc[10]]   
    print("INCREASE SCHEDULE 40-49 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def decrease_scenario_50_59(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[7] - size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[7] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2], solc[3], solc[4], solc[5], solc[6] ,proj_policy_new, solc[8], solc[9], solc[10]]   
    print("DECREASE SCHEDULE 50-59 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def increase_scenario_50_59(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[7] + size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[7] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2], solc[3], solc[4], solc[5], solc[6] ,proj_policy_new, solc[8], solc[9], solc[10]]   
    print("INCREASE SCHEDULE 50-59 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def decrease_scenario_60_69(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[8] - size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[8] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2], solc[3], solc[4], solc[5], solc[6] ,solc[7], proj_policy_new, solc[9], solc[10]]   
    print("DECREASE SCHEDULE 60-69 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def increase_scenario_60_69(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[8] + size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[8] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2], solc[3], solc[4], solc[5], solc[6] ,solc[7], proj_policy_new, solc[9], solc[10]]   
    print("INCREASE SCHEDULE 60-69 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def decrease_scenario_70_79(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[9] - size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[9] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2], solc[3], solc[4], solc[5], solc[6] ,solc[7], solc[8], proj_policy_new, solc[10]]   
    print("DECREASE SCHEDULE 70-79 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def increase_scenario_70_79(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[9] + size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[9] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2], solc[3], solc[4], solc[5], solc[6] ,solc[7], solc[8], proj_policy_new, solc[10]]   
    print("INCREASE SCHEDULE 70-79 YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def decrease_scenario_80_plus(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[10] - size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[10] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2], solc[3], solc[4], solc[5], solc[6] ,solc[7], solc[8], solc[9], proj_policy_new]   
    print("DECREASE SCHEDULE 80+ YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1

def increase_scenario_80_plus(aSolution, size=1):
    # [0]: beginning
    # [1]: end
    sol=aSolution.x
    solc = copy.deepcopy(sol)
    policy_new = solc[10] + size
    proj_policy_new = np.clip(policy_new,1 , 4)
    if solc[10] ==  proj_policy_new :
        return sol, 0
    xplus=[solc[0], solc[1], solc[2], solc[3], solc[4], solc[5], solc[6] ,solc[7], solc[8], solc[9], proj_policy_new]   
    print("INCREASE SCHEDULE 80+ YEARS OLD", xplus, flush=True)
    return onesolution(xplus), 1
