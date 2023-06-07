import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from abm.model import *
from abm.plotting import *
from abm.parallel import *
from abm.population import *
from abm.characteristics import load_population_dataset
from optimization.operators_2 import *


def prepare_calendars(sol2):
    """Prepares the calendars for the simulation and stores it in "data/abm/vaud/prepared/temp_period_activities"
    sol2 : list of scenarios numbers for each segment

    :return: Number of periods for each activity
    """
    #Merge csv files for each segmantation according to the solution
    segments = ['0 - 9', '10 - 19', '20 - 29', '30 - 39', '40 - 49', '50 - 59', '60 - 69', '70 - 79', '80+']
    for i in range(len(sol2)):
        folder = "data/abm/vaud/prepared/scenarios/scenario_" + str(sol2[i]) + "/"
        file = folder + "vaud_period_activities_" + segments[i] + ".csv.gz"
        temp = pd.read_csv(file)
        if i == 0:
            df = temp
        else: 
            df = pd.concat([df, temp])
    
    unique_periods = df['period'].unique()
    activities = df['type'].unique()
    # Count the number of periods for each activity
    count_activities = dict()
    for activity in activities:
        count_activities[activity] = len(df[df['type'] == activity])

    #Applies preprocessing and saves the restriction schecule
    _AGENTS_ID_TRANSLATIONS_FILE_ = "data/abm/vaud/prepared/vaud_agents_id_translations.csv.gz"
    _FACILITIES_ID_TRANSLATIONS_FILE_ = "data/abm/vaud/prepared/vaud_facilities_id_translations.csv.gz"
    _PERIOD_ACTIVITIES_REP_ = "data/abm/vaud/prepared/temp_period_activities/"
    
    agents_translations = pd.read_csv(_AGENTS_ID_TRANSLATIONS_FILE_, index_col=0)
    agents_translations['agent_index']=agents_translations['agent_index'].astype(int)
    facilities_translations = pd.read_csv(_FACILITIES_ID_TRANSLATIONS_FILE_, index_col=0)
    facilities_translations['facility_index']=facilities_translations['facility_index'].astype(int)
    

    for period_index, period in enumerate(unique_periods):
        #print("Processing period ", period)
        # Isolate the activities that occurred during that period
        sub_activ = df[df['period'] == period]
        # Translate the agent ids to agent index
        sub_activ = sub_activ.merge(agents_translations, left_on="id", right_index=True)
        # Translate the facility names to indexes
        sub_activ = sub_activ.merge(facilities_translations, left_on="facility", right_index=True)
        # Only keep the relevant info
        sub_activ = sub_activ.drop(['age', 'period', 'id', 'facility'], axis=1)
        # Save the sub dataset
        sub_activ.to_csv(os.path.join(_PERIOD_ACTIVITIES_REP_, f"{str(period_index)}.csv.gz"),
                        index=False)
        
    return count_activities


class optimisationcovid(vns.problemClass):
    """Class defining the covid policy problem. Note the inheritance from the
    abstract class used by the VNS algorithm. It guarantees the
    compliance with the requirements of the algorithm.

    """

    def __init__(self, costs,normal_gdp,N,lengthsimulation,n_scenarios, initial_infections, average_remaining_working_years, proportion_work_infection,cost_healthcare, severeforms,proportion_remoteless,reduction_efficiency_remote_work, params):
        """Ctor"""
        super().__init__()
        self.cost_activities=costs
        self.average_remaining_working_years = average_remaining_working_years # Number of years to work
        self.proportion_work_infection = proportion_work_infection
        self.proportion_remoteless = proportion_remoteless
        self.reduction_efficiency_remote_work = reduction_efficiency_remote_work
        self.cost_healthcare = cost_healthcare # Cost of healthcare per day
        self.severeforms = severeforms #Pourcentage of severe forms in the positive tests
        self.params = params
        self.normal_gdp = normal_gdp
        self.N= N
        self.n_scenarios = n_scenarios
        self.lengthsimulation=lengthsimulation
        self.initial_infections = initial_infections,
        self.operators = {
            'Increase_policy_end': increase_policy_end,
            'Decrease_policy_end': decrease_policy_end,
            'Increase_policy_beginning': increase_policy_beginning,
            'Decrease_policy_beginning': decrease_policy_beginning,
            'Shift_later': shift_later,
            'Shift_earlier':shift_earlier,
            'decrease_scenario_0_9': decrease_scenario_0_9,
            'increase_scenario_0_9': increase_scenario_0_9,
            'decrease_scenario_10_19': decrease_scenario_10_19,
            'increase_scenario_10_19': increase_scenario_10_19,
            'decrease_scenario_20_29': decrease_scenario_20_29,
            'increase_scenario_20_29': increase_scenario_20_29,
            'decrease_scenario_30_39': decrease_scenario_30_39,
            'increase_scenario_30_39': increase_scenario_30_39,
            'decrease_scenario_40_49': decrease_scenario_40_49,
            'increase_scenario_40_49': increase_scenario_40_49,
            'decrease_scenario_50_59': decrease_scenario_50_59,
            'increase_scenario_50_59': increase_scenario_50_59,
            'decrease_scenario_60_69': decrease_scenario_60_69,
            'increase_scenario_60_69': increase_scenario_60_69,
            'decrease_scenario_70_79': decrease_scenario_70_79,
            'increase_scenario_70_79': increase_scenario_70_79,
            'decrease_scenario_80_plus': decrease_scenario_80_plus,
            'increase_scenario_80_plus': increase_scenario_80_plus        
        }
        self.operatorsManagement = vns.operatorsManagement(
            self.operators.keys()
        )
        self.currentSolution = None
        self.lastOperator = None
        

    def startsolution(self):
        """
        :return: the first sceneario for each segment (9 age segments), and counts the number of activities in the no-restriction schedules
        :rtype: class onesolution
        """        
        _PERIOD_ACTIVITIES_ = 'data/abm/vaud/prepared/vaud_period_activities.csv.gz'    
        df = pd.read_csv(_PERIOD_ACTIVITIES_, index_col=0)
        activities = df['type'].unique()    
        self.count_activities = dict()
        for activity in activities:
            self.count_activities[activity] = len(df[df['type'] == activity])
        z=np.array([0,self.lengthsimulation,1,1,1,1,1,1,1,1,1])
        return onesolution(z)

    def isValid(self, aSolution):
        """Check if the policy is feasible

        :param aSolution: solution to check
        :type aSolution: class onesolution

        :return: True if the policy is valid
        :rtype: bool
        """
        sol = aSolution.x
        for i in range(2,11):
            if sol[i]>self.n_scenarios:
                return False, 'Scenario must be less than the number of scenarios'
            if sol[i]<1:
                return False, 'Scenario must be greater than 0'
        if sol[1] > self.lengthsimulation:
            return False, 'Policy ends after the simulation ends'
        if sol[0] < 0 :
            return False, 'Policy starts before the simulation starts'
        if sol[1]-sol[0]<3:
            return False, 'Policy must last at least 3 days'
        return True, 'Feasible policy'

    def evaluate(self, aSolution):

        """ Creates a ABM model and evalautes

        :param aSolution: solution to evaluate
        :type aSolution: class onesolution

        """
        sol=aSolution.x
        #if self.isValid(aSolution)[0]:
        solc = copy.deepcopy(sol)
        sol2 = list(solc[2:])

        #Call preprocessing function to obtain the restriction schedule according to the solution and count the number of period per activity
        self.restriction_count_activities = prepare_calendars(sol2)
        #Initializes the simulation and forces initial infections
        self.abm = ABM(self.params)
        self.abm.force_simulation_start(self.initial_infections)
        # Runs the simulation with restriction period
        self.abm.set_param('Restriction_begin', solc[0])
        self.abm.set_param('Restriction_end', solc[1] )
        self.abm.force_simulation_start(self.initial_infections)
        self.abm.run_simulation(self.lengthsimulation, verbose=False)
        
        # Compute I and D
        results = self.abm.results.get_daily_results()
        I = results["daily summed new infections"] # daily new infections
        D = results["daily summed new infections"] # We should replace by number of death 
        totalI= I.sum()
        totalI.round(0)
        totalD= D.sum()
        totalD.round(2)

        # Duration of the policy
        Deltat = solc[1]-solc[0] 
        # Number of confined activities
        count_confined_activities = dict()
        C_policies = 0
        for activity in self.count_activities.keys():
            if activity in self.restriction_count_activities.keys():
                count_confined_activities[activity] = Deltat*(self.count_activities[activity] - self.restriction_count_activities[activity])
                C_policies += count_confined_activities[activity]*self.cost_activities[activity]
            else:
                pass
        C_policies = self.proportion_remoteless * self.reduction_efficiency_remote_work * C_policies

        #Loss
        sanitary_cost = (self.severeforms * totalD + totalI).round(2)
        C_death = self.severeforms * totalD * self.normal_gdp / self.N *self.average_remaining_working_years 
        C_infection = self.proportion_work_infection * self.normal_gdp / self.N / 365 * totalI * 5 # 5 is the average number of days of work lost per infection - should adapt with the model
        C_helthcare = self.cost_healthcare * self.severeforms * totalI 
        economic_cost = C_policies.round(2) #(C_death + C_infection + C_helthcare + C_policies).round(2)/1000
        
        aSolution.objectives = [sanitary_cost, economic_cost]
        
        print("EVALUATE")
        print(aSolution.objectives)
    #else:
        #    aSolution.objectives = [np.inf, np.inf]


    def describe(self, aSolution):
        """Short description of a solution

        :param aSolution: solution to describe
        :type aSolution: class onesolution

        :return: description
        :rtype: str
        """
        sol = aSolution.x
        print(f"{sol[1]}")
        for x in sol[2:]:
            print("Scenario n° ",x)
        return str(aSolution)

    def generateNeighbor(self, aSolution, neighborhoodSize):
        """Generate a neighbor from the negihborhood of size
        ``neighborhoodSize``using one of the operators

        :param aSolution: current solution
        :type aSolution: class onesolution

        :param neighborhoodSize: size of the neighborhood
        :type neighborhoodSize: int

        :return: number of modifications actually made
        :rtype: int

        """
        #print('on est dans generateNeighbor')
        # Select one operator.
        self.lastOperator = self.operatorsManagement.selectOperator()
        return self.applyOperator(
            aSolution, self.lastOperator, neighborhoodSize
        )

    def neighborRejected(self, aSolution, aNeighbor):
        """Informs the operator management object that the neighbor has been
        rejected.

        :param aSolution: current solution
        :type aSolution: class onesolution

        :param aNeighbor: proposed neighbor
        :type aNeighbor: class onesolution

        :raise biogemeError: if no operator has been used yet.
        """
        if self.lastOperator is None:
            raise excep.biogemeError('No operator has been used yet.')
        self.operatorsManagement.decreaseScore(self.lastOperator)

    def neighborAccepted(self, aSolution, aNeighbor):
        """Informs the operator management object that the neighbor has been
        accepted.

        :param aSolution: current solution
        :type aSolution: class onesolution

        :param aNeighbor: proposed neighbor
        :type aNeighbor: class onesolution

        :raise biogemeError: if no operator has been used yet.
        """
        if self.lastOperator is None:
            raise excep.biogemeError('No operator has been used yet.')
        self.operatorsManagement.increaseScore(self.lastOperator)

    def applyOperator(self, solution, name, size=1):
        """Apply a specific operator on a solution, using a neighborhood of
        size ``size``

        :param solution: current solution
        :type solution: class onesolution

        :param name: name of the operator
        :type name: str

        :param size: size of the neighborhood
        :type size: int

        :return: number of modifications actually made
        :rtype: int

        :raise biogemeError: if the name of the operator is unknown.

        """
        op = self.operators.get(name)
        if op is None:
            raise excep.biogemeError(f'Unknown operator: {name}')
        return op(solution, size) 
    




params = {
        'inf_params': {'age': 0.000},
        'test_params': {'age': 0.000},
        'inf_fraction_param': 30,
        'inf_lvl_error_term': -15,
        'inf_proba_sigmoid_slope': 1.0,
        'test_inf_lvl_param': 0.1,
        'test_error_term': -15,
        'test_proba_sigmoid_slope': 10.0,
        'recovery_mean_time': 8.0,
        'recovery_std_time': 2.0,
        'Restriction_begin': 1,
        'Restriction_end': 6}
    

costs = dict({'home': 0, 'shop': 2, 'leisure': 1.5, 'work': 17.2, 'other': 2, 'education': 17.2})
normal_gdp = 62000000
N = 814000
lengthsimulation = 60
initial_infections = (np.arange(8)+1) * 100
n_scenarios = 4
average_remaining_working_years = 20 # Number of years to work
proportion_work_infection = 0.4 # Proportion of the cost of infection that is due to the loss of productivity
proportion_remoteless = 0.50
reduction_efficiency_remote_work = 0.2
cost_healthcare = 4700 # Cost of healthcare per day
severeforms = 0.1 #Pourcentage of severe forms in the positive tests


optim = optimisationcovid(costs,normal_gdp,N,lengthsimulation,n_scenarios, initial_infections, average_remaining_working_years, proportion_work_infection,cost_healthcare, severeforms,proportion_remoteless,reduction_efficiency_remote_work, params)
startsolution = optim.startsolution()
print("start solution",startsolution)

thePareto = vns.vns(
    optim,
    [startsolution],   
    maxNeighborhood=100, 
    numberOfNeighbors=200, 
    archiveInputFile='optimPareto_60days_v1juin_v2_1.pickle',
    pickleOutputFile='optimPareto_60days_v1juin_v2_1.pickle',
)

print(f'Number of pareto solutions: {len(thePareto.pareto)}')
print(f'Pareto solutions: {thePareto.pareto}')
 

for p in thePareto.pareto.keys():
    obj = [f'{t}: {r} ' for t, r in zip(p.objectivesNames, p.objectives)]
    print(f'{p} {obj}')