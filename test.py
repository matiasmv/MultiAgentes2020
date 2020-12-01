from game_logic.ghostAgents import RandomGhost
from entregables.maxNAgent import MaxNAgent
from game_logic.randomPacman import RandomPacman
from game_logic.PacmanEnvAbs import PacmanEnvAbs
import random
import math
import numpy as np
from game_logic.game_util import process_state
from main import get_default_agents
import timeit 
import pandas as pd

all_layouts = [
        "custom1",
        "custom2",
        "capsuleClassic",
        "contestClassic",
        "mediumClassic",
        "minimaxClassic",
        "openClassic",
        "originalClassic",
        "smallClassic",
        "testClassic",
        "trappedClassic",
        "trickyClassic",
        "mediumGrid",
        "smallGrid"
    ]

# Parametros para un test
class TestParams():
    def __init__(self, test_name, layout, pacman_agent, ghost_agent_0, ghost_agent_1):        
        self.test_name = test_name
        self.layout = layout
        self.pacman_agent = pacman_agent
        self.ghost_agent_0 = ghost_agent_0
        self.ghost_agent_1 = ghost_agent_1


def run_test(test_params):  
    t0 = timeit.default_timer()   
    pacman_agent = test_params.pacman_agent
    ghost_agent_0 = test_params.ghost_agent_0
    ghost_agent_1 = test_params.ghost_agent_1
    agents = [pacman_agent, ghost_agent_0, ghost_agent_1]
    agents.extend(get_default_agents(3, 10))    
    done = False
    env = PacmanEnvAbs(agents = agents, view_distance = (2, 2))      
    game_state = env.reset(enable_render= False, layout_name= test_params.layout)
    turn_index = 0
    while (not(done)): # jugar   
        action = agents[turn_index].getAction(game_state)
        game_state, rewards, done, info = env.step(action, turn_index)        
        turn_index = (turn_index + 1) % env._get_num_agents()  
    t1 = timeit.default_timer()    
    time = t1-t0  
    assert(game_state.getNumAgents()>=2) # que el juego tenga mas de 2 agentes
    if game_state.getNumAgents()==2: # vector de rewards con los primeros 2 rewards y nan
        ret = game_state.get_rewards()
        ret.append(np.nan)
    else: # vector de rewards con los primeros 3 rewards
        ret = game_state.get_rewards()[0:3]
    return ret, time


if __name__ == '__main__':
  pacman_agent = RandomPacman(index = 0)
  ghost_agent_0 = MaxNAgent(index = 1, unroll_type="MCTS", max_unroll_depth=12, number_of_unrolls=6)
  ghost_agent_1 = RandomGhost(index = 2)
  sample_test = TestParams("PrimerTest", "mediumGrid", pacman_agent, ghost_agent_0, ghost_agent_1)
  print(run_test(sample_test))