import random
from entregables import game_util 
from game_logic.game import Agent
from game_logic.gameExtended import GameStateExtended
import numpy as np
from game_logic import mcts_util
import math

class MaxNAgent(Agent):
    def __init__(self, index, max_depth=2, unroll_type="MC", max_unroll_depth=5, number_of_unrolls=10, view_distance=(2, 2)):
        super().__init__(index)

        self.max_depth = max_depth # profundidad máxima de maxN
        self.unroll_type = unroll_type # tipo de unrolling a realizar (“MC” - Monte Carlo o “MCTS” - Monte Carlo Tree Search)
        self.max_unroll_depth = max_unroll_depth # largo máximo de unrolling
        self.number_of_unrolls = number_of_unrolls # cantidad de unrolls por estado a evaluar 
        self.view_distance = view_distance # distancia máxima de visión para la función heurística.

    def evaluationFunction(self, gameState: GameStateExtended, agentIndex):
        """[summary]
            Processed_obs 
            Parámetros: estado del juego, distancia máxima de visión (tupla), indice del agente
            qué está jugando
            ● Retorna: Una matriz de numpy con las siguientes características:
            ○ Shape: (view_distance[0]*2+1, view_distance[1]*2+1) (salvo que área fuera
            de juego sea visible)
            ■ Ej: view_distance: (2,2), shape: (5,5)
            ○ Contienen los elementos que se encuentran rodeando al agente, con éste en
            el centro (en la posición (view_distance[0], view_distance[1])
            ○ Cada elemento está simbolizado por un número:
                ■ 0: empty
                ■ 1: wall
                ■ 2: food
                ■ 3: capsule
                ■ 4: ghost
                ■ 5: scared ghost
                ■ 6: Pac-Man
                ■ 7: playing ghost
                ■ 8: playing scared ghost
            ○ Si el agente se encuentra el un borde del mapa, y área fuera de juego queda
            dentro del view_distance, esta área es omitida, generando una observación
            con una shape modificada.
        Args:
            gameState (GameStateExtended): [estado del juego]
            agentIndex ([int]): [indice del agente]
        Returns:
            [array]: [rewards de los agentes]
        """
        processed_obs = game_util.process_state(
            gameState, self.view_distance, agentIndex)

        print(f"Agent {agentIndex} => evaluation function obs")
        print(processed_obs)
        # TODO: Implementar función de evaluación que utilice "processed_obs"
        fantasma_x, fantasma_y, fantamsa_v = self.find_procesed_obs_values(processed_obs, [7,8])
        pacman_x, pacman_y, _ = self.find_procesed_obs_values(processed_obs, [6])

        obs = processed_obs
        
        x = abs(fantasma_x - pacman_x) 
        y = abs(fantasma_y - pacman_y)

        x= 1/x if x>0 else 1
        y= 1/y if y>0 else 1
        
        x= 0 if x<0.8 else 1
        y= 0 if y<0.8 else 1


        rewards = np.zeros(gameState.getNumAgents())
        rewards[agentIndex] =  (x+y)
        print(f"Agent {agentIndex} => rewards ={rewards}")
        return rewards # vector de recompensas por agente

    # auxiliar
    def find_procesed_obs_values(self, processed_obs, values):
        px = -1
        py = -1
        pv = 0
        i = 0
        j = 0
        encontre = False
        while (not encontre) and (i < processed_obs.shape[0]):
            j=0
            while (not encontre) and (j < processed_obs.shape[1]):
                obs_value = processed_obs[i][j]
                #print(obs_value, values)
                if (obs_value in values):
                    encontre = True
                    px = i
                    py = j
                    pv = processed_obs[i][j]
                
                j += 1
            i += 1

        return px, py, pv

    def getAction(self, gameState):
        action, value = self.maxN(gameState, self.index, self.max_depth)
        return action

    def getNextAgentIndex(self, agentIndex, gameState):
        nextIndex = ((agentIndex+1) % gameState.getNumAgents())
        # print(f"Agent {agentIndex} => nextAgent{nextIndex}")
        return nextIndex

    def maxN(self, gameState: GameStateExtended, agentIndex, depth):
        print(f"Agent {agentIndex} => maxN depth={depth}")
        # Casos base:
        if depth == 0:
            if self.unroll_type == "MC":
                return None, self.montecarlo_eval(gameState, agentIndex)
            else:
                return None, self.montecarlo_tree_search_eval(gameState, agentIndex)
        elif gameState.isEnd():
            return None, gameState.get_rewards()

        # Llamada recursiva
        legalActions = gameState.getLegalActions(agentIndex)
        random.shuffle(legalActions)
        nextAgent = self.getNextAgentIndex(agentIndex, gameState) 
        nextStatesValues = []
        for action in legalActions:
            state = gameState.deepCopy()
            nextState = state.generateSuccessor(agentIndex, action)            
            # De la llamada recursiva solo interesa los scores y no la accion
            _, scores = self.maxN(nextState, nextAgent, depth-1)
            nextStatesValues.append([action, scores])

        #print(f"agentIndex={agentIndex} values ={nextStatesValues}")
        best_action, best_score_array = self.get_best_action_score(agentIndex, nextStatesValues)
        print(f"Agent {agentIndex} => maxN Best Action={best_action} Best Score Array = {best_score_array}")
        return best_action, best_score_array
    
    def get_best_action_score(self, agent, values):
        aux_val = float('-inf')
        aux_index = 0
        for i in range(len(values)):
            if (values[i][1][agent] > aux_val):
                aux_val = values[i][1][agent]
                aux_index = i
        return values[aux_index][0], values[aux_index][1]

    def montecarlo_eval(self, gameState, agentIndex):
        # Pista: usar random_unroll
        rewards = np.zeros(gameState.getNumAgents())
        for i in range(self.number_of_unrolls):
            #print(f"agentIndex={agentIndex} => numero de unroll {i}")
            unroll_array = self.random_unroll(gameState.deepCopy(), agentIndex)
            #print(f"agentIndex={agentIndex} => unroll_array {unroll_array}")
            for j in range(len(unroll_array)):
                rewards[j] += unroll_array[j]
        
        print(f"Agent {agentIndex} => 1 montecarlo eval {rewards}")
        for j in range(len(rewards)):
            rewards[j] = rewards[j] / self.number_of_unrolls 

        print(f"Agent {agentIndex} => montecarlo eval {rewards}")
        return rewards 

    def random_unroll(self, gameState: GameStateExtended, agentIndex):
        """
            Parámetros: estado del juego y número de agente
            Retorna: valor del estado luego de realizar un unroll aleatorio
        """
        #print(f'Agent {agentIndex} => Unroll Start')
        state = gameState
        V = np.zeros(state.getNumAgents())
        unroll_number = self.max_unroll_depth
        player = agentIndex
        while (not state.isEnd()) & (unroll_number > 0):
            unroll_number -= 1
            player = self.getNextAgentIndex(player, state)

            #print(f'Agent {player} => Unroll step {unroll_number}')
            actions = state.getLegalActions(player)
            random_action = random.choice(actions)
            state = state.generateSuccessor(player, random_action)

            if state.isEnd():
                V = state.get_rewards()
            elif unroll_number <= 0:
                # duda player o agent index?
                V = self.evaluationFunction(state, player)
                #print(f"Agent {player} => unroll value = {V}")

        print(f"Agent {agentIndex} => end unroll V={V}")
        return V

    def montecarlo_tree_search_eval(self, gameState, agentIndex):
        # TODO: Implementar función
        # PISTA: Utilizar selection_stage, expansion_stage, random_unroll y back_prop_stage
        root = mcts_util.MCTSNode(
            parent=None, action=None, player=agentIndex, numberOfAgents=gameState.getNumAgents())
        state = gameState.deepCopy()
        for _ in range(self.number_of_unrolls):
            pass

        return np.zeros(gameState.getNumAgents())

    def selection_stage(self, node, gameState):
        # TODO: Implementar función
        return node, gameState

    def expansion_stage(self, node, gameState):
        # TODO: Implementar función
        return node, gameState

    def back_prop_stage(self, node, value):
        # TODO: Implementar función
        pass
