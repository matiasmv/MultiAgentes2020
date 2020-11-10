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
            agentIndex ([type]): [indice del agente]
        Returns:
            [type]: [description]
        """
        processed_obs = game_util.process_state_ghost(
            gameState, self.view_distance, agentIndex)
        # TODO: Implementar función de evaluación que utilice "processed_obs"
        rewards = []
        fantasma_x, fantasma_y, fantamsa_v = self.find_procesed_obs_values(processed_obs, [7,8])
        pacman_x, pacman_y, _ = self.find_procesed_obs_values(processed_obs, [6])
        
        caminos = []

        max_x = max(pacman_x, fantasma_x)
        max_y = max(pacman_y, fantasma_y)
        min_x = min(pacman_x, fantasma_x)
        min_y = min(pacman_y, fantasma_y)

        direccion_x = 1 if pacman_x - fantasma_x > 0 else -1
        direccion_y = 1 if pacman_y - fantasma_y > 0 else -1

        obs = processed_obs
        # prueba de camino al pacman 
        print(min_x, max_x)
        print(min_y, max_y)
        for i in range(min_x, max_x):
            next_x_move = (i+1) * direccion_x
            if next_x_move >=min_x and next_x_move < max_x:
                value = obs[next_x_move, fantasma_y]
                if value == 1:
                    next_y_move = (fantasma_y+1)*direccion_y
                    if next_y_move >= min_y and next_y_move < max_y:
                        value = obs[i, next_y_move]

                caminos.append(value)
            
               
        if not 6 in caminos:
            for j in range(min_y, max_y):
                next_y_move = (j+1) * direccion_y
                if next_y_move >=min_y and next_y_move < max_y:
                    value = obs[fantasma_x, next_y_move]
                    if value == 1:
                        next_x_move = (fantasma_x+1)*direccion_x
                        if next_x_move >=min_x and next_x_move < max_x:
                            value = value = obs[next_x_move, j]

                    caminos.append(value)
        
        reward = 50 if 6 in caminos else 0
        
        rewards = np.zeros(gameState.getNumAgents())
        rewards[agentIndex] = reward
        return rewards # vector de recompensas por agente


    # auxiliar
    def estimate(self, agent_value, cell_value):
        value = cell_value
        value =  -value if value == 1 else value

        if (value == 6) & (agent_value == 7):
            value = 50
        if (value == 6) & (agent_value == 8):
            value = -50

        return value
    
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
                print(obs_value, values)
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

    def getNextAgentIndex(self, agentIndex, gameState):  # TODO: Implementar función
        nextIndex = ((agentIndex+1) % gameState.getNumAgents())
        return nextIndex

    def maxN(self, gameState: GameStateExtended, agentIndex, depth):
        #TODO: Implementar
        
        # Casos base:
        if depth == 0:
            result = self.evaluationFunction(gameState, agentIndex)
            print(result)
            return result
        elif gameState.isEnd():
            result = gameState._get_agent_reward(agentIndex)
            print(f"end {result}")
            return result

        # Llamada recursiva
        legalActions = gameState.getLegalActions(agentIndex)
        random.shuffle(legalActions)
        nextAgent = self.getNextAgentIndex(agentIndex, gameState)
        
        ## self.maxN(gameState, nextAgent, depth-1)
        nextStatesValues = []
        print(f"legal actions = {legalActions}")
        for action in legalActions:
            
            state = gameState.deepCopy()
    
            nextState = state.generateSuccessor(agentIndex, action)            
            #print(f"maxN player {agentIndex} action {action}, nextPlayer:{nextAgent}")

            state_values = self.montecarlo_eval(nextState, nextAgent)
            nextStatesValues.append([action, state_values[agentIndex]])

        # print(f"values ={nextStatesValues}")
        max_index, max_value = max(enumerate(nextStatesValues), key=lambda p: p[0])
        # print(f"r = {max_value}, {max_index}")
        best_action = max_value[0] 
        best_score_array = max_value[1]
        # best_score_array = np.zeros(gameState.getNumAgents())

        return best_action, best_score_array

    def montecarlo_eval(self, gameState, agentIndex):
        # TODO: Implementar función
        # Pista: usar random_unroll

        suma = np.zeros(gameState.getNumAgents())
        for i in range(self.number_of_unrolls):
            #print(f"numero de unroll {i}")
            unroll_array = self.random_unroll(gameState, agentIndex)
            
            #print(f"unroll_array {unroll_array}")
            for j in range(len(unroll_array)):
                suma[j] += unroll_array[j]
        
        for j in range(len(suma)):
            suma[j] = suma[j] / self.number_of_unrolls 

        return suma

    def random_unroll(self, gameState: GameStateExtended, agentIndex):
        """
            Parámetros: estado del juego y número de agente
            Retorna: valor del estado luego de realizar un unroll aleatorio
        """
        state = gameState
        V = np.zeros(state.getNumAgents())
        # generar un episodio random
        # computar su valor
        print("------------------------------------------------------")
        unroll_number = self.max_unroll_depth
        player = agentIndex
        while (not state.isEnd()) & (unroll_number > 0):
            unroll_number -= 1

            actions = state.getLegalActions(player)
            random_action = random.choice(actions)
            #print(f"Agent: {player} action: {random_action}")
            state = state.generateSuccessor(player, random_action)

            if state.isEnd():
                print(f"is end")
                V = state.get_rewards()
            elif unroll_number <= 0:
                unroll_value = self.evaluationFunction(state, player)
                print(f"unroll value")
                V = unroll_value
            
            player = self.getNextAgentIndex(player, state)
        # print(f"unroll v= {V}")
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