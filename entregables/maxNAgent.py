import random
from game_logic import game_util
from game_logic.mcts_util import ucb
from game_logic.game import Agent
from game_logic.gameExtended import GameStateExtended
import numpy as np
from game_logic import mcts_util
import math

class MaxNAgent(Agent):
    def __init__(
        self, 
        index, 
        max_depth=2, 
        unroll_type="MC", 
        max_unroll_depth=5, 
        number_of_unrolls=10, 
        view_distance=(2, 2),
        use_eval_in_end=False,
        eval_id=1
    ):
        super().__init__(index)
        
        self.EMPTY = 0
        self.WALL = 1
        self.FOOD = 2
        self.CAPSULE = 3
        self.GHOST = 4
        self.GHOST_SCARED = 5
        self.PACMAN = 6

        self.PLAYING_GHOST = 7
        self.PLAYING_GHOST_SCARED = 8 
        self.max_depth = max_depth # profundidad máxima de maxN
        self.unroll_type = unroll_type # tipo de unrolling a realizar (“MC” - Monte Carlo o “MCTS” - Monte Carlo Tree Search)
        self.max_unroll_depth = max_unroll_depth # largo máximo de unrolling
        self.number_of_unrolls = number_of_unrolls # cantidad de unrolls por estado a evaluar 
        self.view_distance = view_distance # distancia máxima de visión para la función heurística.
        self.use_eval_in_end = use_eval_in_end # usar las rewards del juego o la funcion de eval en el is_end
        self.eval_id = eval_id # que funcion de evaluacion usar?

    # Función encargada de devolver un estimativo de ese momento del juego
        # En base al estado del juego, devuelve un valor para cada agente
    def evaluation_function(self, gameState: GameStateExtended, agentIndex):
        pacman_weigths = []
        ghost_weigths = []
        scared_ghost_weigths = []

        if (self.eval_id == 1):
            # Esta evaluacion es como en el ejemplo del Ajedrez
                # Las cosas positivas para cada agente suman
                # Las negativas restan
                # pacman a la vista esta desproporcionado para que se sepa que es un buen estado
            pacman_weigths = [-100, 0, 200, 500, -500, 1000, 0]
            ghost_weigths = [-100, 0, 200, -100, 0, -200, 10000]
            scared_ghost_weigths = [100, 0, -200, -500, 100, -200, -2000]
        elif (self.eval_id == 2):
            # Esta evaluación es como la anterior, pero con otra magnitud en el pacman
                # El pacman en el juego se le va restando de a uno, asi q se ponen magnitudes chicas
                # El fantasma en sus estados se toma como en eval_1 (lo positivo para el +, lo negativo para el -)  
                # Pacman a la vista esta desproporcionado para que se sepa que es un buen estado      
            pacman_weigths = [-1, 0, 4, 10, -5, 10, 0]
            ghost_weigths = [-100, 0, 200, -100, 0, -200, 10000]
            scared_ghost_weigths = [100, 0, -200, -500, 100, -200, -2000]
        else:
            # Esta evaluación trata de castigar al fantasma cada vez q no termino el juego
                # El pacman usa la magnitud anterior
                # Todo resta en el fantasma, cualquiera sea su estado en distintas medidas (menos ver a pacman)       
            pacman_weigths = [-1, 0, 4, 10, -5, 10, 0]
            ghost_weigths = [-100, -100, -200, -100, 0, -200, 10000]
            scared_ghost_weigths = [-100, -100, -200, -500, 0, -200, -2000]                      

        rewards = np.zeros(gameState.getNumAgents())        
        for player in range(gameState.getNumAgents()): # Para cada agente

            value = 0
            # Calcula el estado del juego
            processed_obs = game_util.process_state(gameState, self.view_distance, player)
            # Calcula el agente jugando y la cantidad de elementos que ve en su view
            agent, elements = self.get_view_state(processed_obs, player)
            # Dependiendo de que agente es y que elementos ve, se calcula el value como los elements * pesos
            if agent == self.PACMAN:
                value = np.dot(elements, pacman_weigths)
            elif agent == self.PLAYING_GHOST:
                value = np.dot(elements, ghost_weigths)
            else:
                value = np.dot(elements, scared_ghost_weigths)

            # append el valor a un vector de rewards.
            rewards[player] = value
        # se devuelve las rewards estimadas
        return rewards
        
    # Función encargada de devolver que agente esta jugando y los elementos q ve en su view
    def get_view_state(self, processed_obs, player):
        agent = 0
        if(player == 0):
            agent = self.PACMAN
        else: 
            _, _, agent = self.find_procesed_obs_values(processed_obs, [7,8])

        elements = np.zeros(7)

        for i in range(processed_obs.shape[0]):
            for j in range(processed_obs.shape[1]):
                element = processed_obs[i][j]
                if element != self.PLAYING_GHOST and element != self.PLAYING_GHOST_SCARED:
                    elements[element] += 1

        return agent, elements

    # Función que solo se usa para saber si en una view hay un fantasma o un fantasma asustado jugando
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
                if (obs_value in values):
                    encontre = True
                    px = i
                    py = j
                    pv = processed_obs[i][j]
                
                j += 1
            i += 1

        return px, py, pv

    # Función que usa maxn para definir la accion a realizar
    def getAction(self, gameState):
        action, value = self.max_n(gameState, self.index, self.max_depth)
        return action

    # Función que determina el proximo jugador
    def get_next_agent_index(self, agentIndex, gameState):
        nextIndex = ((agentIndex+1) % gameState.getNumAgents())
        return nextIndex

    # Función que implementa maxn
        # Casos base
            # Si se llega al nivel de profundidad 0 --> Estimacion MonteCarlo o MonteCarlo Tree Search
            # Si se llega al final del juego --> Se devuelve la estimacion o las rewards del juego
                # Esto se realiza para testear si obtener las mismas magnitudes en los dos es bueno
        # Sino se ejecuta maxn como se vio en el teorico
    def max_n(self, gameState: GameStateExtended, agentIndex, depth):
        # Casos base:
        if depth == 0:
            if self.unroll_type == "MC":
                return None, self.montecarlo_eval(gameState, agentIndex)
            else:
                return None, self.montecarlo_tree_search_eval(gameState, agentIndex)
        elif gameState.isEnd():
            if self.use_eval_in_end:
                return None, self.evaluation_function(gameState, agentIndex)
            else:
                return None, gameState.get_rewards()

        # Llamada recursiva
        legalActions = gameState.getLegalActions(agentIndex)
        random.shuffle(legalActions)
        nextAgent = self.get_next_agent_index(agentIndex, gameState) 
        nextStatesValues = []
        for action in legalActions: # por cada accion en las acciones legales
            state = gameState.deepCopy()
            nextState = state.generateSuccessor(agentIndex, action)            

            # De la llamada recursiva solo interesa los scores y no la accion
            _, scores = self.max_n(nextState, nextAgent, depth-1) # llamada recursiva a maxn
            
            nextStatesValues.append([action, scores]) # se guarda la accion --> [values]

        # Se elije la mejor accion y el value del juego asociado
        best_action, best_score_array = self.get_best_action_score(agentIndex, nextStatesValues)
        
        return best_action, best_score_array
    
    # Función que recibe el agente y el mapa accion --> [values]
        # Para cada agente, evalua en base de los values en cada accion cual es la mejor
            # Se toma como mejor accion la que tenga un value mas grande para ese agente
    def get_best_action_score(self, agent, values):
        aux_val = float('-inf')
        aux_index = 0
        for i in range(len(values)):
            if (values[i][1][agent] > aux_val):
                aux_val = values[i][1][agent]
                aux_index = i
        return values[aux_index][0], values[aux_index][1]

    # Función encargada de Monte Carlo
    def montecarlo_eval(self, gameState, agentIndex):
        suma = np.zeros(gameState.getNumAgents())
        for i in range(self.number_of_unrolls):
            unroll_array = self.random_unroll(gameState, agentIndex)
            
            for j in range(len(unroll_array)):
                suma[j] += unroll_array[j]
        
        for j in range(len(suma)):
            suma[j] = suma[j] / self.number_of_unrolls 

        return suma 

    # Función encargada del random_unroll
    def random_unroll(self, gameState: GameStateExtended, agentIndex):
        """
            Parámetros: estado del juego y número de agente
            Retorna: valor del estado luego de realizar un unroll aleatorio
        """
        state = gameState
        V = np.zeros(state.getNumAgents())
        unroll_number = self.max_unroll_depth
        player = agentIndex
        while (not state.isEnd()) & (unroll_number > 0):
            unroll_number -= 1
            actions = state.getLegalActions(player)
            random_action = random.choice(actions)
            state = state.generateSuccessor(player, random_action)

            if state.isEnd():
                V = state.get_rewards()
            elif unroll_number <= 0:
                V = self.evaluation_function(state, player)

            player = self.get_next_agent_index(player, state)
        return V

    # Función encargada del Monte Carlo Tree Search
    def montecarlo_tree_search_eval(self, gameState, agentIndex):
       
        root = mcts_util.MCTSNode(
            parent=None, action=None, player=agentIndex, numberOfAgents=gameState.getNumAgents())
        
        for _ in range(self.number_of_unrolls):
            node = root
            state = gameState.deepCopy()
            # 4 pasos de MCTS: selection, expansion, simulation, backup
            node, state = self.selection_stage(node, state)
            node, state = self.expansion_stage(node, state)
            values = self.random_unroll(state, node.player)
            self.back_prop_stage(node, values)

        return root.value

    def selection_stage(self, node, gameState):
        while node.children:
            if(node.explored_children < len(node.children)):
                child_to_visit = node.children[node.explored_children]
                node.explored_children +=1
                node = child_to_visit
            else:
                node = max(node.children, key = ucb)
                
            gameState = gameState.generateSuccessor(node.parent.player, node.action)
        return node, gameState

    def expansion_stage(self, node, gameState):
        if not gameState.isEnd():
            node.children = [
                mcts_util.MCTSNode(
                    parent=node,
                    action=action,
                    player=self.get_next_agent_index(node.player, gameState),
                    numberOfAgents=gameState.getNumAgents()
                ) for action in gameState.getLegalActions(node.player) 
            ]
            random.shuffle(node.children)
        return node, gameState

    def back_prop_stage(self, node, value):
        while node:
            node.visits += 1
            node.value = node.value + value
            node = node.parent

