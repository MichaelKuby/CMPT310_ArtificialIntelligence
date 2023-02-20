# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from operator import itemgetter


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)  # the SuccessorGameState
        newPos = successorGameState.getPacmanPosition()  # The new position
        newFood = successorGameState.getFood()  # Map of the food in the new position
        newGhostStates = successorGameState.getGhostStates()  # Ghost states in the new position
        newScaredTimes = [ghostState.scaredTimer for ghostState in
                          newGhostStates]  # ScaredTimees of ghosts in new position

        x, y = newPos

        # Check if the legal moves are near food
        food_score = float("inf")
        for food in newFood.asList():
            fx, fy = food
            food_score = min(food_score, (abs(fx - x)) + (abs(fy - y)))

        ghost_score = float("-inf")
        for ghost in newGhostStates:
            # Check if the ghosts are scared
            for time in newScaredTimes:
                if time != 0:
                    ghost_score = 100
                else:
                    gx, gy = ghost.getPosition()
                    ghost_score = max(ghost_score, abs(gx - x)) + abs((gy - y))

        return 25 * successorGameState.getScore() + 50 / food_score + 2 * ghost_score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        evalFunc = self.evaluationFunction
        depth = self.depth
        pacman = 0

        res = minimax(evalFunc, depth, pacman, gameState)
        return res[1]

def minimax(evalFunc, depth, agentIndex, gameState):
    if gameState.isWin() or gameState.isLose() or depth == 0:
        value, action = evalFunc(gameState), Directions.STOP
        return value, action

    nextAgent = agentIndex + 1
    if gameState.getNumAgents() == nextAgent:
        nextAgent = 0  # Max player (pacman)
        depth -= 1  # We have gone through an iteration

    # For each possible action, return all possible successor states
    successorStates = []
    legalActions = gameState.getLegalActions(agentIndex)
    for action in legalActions:
        successorStates.append((gameState.generateSuccessor(agentIndex, action), action))

    if agentIndex == 0:
        # max player
        return max_value(evalFunc, depth, nextAgent, successorStates)
    else:
        # min player
        return min_value(evalFunc, depth, nextAgent, successorStates)

def max_value(evalFunc, depth, agentIndex, successorStates):
    successorValues = []
    for state in successorStates:
        nextState = state[0]
        nextAction = state[1]
        successorValues.append((minimax(evalFunc, depth, agentIndex, nextState), nextAction))

    max = None
    for value in successorValues:
        if max == None or max < value[0][0]:
            max = value[0][0]
            direction = value[1]
    return (max, direction)

def min_value(evalFunc, depth, agentIndex, successorStates):
    successorValues = []
    for state in successorStates:
        nextState = state[0]
        nextAction = state[1]
        successorValues.append((minimax(evalFunc, depth, agentIndex, nextState), nextAction))

    min = None
    direction = None
    for value in successorValues:
        if min == None or min > value[0][0]:
            min = value[0][0]
            direction = value[1]
    return (min, direction)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        evalFunc = self.evaluationFunction
        depth = self.depth
        pacman = 0
        alpha = float('-inf')
        beta = float('inf')
        res = alphaBetaPruning(evalFunc, depth, alpha, beta, pacman, gameState)
        return res[1]

def alphaBetaPruning(evalFunc, depth, alpha, beta, agentIndex, gameState):
    if gameState.isWin() or gameState.isLose() or depth == 0:
        value, action = evalFunc(gameState), Directions.STOP
        return value, action

    nextAgent = agentIndex + 1
    if gameState.getNumAgents() == nextAgent:
        nextAgent = 0  # Max player (pacman)
        depth -= 1  # We have gone through a ply of turns

    # Get all legal actions for the current agent
    legalActions = gameState.getLegalActions(agentIndex)

    if agentIndex == 0:
        # max player
        maximum = None
        direction = None
        # iterative over all possible actions
        for action in legalActions:
            next_state = gameState.generateSuccessor(agentIndex, action)
            next_state_value = alphaBetaPruning(evalFunc, depth, alpha, beta, nextAgent, next_state)
            # update the max value and direction
            if maximum == None or maximum < next_state_value[0]:
                maximum = next_state_value[0]
                direction = action

            if maximum > beta:  # this is the cutoff
                break

            alpha = max(alpha, maximum)  # update alpha?
        return(maximum, direction, alpha, beta)
    else:
        # min player
        minimum = None
        direction = None
        # iterative over all possible actions
        for action in legalActions:
            next_state = gameState.generateSuccessor(agentIndex, action)
            next_state_value = alphaBetaPruning(evalFunc, depth, alpha, beta, nextAgent, next_state)
            # update the min value and direction
            if minimum == None or minimum > next_state_value[0]:
                minimum = next_state_value[0]
                direction = action

            if minimum < alpha:  # this is the cutoff
                break

            beta = min(beta, minimum)
        return(minimum, direction, alpha, beta)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        evalFunc = self.evaluationFunction
        depth = self.depth
        pacman = 0

        res = expectimax(evalFunc, depth, pacman, gameState)
        return res[1]

def expectimax(evalFunc, depth, agentIndex, gameState):
    if depth == 0 or gameState.isWin() or gameState.isLose():
        val, action = evalFunc(gameState), Directions.STOP
        return val, action

    nextAgent = agentIndex + 1

    if nextAgent == gameState.getNumAgents():
        nextAgent = 0
        depth -= 1

    legalActions = gameState.getLegalActions(agentIndex)

    if agentIndex == 0:
        # Max player
        maximum = None
        direction = None
        for action in legalActions:
            next_state = gameState.generateSuccessor(agentIndex, action)
            next_state_value = expectimax(evalFunc, depth, nextAgent, next_state)
            if maximum == None or next_state_value[0] > maximum:
                maximum = next_state_value[0]
                direction = action
        return maximum, direction
    else:
        # Chance node!
        value = 0
        direction = None
        p = 1 / len(legalActions)
        for action in legalActions:
            next_state = gameState.generateSuccessor(agentIndex, action)
            next_state_value = expectimax(evalFunc, depth, nextAgent, next_state)
            value += p * next_state_value[0]
        return value, direction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Four functions were used:
    1) Score of the current game state
        This simplpy uses the inbuilt .getScore() function from the
        current game state as a baseline evaluation.
    2) Proximity to food
        From the food list we look for the nearest dot. The closer it is, the better.
        Since close dots are more valuable, score this as the inverse of the distance.
    3) Proximity to a power pellet
        Checks first to see if the ghosts are scared. If false, hunting down a power
        pellet becomes very valuable.
    4) Ghost hunting
        Checks first to see if the ghosts are scared. If true, locations closer to
        ghosts become very valuable.
    """

    # Useful information you can extract from a GameState (pacman.py)
    pacman_position = currentGameState.getPacmanPosition()  # The new position
    food_list = currentGameState.getFood().asList()  # Map of the food in the new position
    ghostStates = currentGameState.getGhostStates()  # Ghost states in the new position
    newScaredTimes = [ghostState.scaredTimer for ghostState in
                      ghostStates]  # ScaredTimees of ghosts in new position
    ghost_positions = currentGameState.getGhostPositions()
    power_pellet_positions = currentGameState.getCapsules()

    # Feature 1: Score of the current game state
    game_state_score = currentGameState.getScore()

    # Feature 2: Proximity to food
    food_score = 0
    if food_list:
        closest_food = min(food_list, key=lambda x: manhattanDistance(x, pacman_position))
        food_score += 1 / (manhattanDistance(pacman_position, closest_food))

    # Feature 3: Proximity to a power-pellet
    power_pellet_score = 0
    if max(newScaredTimes) == 0:
        if power_pellet_positions:
            closest_pellet = min(power_pellet_positions, key=lambda x: manhattanDistance(pacman_position, x))
            power_pellet_score += 1 / manhattanDistance(closest_pellet, pacman_position)


    # Feature 4: Ghost-hunting
    ghost_hunter_score = 0
    if min(newScaredTimes) > 0:
        closest_ghost = min(ghost_positions, key=lambda x: manhattanDistance(x, pacman_position))
        ghost_hunter_score += 1 / (manhattanDistance(pacman_position, closest_ghost))

    return 5 * game_state_score + 3 * food_score + 200 * ghost_hunter_score + 225 * power_pellet_score


# Abbreviation
better = betterEvaluationFunction
