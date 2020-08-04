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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # print("next Game State:", successorGameState)
        # print("new Pos:", newPos)
        # print("new Food:", len(newFood.asList()))
        # print("new GhostState:", newGhostStates)
        # for ghostState in newGhostStates:
        #     print(ghostState)
        # print("new ScaredTimer:", newScaredTimes)


        "*** YOUR CODE HERE ***"
        eval = successorGameState.getScore()
        foodDist = float("inf")
        for food in newFood:
            foodDist = min(foodDist, util.manhattanDistance(food, newPos))
        eval += 1.0/foodDist

        return eval

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, state, depth, agent):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), Directions.STOP
        if agent == 0:
            return self.maxValue(state, depth, agent)
        else:
            return self.minValue(state, depth, agent)

    def maxValue(self, state, depth, agent):
        bestScore = (-float("inf"), "null")
        actions = state.getLegalActions(agent)
        for action in actions:
            nextState = state.generateSuccessor(agent, action)
            nextAgent = (agent + 1) % state.getNumAgents()
            nextScore = (self.minimax(nextState, depth, nextAgent)[0], action)
            bestScore = max(bestScore, nextScore, key = lambda x : x[0])
        return bestScore

    def minValue(self, state, depth, agent):
        bestScore = (float("inf"), "null")
        actions = state.getLegalActions(agent)
        for action in actions:
            nextState = state.generateSuccessor(agent, action)
            nextAgent = (agent + 1) % state.getNumAgents()
            if nextAgent == 0:
                nextScore = (self.minimax(nextState, depth - 1, nextAgent)[0], action)
            else:
                nextScore = (self.minimax(nextState, depth, nextAgent)[0], action)
            bestScore = min(bestScore, nextScore, key = lambda x : x[0])
        return bestScore

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
        "*** YOUR CODE HERE ***"

        return self.minimax(gameState, self.depth, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def minimaxAB(self, state, depth, agent, alpha, beta):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), Directions.STOP
        if agent == 0:
            return self.maxValue(state, depth, agent, alpha, beta)
        else:
            return self.minValue(state, depth, agent, alpha, beta)

    def maxValue(self, state, depth, agent, alpha, beta):
        bestScore = (-float("inf"), "null")
        actions = state.getLegalActions(agent)
        for action in actions:
            nextState = state.generateSuccessor(agent, action)
            nextAgent = (agent + 1) % state.getNumAgents()
            nextScore = (self.minimaxAB(nextState, depth, nextAgent, alpha, beta)[0], action)
            bestScore = max(bestScore, nextScore, key = lambda x : x[0])
            if (bestScore[0] > beta):
                return bestScore
            alpha = max(alpha, bestScore[0])
        return bestScore

    def minValue(self, state, depth, agent, alpha, beta):
        bestScore = (float("inf"), "null")
        actions = state.getLegalActions(agent)
        for action in actions:
            nextState = state.generateSuccessor(agent, action)
            nextAgent = (agent + 1) % state.getNumAgents()
            if nextAgent == 0:
                nextScore = (self.minimaxAB(nextState, depth - 1, nextAgent, alpha, beta)[0], action)
            else:
                nextScore = (self.minimaxAB(nextState, depth, nextAgent, alpha, beta)[0], action)
            bestScore = min(bestScore, nextScore, key = lambda x : x[0])
            if (bestScore[0] < alpha):
                return bestScore
            beta = min(beta, bestScore[0])
        return bestScore

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimaxAB(gameState, self.depth, 0, -float("inf"), float("inf"))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expMax(self, state, depth, agent):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), Directions.STOP
        if agent == 0:
            return self.maxValue(state, depth, agent)
        else:
            return self.expValue(state, depth, agent)

    def maxValue(self, state, depth, agent):
        bestScore = (-float("inf"), "null")
        actions = state.getLegalActions(agent)
        for action in actions:
            nextState = state.generateSuccessor(agent, action)
            nextAgent = (agent + 1) % state.getNumAgents()
            nextScore = (self.expMax(nextState, depth, nextAgent)[0], action)
            bestScore = max(bestScore, nextScore, key = lambda x : x[0])
        return bestScore

    def expValue(self, state, depth, agent):
        score = 0
        actions = state.getLegalActions(agent)
        p = 1.0/len(actions)
        for action in actions:
            nextState = state.generateSuccessor(agent, action)
            nextAgent = (agent + 1) % state.getNumAgents()
            if nextAgent == 0:
                nextScore = self.expMax(nextState, depth-1, nextAgent)[0]
            else:
                nextScore = self.expMax(nextState, depth, nextAgent)[0]
            score += p * nextScore
        return (score, actions[0])

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expMax(gameState, self.depth, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I used the same evaluation function as in problem 1, but with the current game state.
    The evaluation function is given by the distance to the closest food pellet, or 0 if there are no pellets.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    eval = currentGameState.getScore()
    foodDist = float("inf")
    for food in newFood:
        foodDist = min(foodDist, util.manhattanDistance(food, newPos))
    eval += 1.0/foodDist

    return eval

# Abbreviation
better = betterEvaluationFunction
