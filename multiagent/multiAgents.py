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
import searchAgents

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        # raw_input('Press enter to continue')

        try:
            closestFoodDistance = len(
                searchAgents
                    .ClosestDotSearchAgent()
                        .findPathToClosestDot(
                            successorGameState
                        )
            )
        except:
            closestFoodDistance = 0

        foodModifier = -closestFoodDistance
        eatModifier = 100 if len(
            currentGameState.getFood().asList()
        ) != len(
            successorGameState.getFood().asList()
        ) else 0

        try:
            closestGhostDistance = min([
                searchAgents
                    .mazeDistance(
                        newPos,
                        tuple(map(int, ghost.configuration.pos)),
                    successorGameState,
                )
                for ghost in newGhostStates
            ])
        except:
            closestGhostDistance = 0

        ghostDistanceScore = {
            0: -10000,
            1: -1000,
        }


        ghostModifier = ghostDistanceScore.get(closestGhostDistance, 5)

        "*** YOUR CODE HERE ***"
        return foodModifier + ghostModifier + successorGameState.getScore() + eatModifier

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
        """
        "*** YOUR CODE HERE ***"

        def value(state, agentId, remainingCalls):
            agentId = agentId % state.getNumAgents()
            if state.isWin() or state.isLose() or remainingCalls == 0:
                return self.evaluationFunction(state)
            if agentId == 0:
                return maxValue(state, agentId, remainingCalls)
            else:
                return minValue(state, agentId, remainingCalls)

        def maxValue(state, agentId, remainingCalls):
            maximum = float('-inf')
            for action in state.getLegalActions(agentId):
                maximum_in_branch = value(
                    state.generateSuccessor(agentId, action),
                    (agentId + 1) % state.getNumAgents(),
                    remainingCalls - 1,   
                )
                if maximum_in_branch > maximum:
                    maximum = maximum_in_branch
            return maximum

        def minValue(state, agentId, remainingCalls):
            minimum = float('inf')
            for action in state.getLegalActions(agentId):
                minimum_in_branch = value(
                    state.generateSuccessor(agentId, action),
                    (agentId + 1) % state.getNumAgents(),
                    remainingCalls - 1,   
                )
                if minimum_in_branch < minimum:
                    minimum = minimum_in_branch
            return minimum

        successors = [(action, gameState.generateSuccessor(0, action)) for action in gameState.getLegalActions(0)]
        successors_values = list(map(
            lambda state: value(state, 1, self.depth * state.getNumAgents() - 1),
            [successor[1] for successor in successors]
        ))
        idx_of_max = successors_values.index(max(successors_values))
        maxValueOfState = [successor[0] for successor in successors][idx_of_max]
        return maxValueOfState
            
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def value(state, agentId, remainingCalls, prunings):
            agentId = agentId % state.getNumAgents()
            if state.isWin() or state.isLose() or remainingCalls == 0:
                return self.evaluationFunction(state)
            if agentId == 0:
                return maxValue(state, agentId, remainingCalls, prunings)
            else:
                return minValue(state, agentId, remainingCalls, prunings)

        def maxValue(state, agentId, remainingCalls, prunings):
            alpha, beta = prunings
            maximum = float('-inf')
            for action in state.getLegalActions(agentId):
                maximum_in_branch = value(
                    state.generateSuccessor(agentId, action),
                    (agentId + 1) % state.getNumAgents(),
                    remainingCalls - 1,
                    (alpha, beta),
                )
                if maximum_in_branch > maximum:
                    maximum = maximum_in_branch
                if maximum > beta:
                    return maximum
                alpha = max(alpha, maximum)
            return maximum

        def minValue(state, agentId, remainingCalls, prunings):
            alpha, beta = prunings
            minimum = float('inf')
            for action in state.getLegalActions(agentId):
                minimum_in_branch = value(
                    state.generateSuccessor(agentId, action),
                    (agentId + 1) % state.getNumAgents(),
                    remainingCalls - 1,
                    (alpha, beta),
                )
                if minimum_in_branch < minimum:
                    minimum = minimum_in_branch
                if minimum < alpha:
                    return minimum
                beta = min(beta, minimum)
            return minimum

        successors_values = []
        successors = [(action, gameState.generateSuccessor(0, action)) for action in gameState.getLegalActions(0)]
        alpha, beta = float('-inf'), float('inf')
        for successor in successors:
            successorValue = value(
                successor[1],
                1,
                self.depth * gameState.getNumAgents() - 1,
                (alpha, beta)
            )
            if successorValue > beta:
                return successorValue
            successors_values.append(successorValue)
            alpha = max((alpha, successorValue))
        idx_of_max = successors_values.index(max(successors_values))
        maxValueOfState = [successor[0] for successor in successors][idx_of_max]
        return maxValueOfState

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
        "*** YOUR CODE HERE ***"

        def value(state, agentId, remainingCalls):
            agentId = agentId % state.getNumAgents()
            if state.isWin() or state.isLose() or remainingCalls == 0:
                return self.evaluationFunction(state)
            if agentId == 0:
                return maxValue(state, agentId, remainingCalls)
            else:
                return expectedValue(state, agentId, remainingCalls)

        def maxValue(state, agentId, remainingCalls):
            maximum = float('-inf')
            for action in state.getLegalActions(agentId):
                maximum_in_branch = value(
                    state.generateSuccessor(agentId, action),
                    (agentId + 1) % state.getNumAgents(),
                    remainingCalls - 1,   
                )
                if maximum_in_branch > maximum:
                    maximum = maximum_in_branch
            return maximum

        def expectedValue(state, agentId, remainingCalls):
            minimum = float('inf')
            subbranchValues = []
            for action in state.getLegalActions(agentId):
                minimum_in_branch = value(
                    state.generateSuccessor(agentId, action),
                    (agentId + 1) % state.getNumAgents(),
                    remainingCalls - 1,   
                )
                subbranchValues.append(minimum_in_branch)
            if not subbranchValues:
                return minimum
            return sum(subbranchValues) / len(subbranchValues)

        successors = [(action, gameState.generateSuccessor(0, action)) for action in gameState.getLegalActions(0)]
        successors_values = list(map(
            lambda state: value(state, 1, self.depth * state.getNumAgents() - 1),
            [successor[1] for successor in successors]
        ))
        idx_of_max = successors_values.index(max(successors_values))
        maxValueOfState = [successor[0] for successor in successors][idx_of_max]
        return maxValueOfState

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    try:
        closestFoodDistance = len(
            searchAgents
                .ClosestDotSearchAgent()
                    .findPathToClosestDot(
                        currentGameState
                    )
        )
    except:
        closestFoodDistance = 0

    try:
        closestScaredGhostDistance = min([
            searchAgents
                .mazeDistance(
                    newPos,
                    tuple(map(int, ghost.configuration.pos)),
                successorGameState,
            )
            for ghost in newGhostStates if ghost.scaredTimer
        ])
    except:
        closestScaredGhostDistance = 0

    scaredGhostModifier = -closestScaredGhostDistance * 10
 
    try:
        capsuleSumDistance = sum([searchAgents.mazeDistance(
            newPos,
            tuple(map(int, capsule)),
            successorGameState,
        ) for capsule in currentGameState.getCapsules()])
    except:
        capsuleSumDistance = -5
    
    capsuleModifier = -capsuleSumDistance * 5

    foodModifier = -closestFoodDistance
    return foodModifier + currentGameState.getScore() + capsuleModifier + sum(newScaredTimes)

# Abbreviation
better = betterEvaluationFunction

