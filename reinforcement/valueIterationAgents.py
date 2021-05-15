# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        def memoize(function):
            memo = dict()
            def new_function(*args):
                if args in new_function.memo:
                    return memo[args]
                new_value = function(*args)
                memo[args] = new_value
                return new_value
            new_function.memo = memo
            return new_function

        # Bellman equation
        @memoize
        def value(state, iterations_left):
            if iterations_left == 0 or mdp.isTerminal(state):
                return 0
            return max(Qvalue(state, action, iterations_left) for action in mdp.getPossibleActions(state))
        
        @memoize
        def Qvalue(state, action, iterations_left):
            states_and_probs = mdp.getTransitionStatesAndProbs(state, action)
            return sum(
                prob * (mdp.getReward(state, action, new_state) + discount * value(new_state, iterations_left - 1))
                for new_state, prob in states_and_probs
            )
        # end of Bellman equation

        for state in mdp.getStates():
            self.values[state] = value(state, iterations)

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        states_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        return sum(
            prob * (self.mdp.getReward(state, action, new_state) + self.discount * self.values[new_state])
            for new_state, prob in states_and_probs
        )

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        qvalues = list(map(lambda action: self.getQValue(state, action), actions))
        max_index = qvalues.index(max(qvalues))
        return actions[max_index]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
