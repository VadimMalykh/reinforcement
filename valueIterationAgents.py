# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util, copy

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
        self.computeValues()

    def computeValues(self):
        states = self.mdp.getStates()
        i = 0
        while i < self.iterations:
          kValues = copy.deepcopy(self.values)
          for state in states:
            if not self.mdp.isTerminal(state):
              bestAction = self.getAction(state)
              bestValue = self.getQValue(state, bestAction)
              if bestAction == 'exit':
                kValues[state] = bestValue
              else:
                kValues[state] = self.getValue(state) + self.discount * bestValue
              kValues[state] = bestValue
          self.values = kValues
          i += 1

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
        Q = 0
        if not self.mdp.isTerminal(state):
          if action == 'exit':
            Q = self.mdp.getReward(state, action, state)
          else:
            transitions = self.mdp.getTransitionStatesAndProbs(state, action)
            for transition in transitions:
              newState, probability = transition
              Q += probability * (self.mdp.getReward(state, action, newState) + self.discount * self.getValue(newState))
        return Q

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
        possibleActions = self.mdp.getPossibleActions(state)
        values = [self.getQValue(state, action) for action in possibleActions]
        maxValue = max(values)
        for index in range(len(values)):
          if values[index] == maxValue:
            return possibleActions[index]
        return None

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

    def getStateFromAction(self, state, action):
        x,y = state
        if (action == "north"):
          return (x, y+1)
        if (action == "south"):
          return (x, y-1)
        if (action == "west"):
          return (x-1, y)
        if (action == "east"):
          return (x+1, y)
        if (action == "exit"):
          return (x,y)
        raise Exception("Unknown action:" + action)