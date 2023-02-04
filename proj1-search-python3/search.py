# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    ############################################################
    # problem = Class PositionSearchProblem in searchAgents.py #
    ############################################################

    visited = []  # keeps track of visited nodes
    fringe = util.Stack()

    # Prepare the start node
    parent = None
    start_position = problem.getStartState()
    start_path = []
    start_cost = 0

    node = {
        "path": start_path,
        "vertex": start_position,
        "cost": start_cost
    }

    # Push the start node onto the fringe
    fringe.push(node)

    while not fringe.isEmpty():
        current_node = fringe.pop()
        if problem.isGoalState(current_node["vertex"]):
            # This is a goal state, return the path.
            return current_node["path"]
        if current_node["vertex"] not in visited:
            visited.append(current_node["vertex"])
            successors = problem.getSuccessors(current_node["vertex"])
            for child in successors:
                child_node = {
                        "path": current_node["path"].copy() + [child[1]],
                        "vertex": child[0],
                        "cost": current_node["cost"] + child[2]
                    }
                if child_node["vertex"] not in visited:
                    fringe.push(child_node)
    print("No solution found")

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    ############################################################
    # problem = Class PositionSearchProblem in searchAgents.py #
    ############################################################

    visited = []  # keeps track of visited nodes
    fringe = util.Queue()

    # Prepare the start node
    parent = None
    start_position = problem.getStartState()
    start_path = []
    start_cost = 0

    node = {
        "path": start_path,
        "vertex": start_position,
        "cost": start_cost
    }

    # Push the start node onto the fringe
    fringe.push(node)

    while not fringe.isEmpty():
        current_node = fringe.pop()
        if problem.isGoalState(current_node["vertex"]):
            # This is a goal state, return the path.
            return current_node["path"]
        if current_node["vertex"] not in visited:
            visited.append(current_node["vertex"])
            successors = problem.getSuccessors(current_node["vertex"])
            for child in successors:
                child_node = {
                        "path": current_node["path"].copy() + [child[1]],
                        "vertex": child[0],
                        "cost": current_node["cost"] + child[2]
                    }
                if child_node["vertex"] not in visited:
                    fringe.push(child_node)
    print("No solution found")

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    ############################################################
    # problem = Class PositionSearchProblem in searchAgents.py #
    ############################################################

    visited = []  # keeps track of visited nodes
    fringe = util.PriorityQueue()

    # Prepare the start node
    parent = None
    start_position = problem.getStartState()
    start_path = []
    start_cost = 0

    node = {
        "path": start_path,
        "vertex": start_position,
        "cost": start_cost
    }

    # Push the start node onto the fringe
    fringe.push(node, node["cost"])

    while not fringe.isEmpty():
        current_node = fringe.pop()
        if problem.isGoalState(current_node["vertex"]):
            # This is a goal state, return the path.
            return current_node["path"]
        if current_node["vertex"] not in visited:
            visited.append(current_node["vertex"])
            successors = problem.getSuccessors(current_node["vertex"])
            for child in successors:
                child_node = {
                        "path": current_node["path"].copy() + [child[1]],
                        "vertex": child[0],
                        "cost": current_node["cost"] + child[2]
                    }
                if child_node["vertex"] not in visited:
                    fringe.push(child_node, child_node["cost"])
    print("No solution found")

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    ############################################################
    # problem = Class PositionSearchProblem in searchAgents.py #
    ############################################################

    visited = []  # keeps track of visited nodes
    fringe = util.PriorityQueue()

    # Prepare the start node
    parent = None
    start_position = problem.getStartState()
    start_path = []
    start_cost = 0

    node = {
        "path": start_path,
        "vertex": start_position,
        "cost": start_cost
    }

    # Push the start node onto the fringe
    fringe.push(node, heuristic(node["vertex"], problem) + node["cost"])

    while not fringe.isEmpty():
        current_node = fringe.pop()
        if problem.isGoalState(current_node["vertex"]):
            # This is a goal state, return the path.
            return current_node["path"]
        if current_node["vertex"] not in visited:
            visited.append(current_node["vertex"])
            successors = problem.getSuccessors(current_node["vertex"])
            for child in successors:
                child_node = {
                        "path": current_node["path"].copy() + [child[1]],
                        "vertex": child[0],
                        "cost": current_node["cost"] + child[2]
                    }
                if child_node["vertex"] not in visited:
                    fringe.push(child_node, heuristic(child_node["vertex"], problem) + child_node["cost"])
    print("No solution found")


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
