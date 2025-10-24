EXERCISE-1:

from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    traversal = []

    while queue:
        node = queue.popleft()
        traversal.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return traversal

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['E'],
    'D': [],
    'E': [],
    'F': []  # disconnected
}

print("BFS Traversal:", bfs(graph, 'A'))
------------------------------------------------------------------------------------------------

EXERCISE-2:

# Recursive DFS
def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    print(node, end=" ")
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

# Iterative DFS
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            print(node, end=" ")
            # Push neighbors in reverse order to visit them left-to-right
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

# Example graph
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print("Recursive DFS:")
dfs_recursive(graph, 'A')  # Output: A B D E F C

print("\nIterative DFS:")
dfs_iterative(graph, 'A')  # Output: A B D E F C
---------------------------------------------------------------------------------------------------------

EXERCISE-3:

import random

def print_board(board):
    print("\n")
    print(f" {board[0]} | {board[1]} | {board[2]} ")
    print("-----------")
    print(f" {board[3]} | {board[4]} | {board[5]} ")
    print("-----------")
    print(f" {board[6]} | {board[7]} | {board[8]} ")
    print("\n")

def check_winner(board):
    # Check all possible winning combinations
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
        [0, 4, 8], [2, 4, 6]             # diagonals
    ]
    
    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != " ":
            return board[combo[0]]  # returns the winning player (X or O)
    
    if " " not in board:
        return "Tie"
    
    return None

def player_move(board):
    while True:
        try:
            move = int(input("Enter your move (1-9): ")) - 1
            if 0 <= move <= 8 and board[move] == " ":
                return move
            else:
                print("Invalid move. Try again.")
        except ValueError:
            print("Please enter a number between 1 and 9.")

def computer_move(board):
    # Simple AI: first checks for winning move, then blocks player, then random
    empty_spots = [i for i, spot in enumerate(board) if spot == " "]
    
    # Check for winning move
    for spot in empty_spots:
        board_copy = board.copy()
        board_copy[spot] = "O"
        if check_winner(board_copy) == "O":
            return spot
    
    # Block player's winning move
    for spot in empty_spots:
        board_copy = board.copy()
        board_copy[spot] = "X"
        if check_winner(board_copy) == "X":
            return spot
    
    # Choose center if available
    if 4 in empty_spots:
        return 4
    
    # Choose a corner if available
    corners = [0, 2, 6, 8]
    available_corners = [c for c in corners if c in empty_spots]
    if available_corners:
        return random.choice(available_corners)
    
    # Choose a random spot
    return random.choice(empty_spots)

def play_game():
    board = [" "] * 9
    current_player = "X"  # Player is X, computer is O
    
    print("Welcome to Tic-Tac-Toe!")
    print("Enter numbers 1-9 to make your move:")
    print_board(["1", "2", "3", "4", "5", "6", "7", "8", "9"])
    
    while True:
        if current_player == "X":
            move = player_move(board)
            board[move] = "X"
        else:
            print("Computer's turn...")
            move = computer_move(board)
            board[move] = "O"
            print(f"Computer chooses position {move + 1}")
        
        print_board(board)
        result = check_winner(board)
        
        if result:
            if result == "Tie":
                print("It's a tie!")
            elif result == "X":
                print("Congratulations! You win!")
            else:
                print("Computer wins!")
            break
        
        current_player = "O" if current_player == "X" else "X"

if __name__ == "__main__":
    while True:
        play_game()
        play_again = input("Play again? (y/n): ").lower()
        if play_again != 'y':
            print("Thanks for playing!")
            break
------------------------------------------------------------------------------------------------------------------

EXERCISE-4:

class Solution:
    def solve(self, board):
        flatten = tuple(sum(board, []))
        state_dict = {flatten: 0}
        goal = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        if flatten == goal:
            return 0
        return self.get_paths(state_dict, goal)
    def get_paths(self, state_dict, goal):
        cnt = 0
        while True:
            current_nodes = [x for x in state_dict if state_dict[x] == cnt]
            if len(current_nodes) == 0:
                return -1  
            for node in current_nodes:
                next_moves = self.find_next(node)
                for move in next_moves:
                    if move not in state_dict:
                        state_dict[move] = cnt + 1
                        if move == goal:
                            return cnt + 1
            cnt += 1
    def find_next(self, node):
        moves = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4, 6],
            4: [1, 3, 5, 7],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7],
        }
        results = []
        pos_0 = node.index(0)
        for move in moves[pos_0]:
            new_node = list(node)
            new_node[move], new_node[pos_0] = new_node[pos_0], new_node[move]
            results.append(tuple(new_node))
        return results
if __name__ == "__main__":
    ob = Solution()
    matrix = [
        [3, 1, 2],
        [4, 7, 5],
        [6, 8, 0],
    ]
    print("Minimum moves to solve puzzle:", ob.solve(matrix))
---------------------------------------------------------------------------------------------------------------

EXERCISE-5:

from collections import deque
def water_jug_bfs(jug1_capacity, jug2_capacity, target):
    start = (0, 0)
    q = deque([(start, [])])
    visited = set([start])
    while q:
        (x, y), path = q.popleft()
        if x == target or y == target:
            return path + [(x, y)]
        possible_states = [
            (jug1_capacity, y),                  
            (x, jug2_capacity),                      
            (0, y),                                  
            (x, 0),
            (max(0, x - (jug2_capacity - y)), min(jug2_capacity, y + x)),
            (min(jug1_capacity, x + y), max(0, y - (jug1_capacity - x))),
        ]
        for state in possible_states:
            if state not in visited:
                visited.add(state)
                q.append((state, path + [(x, y)]))
    return None
if __name__ == "__main__":
    jug1_capacity = 4
    jug2_capacity = 3
    target = 2
    result = water_jug_bfs(jug1_capacity, jug2_capacity, target)
    if result:
        print("Solution Found:")
        for step in result:
            print(step)
    else:
        print("No Solution Exists")
-----------------------------------------------------------------------------------------------------------

EXERCISE-6:

from itertools import permutations
def tsp_simple(graph):
    n = len(graph)
    min_dist = float('inf') 
    for path in permutations(range(1, n)):
        total = graph[0][path[0]]
        for i in range(len(path) - 1):
            total += graph[path[i]][path[i + 1]]
        total += graph[path[-1]][0]
        min_dist = min(min_dist, total)
        return min_dist
graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0],
]
print(f"Shortest Route Distance: {tsp_simple(graph)}")
-------------------------------------------------------------------------------------------------------------

EXERCISE-7:

def hanoi_simple(n, src, dest, aux):
    if n>0:
        hanoi_simple(n-1, src, aux, dest)
        print(f"Move disk {n} from {src} to {dest}")
        hanoi_simple(n-1, aux, dest, src)
for disks in [1,2,3]:
    print(f"\n {disks} disk(s):")
    hanoi_simple(disks, 'A','C','B')
----------------------------------------------------------------------------------------

EXERCISE-8:

class Monkey:
    def __init__(self):
        self.height=0
        self.position=None
        self.has_banana=False
class World:
    def __init__(self):
        self.positions=["A","B","C"]
        self.monkey=Monkey()
        self.box_position="B"
        self.tree_position="C"
    def solve(self):
        print("1. Monkey moves to position B")
        self.monkey.position=="B"
        print("2. Monkey pushes box to position C")
        self.monkey.position="C"
        self.box_position="C"
        print("3. Monkey climbs the box")
        self.monkey.height=2
        print("4. Monkey takes the banana!")
        self.monkey.has_banana=True
        return "Success! Monket got the banana"
world=World()
world.monkey.position="A"
result=world.solve()
print(f"\n Result: {result}")
--------------------------------------------------------------------------------------------------------

EXERCISE-9:

import math
import random
class AlphaBetaPruning:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.nodes_evaluated = 0
    def minimax(self, depth, node_index, maximizing_player, values, alpha, beta):
        self.nodes_evaluated += 1
        if depth == self.max_depth:
            return values[node_index]
        if maximizing_player:
            max_eval = -math.inf
            for i in range(2):  
                eval_val = self.minimax(depth + 1, node_index * 2 + i, False, values, alpha, beta)
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break  
            return max_eval
            min_eval = math.inf
            for i in range(2):
                eval_val = self.minimax(depth + 1, node_index * 2 + i, True, values, alpha, beta)
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break  
            return min_eval
    def basic_example():
        abp = AlphaBetaPruning(max_depth=3)
        values = [3, 5, 6, 9, 1, 2, 0, -1]
        print("Game Tree Values:", values)
        print("Max Depth:", abp.max_depth)
        result = abp.minimax(0, 0, True, values, -math.inf, math.inf)
        print(f"Optimal Value: {result}")
        print(f"Nodes Evaluated: {abp.nodes_evaluated}")
if __name__ == "__main__":
    AlphaBetaPruning.basic_example()
-------------------------------------------------------------------------------------------------------------------------------

EXERCISE-10:

def solve_n_queens(n=8):
    def is_safe(board, row, col):
        # Check this row on the left side
        for j in range(col):
            if board[row][j] == 1:
                return False
        i, j = row, col
        while i >= 0 and j >= 0:
            if board[i][j] == 1:
                return False
            i -= 1
            j -= 1
        i, j = row, col
        while i < n and j >= 0:
            if board[i][j] == 1:
                return False
            i += 1
            j -= 1
        return True
    def solve(col, board, solutions):
        if col >= n:
            solutions.append([row[:] for row in board])
            return
        for row in range(n):
            if is_safe(board, row, col):
                board[row][col] = 1
                solve(col + 1, board, solutions)
                board[row][col] = 0  
    board = [[0] * n for _ in range(n)]
    solutions = []
    solve(0, board, solutions)
    return solutions
def print_solution(solution):
    for row in solution:
        print(' '.join('Q' if cell == 1 else '.' for cell in row))
    print()
solutions = solve_n_queens(8)
print(f"Number of solutions for 8 queens: {len(solutions)}")
print("First solution:\n")
print_solution(solutions[0])
---------------------------------------------------------------------------------------------------------------
