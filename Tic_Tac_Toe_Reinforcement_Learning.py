#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from random import randint
import time


# In[33]:


### This function is used for printing out the current state of a board
def print_state(state):
    print("    1   2   3  ")
    for i in range(3):
        print("  -------------")
        print(str(i+1) + " | " + state[i][0] + " | "+ state[i][1] + " | "+ state[i][2]+ " |")
    print("  -------------")


# In[229]:


#example of printing a state
current_state = [["X", "O", "X"], ["O", "X", "O"], [" ", " ", " "]]
print_state(current_state)


# In[54]:


# This function takes a board state as an input, then returns True if the state represents a Win/Loss, False otherwise
# It checks horizontal, vertical, and diagonal rows
def check_game_over(state):
    game_over = False
    #check horizontals and verticals
    for i in range(3):
        vertical = [state[0][i], state[1][i], state[2][i]]
        horizontal = state[i]
        if horizontal == ["X", "X", "X"] or horizontal == ["O", "O", "O"] or vertical == ["X", "X", "X"] or vertical == ["O", "O", "O"]:
            game_over = True
    #check diagonals
    diag1 = [state[0][0], state[1][1], state[2][2]]
    diag2 = [state[2][0], state[1][1], state[0][2]]
    if diag1 == ["X", "X", "X"] or diag1 == ["O", "O", "O"] or diag2 == ["X", "X", "X"] or diag2 == ["O", "O", "O"]:
        game_over = True
    return game_over


# In[120]:


# This function takes a board state as an input.  It returns True if the board is full and no more moves can be played.
# Otherwise it returns False.
def check_tie(state):
    game_over = True
    for i in state:
        if " " in i:
            game_over = False
    return game_over


# In[209]:


# This function takes a policy, the current board state, and whether it is player 1 (X's) or player 2 (O's) turn to move
# It also takes "exp_rate" which is an exploration rate between 0 and 100.  For example, if the exp_rate is 30, then the
# function will return a random move (i.e. exploring) 30% of the time.  The other 70% of the time it will return the optimal
# move based on the policy input.  It returns the new state resulting from the chosen move.

def choose_move(policy, current_state, player, exp_rate):
    if player == 1:
        player_icon = "X"
    else:
        player_icon = "O"
    # evaluate all possible next states
    possible_next_states = []
    #
    for i in range(3):
        for j in range(3):
            if current_state[i][j] == " ":
                possible_state = [x[:] for x in current_state] #makes copy of current state
                possible_state[i][j] = player_icon 
                possible_next_states.append(possible_state)  # makes list of all possible states for the next move
    explore_variable = randint(1, 100)
    if explore_variable < (100-exp_rate): #choose random move exp_rate% of the time.  Otherwise, choose the optimal move
        best_value = -99999
        for i in possible_next_states:      #search all possible next states for best available value
            if check_game_over(i): #if it ends the game, always choose it
                new_state = i
                value = 100000
            elif str(i) in policy: #check if value has been assigned to board state
                value = policy[str(i)]
            else:
                value = 0
            if value > best_value:
                new_state = i 
                best_value = value
    else: #30% of the time choose a random move
        random_state = randint(0, len(possible_next_states)-1)
        new_state = possible_next_states[random_state]
    return new_state


# In[242]:


# This is the most important function. This is where "learning" actually takes place.  
#The inputs are the two current policies used by the opposing players, and the three states that made up the last turn cycle (i.e. the last move made by each player)
# Based on the results of the last move by each player, the policies for each player are updated.

def update_policy(policy1, policy2, last_state, middle_state, new_state): #updates policies assuming policy1 made the most recent move
    game_over = check_game_over(new_state)
    board_full = check_tie(new_state)
    if game_over:
        #update policy1 for last_state
        if str(last_state) in policy1:
            policy1[str(last_state)] = policy1[str(last_state)] +.1*(10 - policy1[str(last_state)])
        else:
            policy1[str(last_state)] = 1
        
        #update policy2 for middle_state
        if str(middle_state) in policy2:
            policy2[str(middle_state)] = policy2[str(middle_state)] +.1*(-10 - policy2[str(middle_state)])
        else:
            policy2[str(middle_state)] = -1
    else:
        if board_full:
            policy1[str(new_state)] = -5 #penalize ties half as much as losses
        else:
            if str(new_state) not in policy1:
                policy1[str(new_state)] = 0
            
        #update policy1 for last_state
        if str(last_state) in policy1:
            policy1[str(last_state)] = policy1[str(last_state)] +.1*(policy1[str(new_state)] - policy1[str(last_state)])
        else:
            policy1[str(last_state)] = .1*policy1[str(new_state)]
        
    
    return policy1, policy2, game_over, board_full


# In[211]:


# This functions simulates a single turn (i.e. both players move once if possible)
def train_turn(policy1, policy2, current_state, middle_state): #simulates one turn of tic-tac toe, assumes policy1 will go first
    new_middle_state = choose_move(policy1, current_state, 1, 30) #returns state of board after player goes
    policy1, policy2, game_over, board_full = update_policy(policy1, policy2, middle_state, current_state, new_middle_state)
    if game_over or board_full:
        end_state = new_middle_state
    else:
        end_state = choose_move(policy2, new_middle_state, 2, 30)
        policy2, policy1, game_over, board_full = update_policy(policy2, policy1, current_state, new_middle_state, end_state)
    
    return policy1, policy2, end_state, new_middle_state, game_over, board_full


# In[233]:


# This function trains two agents to play against each other 
# Policy 1 will be the policy for an agent who picks first, Policy 2 will be policy for agent who picks second
# By default it uses an exploration rate of 30.  Number of iterations is the only input.

def train_tic_tac_toe(iterations):
    policy1 = {}
    policy2 = {}
    for i in range(iterations):
        player_turn = 1 # randomly choose who goes first
        current_state = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]] #initialize empty board
        middle_state = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]] #initialize empty middle state
        game_over = False
        board_full = False
        counter = 0
        while game_over == False and board_full == False and counter <6: #play until someone wins or board is full
                policy1, policy2, current_state, middle_state, game_over, board_full = train_turn(policy1, policy2, current_state, middle_state)
                
    return policy1, policy2
            
            
            


# In[230]:


# This function allows a human to play against a trained computer.  Two policies are input because we have separate policies
# depending on whether the computer goes first or second.

def play_tic_tac_toe(policy1, policy2):
    #--------------Initialize Game--------------------#
    first_turn = randint(1, 2) # randomly choose who goes first
    print("Starting new game of tic tac toe...")
    if first_turn == 1:
        print("Player 1: Human (X's)")
        print("Player 2: Computer (O's)")
        print("Player 1 Goes First!")
        policy = policy2
        whose_turn = "Human"
        human_marker = "X"
        computer_player = 2
    else:
        print("Player 1: Computer (X's)")
        print("Player 2: Human (O's)")
        print("Player 1 Goes First!")
        policy = policy1
        computer_player = 1
        whose_turn = "Computer"
        human_marker = "O"
    current_state = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]] #initialize empty board
    print_state(current_state)
    
    #- ---------------Play Game-------------------------#
    game_over = False    
    board_full = False
    counter = 0
    while game_over == False and board_full == False:
        if whose_turn == "Human": #human turn
            new_row = int(input("Enter Row Number of next move: "))
            new_col = int(input("Enter Col Number of next move: "))
            current_state[new_row-1][new_col-1] = human_marker
            print_state(current_state)
            whose_turn = "Computer" #toggle whose turn it is at end of turn
            game_over = check_game_over(current_state)
            if game_over:
                print("Human wins!")
        else: #computer turn
            for x in range (0,3):  
                b = "Computer is thinking" + "." * x
                print (b, end="\r")
                time.sleep(1)
            print("                            ")
            current_state = choose_move(policy, current_state, computer_player, 0)
            print_state(current_state)
                
            whose_turn = "Human" #toggle whose turn it is at end of turn
            game_over = check_game_over(current_state)
            if game_over:
                print("Computer Wins!")
        if not game_over:    
            board_full = check_tie(current_state)
            if board_full:
                print("Game ends in a tie.")
        
        counter += 1
        


# In[243]:


policy1, policy2 = train_tic_tac_toe(30000)


# In[246]:


play_tic_tac_toe(policy1, policy2)

