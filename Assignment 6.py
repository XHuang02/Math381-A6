import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Set the seed value
random.seed(122)

# Create a 3x3 grid for two players
rows = 3
cols = 3
player_board = [[0] * cols for i in range(rows)]
opponent_board = [[0] * cols for i in range(rows)]


# Roll the dice and get a number
def roll_dice():
    return random.randint(1, 6)


# When one player has filled all nine slots on their board, the game ends
def matrix_ouccupied(matrix, rows, cols):
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 0:  # Assuming 0 represents an unoccupied square
                return False  # Found an unoccupied square, return False
    return True  # All squares are occupied


# The dice can be randomly put on any square if the square is not occupied
def place_number(matrix, rows, cols, number):
    random_row = random.randint(0, rows - 1)
    random_col = random.randint(0, cols - 1)

    while (matrix[random_row][random_col] != 0):
        random_row = random.randint(0, rows - 1)
        random_col = random.randint(0, cols - 1)

    matrix[random_row][random_col] = number
    return matrix


# To get the total score of the player, which is the sum of all the dice values on the board
# Also include the situations double and triple
def get_score(matrix, cols):
    totals = []
    for column_index in range(cols):
        column = [row[column_index] for row in matrix]
        counts = {}
        for num in column:
            if num in counts:
                counts[num] += 1
            else:
                counts[num] = 1

        total = 0
        for num, count in counts.items():
            if count > 1:
                total += num * count * count
            else:
                total += num

        totals.append(total)

    return sum(totals)


# If number in columns of matrix 1 match one or more values in matrix 2 corresponding column,
# all matching numbers in matrix 2 are removed.
def match_and_zero(matrix1, matrix2, cols):
    for col in range(cols):
        column1 = [row[col] for row in matrix1]
        column2 = [row[col] for row in matrix2]
        matches = set(column1) & set(column2)

        if matches:
            for row in matrix2:
                if row[col] in matches:
                    row[col] = 0


    return matrix2



# Strategy 2: Make eliminating opponents’ dice a priority
def prior_remove(matrix1, matrix2, rows, cols, number):
    for col in range(cols): # check whether the number rolled appears in opponent's board
        column2 = [row[col] for row in matrix2]
        matches = set()
        for value in column2:
            if value == number:
                matches.add(value)

        if matches:
            for i in range(rows):
                if matrix1[i][col] == 0:  # Assuming 0 represents an unoccupied square
                    occupy = True  # Found an unoccupied square
                    break
                else:
                    occupy = False
            if occupy:
                for row in matrix2:
                    if row[col] in matches:
                        row[col] = 0
                random_row = random.randint(0, rows - 1)
                while matrix1[random_row][col] != 0:
                    random_row = random.randint(0, rows - 1)
                matrix1[random_row][col] = number
                return matrix1, matrix2

    random_row = random.randint(0, rows - 1)
    random_col = random.randint(0, cols - 1)
    while matrix1[random_row][random_col] != 0:
        random_row = random.randint(0, rows - 1)
        random_col = random.randint(0, cols - 1)
    matrix1[random_row][random_col] = number
    return matrix1, matrix2

# Stimulation of knucklebones game using Strategy 2 1000*1000
'''
games_win = []
for run in range(1000):
    player_win_count = 0
    total_games = 0
    for i in range(1000):
        player_board = [[0] * cols for i in range(rows)]
        opponent_board = [[0] * cols for i in range(rows)]
        while not matrix_ouccupied(player_board, rows, cols) and not matrix_ouccupied(opponent_board, rows, cols):
            player_dice = roll_dice()  # player roll the dice
            player_board, opponent_board = prior_remove(player_board, opponent_board, rows, cols, player_dice)  # prior removal
            opponent_dice = roll_dice()  # opponent roll the dice
            opponent_board, player_board = prior_remove(opponent_board, player_board, rows, cols,
                                                        opponent_dice)  # opponent place the dice
        player_score = get_score(player_board, cols)
        opponent_score = get_score(opponent_board, cols)
        if player_score > opponent_score:
            player_win_count += 1
        total_games += 1
    player_win_prob = player_win_count / total_games
    games_win.append(player_win_prob)
print(games_win)
'''

# Stimulation of knucklebones game using Strategy 2 10*100000

games_win = np.zeros((10,100000))
long_term_prob = np.zeros((1,10))
for run in range(10):
    player_win_count = 0
    total_games = 0
    for i in range(100000):
        player_board = [[0] * cols for i in range(rows)]
        opponent_board = [[0] * cols for i in range(rows)]
        while not matrix_ouccupied(player_board, rows, cols) and not matrix_ouccupied(opponent_board, rows, cols):
            player_dice = roll_dice()  # player roll the dice
            player_board, opponent_board = prior_remove(player_board, opponent_board, rows, cols, player_dice)  # prior removal
            opponent_dice = roll_dice()  # opponent roll the dice
            opponent_board, player_board = prior_remove(opponent_board, player_board, rows, cols,
                                                        opponent_dice)  # opponent place the dice
        player_score = get_score(player_board, cols)
        opponent_score = get_score(opponent_board, cols)
        if player_score > opponent_score:
            player_win_count += 1
        total_games += 1
        player_win_prob = player_win_count / total_games
        games_win[run, i] = player_win_prob
    long_term_prob[0,run] = games_win[run,100000-1]
    player_win_count = 0

# data for convergence plot
run_1 = games_win[0, :1000]
run_2 = games_win[1, :1000]
run_3 = games_win[2, :1000]
run_4 = games_win[3, :1000]
run_5 = games_win[4, :1000]
run_6 = games_win[5, :1000]
run_7 = games_win[6, :1000]
run_8 = games_win[7, :1000]
run_9 = games_win[8, :1000]
run_10 = games_win[9, :1000]
x = range(1000)
np.savetxt('strategy_1_output.csv', games_win, delimiter=',')

# Plot the data vectors
plt.plot(x, run_1)
plt.plot(x, run_2)
plt.plot(x, run_3)
plt.plot(x, run_4)
plt.plot(x, run_5)
plt.plot(x, run_6)
plt.plot(x, run_7)
plt.plot(x, run_8)
plt.plot(x, run_9)
plt.plot(x, run_10)

# Add labels and legend
plt.ylabel('Player winning Probability')

# Display the graph
plt.show()

'''
# Calculate the mean and standard deviation
mean = np.mean(long_term_prob)
std = np.std(long_term_prob)
n = 10

# Calculate the standard error
se = std / np.sqrt(n)

# Calculate the margin of error (99.8% CI)
moe = 3.0902 * se  # 3 is the z-value corresponding to a 99.8% confidence level for a standard normal distribution

# Calculate the lower and upper bounds of the confidence interval
lower_bound = mean - moe
upper_bound = mean + moe

# Print the confidence interval
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
'''

# code for running 1 game with Strategy 2
# while not matrix_ouccupied(player_board, rows, cols) and not matrix_ouccupied(opponent_board, rows, cols):
#     player_dice = roll_dice()  # player roll the dice
#     player_board, opponent_board = prior_remove(player_board, opponent_board, rows, cols, player_dice)  # prior removal
#     opponent_dice = roll_dice()  # opponent roll the dice
#     opponent_board, player_board = prior_remove(opponent_board, player_board, rows, cols,
#                                                 opponent_dice)  # opponent place the dice
# player_score = get_score(player_board, cols)
# opponent_score = get_score(opponent_board, cols)
# print(player_board)
# print(opponent_board)

# Strategy 3:  if we roll numbers greater or equal to 3, we put the same numbers
# in different columns. If we have numbers 1 or 2, we put them in the same column
def place_dice(matrix, rows, cols, large_number, number):
    if number >= large_number:
        # Check if any column is not fully occupied
        column_indices = [i for i in range(cols) if 0 in [row[i] for row in matrix]]
        if column_indices:
            for col_index in column_indices:
                if matrix[0][col_index] == 0:
                    matrix[0][col_index] = number
                    break
    else:
        random_row = random.randint(0, rows - 1)
        random_col = random.randint(0, cols - 1)

        while matrix[random_row][random_col] != 0:
            random_row = random.randint(0, rows - 1)
            random_col = random.randint(0, cols - 1)

        matrix[random_row][random_col] = number

    return matrix

# code for running 1 game with Strategy 3
'''
while not matrix_ouccupied(player_board, rows, cols) and not matrix_ouccupied(opponent_board, rows, cols):
    player_dice = roll_dice()  # player roll the dice
    player_board = place_dice(player_board, rows, cols, player_dice)  # player place the dice
    opponent_board = match_and_zero(player_board, opponent_board, cols)  # the opponent's dice may be removed
    opponent_dice = roll_dice()  # opponent roll the dice
    opponent_board = place_dice(opponent_board, rows, cols,opponent_dice)  # opponent place the dice
    player_board = match_and_zero(opponent_board, player_board, cols)  # the player's dice may be removed
player_score = get_score(player_board, cols)
opponent_score = get_score(opponent_board, cols)
print(player_board)
print(opponent_board)
'''

# Stimulation of knucklebones game using strategy 3 1000*1000
# games_win = []
# for run in range(1000):
#     player_win_count = 0
#     total_games = 0
#     large_number = 3
#     for i in range(1000):
#         player_board = [[0] * cols for i in range(rows)]
#         opponent_board = [[0] * cols for i in range(rows)]
#         while not matrix_ouccupied(player_board, rows, cols) and not matrix_ouccupied(opponent_board, rows, cols):
#             player_dice = roll_dice()  # player roll the dice
#             player_board = place_dice(player_board, rows, cols, large_number, player_dice)  # player place the dice
#             opponent_board = match_and_zero(player_board, opponent_board, cols)  # the opponent's dice may be removed
#             opponent_dice = roll_dice()  # opponent roll the dice
#             opponent_board = place_dice(opponent_board, rows, cols, large_number, opponent_dice)  # opponent place the dice
#             player_board = match_and_zero(opponent_board, player_board, cols)  # the player's dice may be removed
#         player_score = get_score(player_board, cols)
#         opponent_score = get_score(opponent_board, cols)
#         if player_score > opponent_score:
#             player_win_count += 1
#         total_games += 1
#     player_win_prob = player_win_count / total_games
#     games_win.append(player_win_prob)
# print(games_win)
#
# plt.hist(games_win, edgecolor='black', density=True)
# # Adding labels and title
# plt.xlabel('Probability of the player winning the game')
# plt.ylabel('Frequency')
# plt.title('Histogram')
#
# # Displaying the plot
# plt.show()

# Stimulation of 10*100000 knucklebones game using strategy 3
'''
games_win_2 = np.zeros((10,100000))
long_term_prob_2 = np.zeros((1,10))
large_number_1 = 6
large_number_2 = 6
for run in range(10):
    win_count = 0
    total_games = 0
    for i in range(100000):
        player_board = [[0] * cols for i in range(rows)]
        opponent_board = [[0] * cols for i in range(rows)]
        while not matrix_ouccupied(player_board, rows, cols) and not matrix_ouccupied(opponent_board, rows, cols):
            player_dice = roll_dice()  # player roll the dice
            player_board = place_dice(player_board, rows, cols, large_number_1, player_dice)  # player place the dice
            opponent_board = match_and_zero(player_board, opponent_board, cols)  # the opponent's dice may be removed
            opponent_dice = roll_dice()  # opponent roll the dice
            opponent_board = place_dice(opponent_board, rows, cols, large_number_2, opponent_dice)  # opponent place the dice
            player_board = match_and_zero(opponent_board, player_board, cols)  # the player's dice may be removed
        player_score = get_score(player_board, cols)
        opponent_score = get_score(opponent_board, cols)
        if player_score > opponent_score:
            win_count += 1
        total_games += 1
        player_win_prob_2 = win_count / total_games
        games_win_2[run, i] = player_win_prob_2
    long_term_prob_2[0,run] = games_win_2[run,100000-1]
    win_count = 0


# # data for convergence plot
# run_1 = games_win_2[0, :1000]
# run_2 = games_win_2[1, :1000]
# run_3 = games_win_2[2, :1000]
# run_4 = games_win_2[3, :1000]
# run_5 = games_win_2[4, :1000]
# run_6 = games_win_2[5, :1000]
# run_7 = games_win_2[6, :1000]
# run_8 = games_win_2[7, :1000]
# run_9 = games_win_2[8, :1000]
# run_10 = games_win_2[9, :1000]
# x = range(1000)
# np.savetxt('strategy_2_output_3.csv', games_win_2, delimiter=',')
#
# # Plot the data vectors
# plt.plot(x, run_1)
# plt.plot(x, run_2)
# plt.plot(x, run_3)
# plt.plot(x, run_4)
# plt.plot(x, run_5)
# plt.plot(x, run_6)
# plt.plot(x, run_7)
# plt.plot(x, run_8)
# plt.plot(x, run_9)
# plt.plot(x, run_10)
#
# # Add labels and legend
# plt.ylabel('Player winning Probability')
#
# # Display the graph
# plt.show()

# Calculate the mean and standard deviation
mean = np.mean(long_term_prob_2)
std = np.std(long_term_prob_2)
n = 10

# Calculate the standard error
se = std / np.sqrt(n)

# Calculate the margin of error (99.8% CI)
moe = 3.0902 * se  # 3 is the z-value corresponding to a 99.8% confidence level for a standard normal distribution

# Calculate the lower and upper bounds of the confidence interval
lower_bound = mean - moe
upper_bound = mean + moe

# Print the confidence interval
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
'''

# Strategy 4: remove opponent's dice first, if no remove, try to spread the large numbers
def combined(matrix1, matrix2, rows, cols, large_number, number):
    for col in range(cols): # check whether the number rolled appears in opponent's board
        column2 = [row[col] for row in matrix2]
        matches = set()
        for value in column2:
            if value == number:
                matches.add(value)

        if matches:
            for i in range(rows):
                if matrix1[i][col] == 0:  # Assuming 0 represents an unoccupied square
                    occupy = True  # Found an unoccupied square
                    break
                else:
                    occupy = False
            if occupy:
                for row in matrix2:
                    if row[col] in matches:
                        row[col] = 0
                random_row = random.randint(0, rows - 1)
                while matrix1[random_row][col] != 0:
                    random_row = random.randint(0, rows - 1)
                matrix1[random_row][col] = number
                return matrix1, matrix2
    else:
        if number >= large_number:
            # Check if any column is not fully occupied
            column_indices = [i for i in range(cols) if 0 in [row[i] for row in matrix1]]
            if column_indices:
                for col_index in column_indices:
                    if matrix1[0][col_index] == 0:
                        matrix1[0][col_index] = number
                        break
        else:
            random_row = random.randint(0, rows - 1)
            random_col = random.randint(0, cols - 1)

            while matrix1[random_row][random_col] != 0:
                random_row = random.randint(0, rows - 1)
                random_col = random.randint(0, cols - 1)

            matrix1[random_row][random_col] = number

    return matrix1, matrix2

# check whether it works
'''
matrix1 = [[1,0,4],
           [1,6,0],
           [1,0,0]]
matrix2 = [[0,0,5],
           [2,0,0],
           [3,1,6]]
rows = 3
cols = 3
large_number = 3
matrix1, matrix2 = combined(matrix1, matrix2, rows, cols, large_number, 2)
print(matrix1)
print(matrix2)
'''
# S3 1000*1000
'''
games_win = []
for run in range(1000):
    player_win_count = 0
    total_games = 0
    large_number = 3
    for i in range(1000):
        player_board = [[0] * cols for i in range(rows)]
        opponent_board = [[0] * cols for i in range(rows)]
        while not matrix_ouccupied(player_board, rows, cols) and not matrix_ouccupied(opponent_board, rows, cols):
            player_dice = roll_dice()  # player roll the dice
            player_board, opponent_board  = combined(player_board, opponent_board, rows, cols, large_number, player_dice)  # player place the dice
            opponent_dice = roll_dice()  # opponent roll the dice
            opponent_board, player_board  = combined( opponent_board, player_board, rows, cols, large_number, opponent_dice)  # player place the dice
        player_score = get_score(player_board, cols)
        opponent_score = get_score(opponent_board, cols)
        if player_score > opponent_score:
            player_win_count += 1
        total_games += 1
    player_win_prob = player_win_count / total_games
    games_win.append(player_win_prob)
print(games_win)
'''

# Stimulation of 10*100000 knucklebones game using strategy 4
'''
games_win_3 = np.zeros((10,100000))
long_term_prob_3 = np.zeros((1,10))
large_number_1 = 6
large_number_2 = 6
for run in range(10):
    win_count = 0
    total_games = 0
    for i in range(100000):
        player_board = [[0] * cols for i in range(rows)]
        opponent_board = [[0] * cols for i in range(rows)]
        while not matrix_ouccupied(player_board, rows, cols) and not matrix_ouccupied(opponent_board, rows, cols):
            player_dice = roll_dice()  # player roll the dice
            player_board, opponent_board  = combined(player_board, opponent_board, rows, cols, large_number_1, player_dice)  # player place the dice
            opponent_dice = roll_dice()  # opponent roll the dice
            opponent_board, player_board  = combined( opponent_board, player_board, rows, cols, large_number_2, opponent_dice)  # player place the dice
        player_score = get_score(player_board, cols)
        opponent_score = get_score(opponent_board, cols)
        if player_score > opponent_score:
            win_count += 1
        total_games += 1
        player_win_prob_2 = win_count / total_games
        games_win_3[run, i] = player_win_prob_2
    long_term_prob_3[0,run] = games_win_3[run,100000-1]
    win_count = 0


# # data for convergence plot
# run_1 = games_win_3[0, :1000]
# run_2 = games_win_3[1, :1000]
# run_3 = games_win_3[2, :1000]
# run_4 = games_win_3[3, :1000]
# run_5 = games_win_3[4, :1000]
# run_6 = games_win_3[5, :1000]
# run_7 = games_win_3[6, :1000]
# run_8 = games_win_3[7, :1000]
# run_9 = games_win_3[8, :1000]
# run_10 = games_win_3[9, :1000]
# x = range(1000)
# 
# # Plot the data vectors
# plt.plot(x, run_1)
# plt.plot(x, run_2)
# plt.plot(x, run_3)
# plt.plot(x, run_4)
# plt.plot(x, run_5)
# plt.plot(x, run_6)
# plt.plot(x, run_7)
# plt.plot(x, run_8)
# plt.plot(x, run_9)
# plt.plot(x, run_10)

# Add labels and legend
plt.ylabel('Player winning Probability')

# Display the graph
plt.show()

# Calculate the mean and standard deviation
mean = np.mean(long_term_prob_3)
std = np.std(long_term_prob_3)
n = 10

# Calculate the standard error
se = std / np.sqrt(n)

# Calculate the margin of error (99.8% CI)
moe = 3.0902 * se  # 3 is the z-value corresponding to a 99.8% confidence level for a standard normal distribution

# Calculate the lower and upper bounds of the confidence interval
lower_bound = mean - moe
upper_bound = mean + moe

# Print the confidence interval
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
'''

# S2 vs S4
games_win = np.zeros((10,100000))
long_term_prob = np.zeros((1,10))
large_number_1 = 6
large_number_2 = 6
for run in range(10):
    win_count = 0
    total_games = 0
    for i in range(100000):
        player_board = [[0] * cols for i in range(rows)]
        opponent_board = [[0] * cols for i in range(rows)]
        while not matrix_ouccupied(player_board, rows, cols) and not matrix_ouccupied(opponent_board, rows, cols):
            player_dice = roll_dice()  # player roll the dice
            player_board = place_dice(player_board, rows, cols, player_dice)  # player place the dice
            opponent_board = match_and_zero(player_board, opponent_board,
                                            cols)  # the opponent's dice may be removed
            opponent_dice = roll_dice()  # opponent roll the dice
            opponent_board, player_board  = combined( opponent_board, player_board, rows, cols, large_number_2, opponent_dice)  # player place the dice
        player_score = get_score(player_board, cols)
        opponent_score = get_score(opponent_board, cols)
        if player_score > opponent_score:
            win_count += 1
        total_games += 1
        player_win_prob_2 = win_count / total_games
        games_win[run, i] = player_win_prob_2
    long_term_prob[0,run] = games_win[run,100000-1]
    win_count = 0

# Calculate the mean and standard deviation
mean = np.mean(long_term_prob)
std = np.std(long_term_prob)
n = 10

# Calculate the standard error
se = std / np.sqrt(n)

# Calculate the margin of error (99.8% CI)
moe = 3.0902 * se  # 3 is the z-value corresponding to a 99.8% confidence level for a standard normal distribution

# Calculate the lower and upper bounds of the confidence interval
lower_bound = mean - moe
upper_bound = mean + moe

# Print the confidence interval
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
