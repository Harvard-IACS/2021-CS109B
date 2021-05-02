# This helper file, setups the rules and rewards for the mouse grid system

# State = 1, start point
# Action - 1: Top, 2:Left, 3:Right, 4:Down

def transition_rules(state, action):
    # For state 1
    if state == 1 and (action == 3 or action == 4):
        state = 1
    elif state == 1 and action == 1:
        state = 5
    elif state == 1 and action == 2:
        state = 2

    # For state 2
    elif state == 2 and action == 4:
        state = 2
    elif state == 2 and action == 1:
        state = 5
    elif state == 2 and action == 2:
        state = 3
    elif state == 2 and action == 3:
        state = 1

    # For state 3
    elif state == 3 and action == 4:
        state = 3
    elif state == 3 and action == 1:
        state = 7
    elif state == 3 and action == 2:
        state = 4
    elif state == 3 and action == 3:
        state = 2

    # For state 4
    elif state == 4 and (action == 4 or action == 2):
        state = 4
    elif state == 4 and action == 1:
        state = 8
    elif state == 4 and action == 3:
        state = 3

    # For state 5
    elif state == 5:
        state = 1

    # For state 6
    elif state == 6 and action == 1:
        state = 10
    elif state == 6 and action == 2:
        state = 7
    elif state == 6 and action == 3:
        state = 5
    elif state == 6 and action == 4:
        state = 2

    # For state 7
    elif state == 7:
        state = 1

    # For state 8
    elif state == 8 and action == 1:
        state = 12
    elif state == 8 and action == 2:
        state = 8
    elif state == 8 and action == 3:
        state = 7
    elif state == 8 and action == 4:
        state = 3

    # For state 9
    elif state == 9 and action == 1:
        state = 13
    elif state == 9 and action == 2:
        state = 10
    elif state == 9 and action == 3:
        state = 9
    elif state == 9 and action == 4:
        state = 5

    # For state 10
    elif state == 10 and action == 1:
        state = 14
    elif state == 10 and action == 2:
        state = 11
    elif state == 10 and action == 3:
        state = 9
    elif state == 10 and action == 4:
        state = 6


    # For state 11
    elif state == 11 and action == 1:
        state = 15
    elif state == 11 and action == 2:
        state = 12
    elif state == 11 and action == 3:
        state = 10
    elif state == 11 and action == 4:
        state = 7

    # For state 12
    elif state == 12 and action == 1:
        state = 16
    elif state == 12 and action == 2:
        state = 12
    elif state == 12 and action == 3:
        state = 11
    elif state == 12 and action == 4:
        state = 8

    # For state 13
    elif state == 13:
        state = 1

    # For state 14
    elif state == 14 and action == 1:
        state = 14
    elif state == 14 and action == 2:
        state = 15
    elif state == 14 and action == 3:
        state = 13
    elif state == 14 and action == 4:
        state = 10

    # For state 15
    elif state == 15 and action == 1:
        state = 15
    elif state == 15 and action == 2:
        state = 16
    elif state == 15 and action == 3:
        state = 14
    elif state == 15 and action == 4:
        state = 11

    # For state 16
    elif state == 16:
        state = 16

    return state


def reward_rules(state, prev_state):
    if state == 16:
        reward = 100

    elif state == 5 or state == 7 or state == 13 or state == 14:
        reward = -10

    elif prev_state > state:
        reward = -1

    elif prev_state == state:
        reward = 0

    else:
        reward = 1

    return reward
