import numpy as np


# Function to get the initial loss
# Arguments: vocabbulary size, length of sequences i.e. number of words to generate
# *****
def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0 / vocab_size) * seq_length


# Moving average instead current loss
# *****
def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001


# *****
def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character
    print('%s' % (txt,), end='')


# Function to get the initial parameters
# Arguments: number of data points, vocab_size = 27, vocab_size = 27
# *****
def initialize_parameters(n_h, n_x, n_y):
    # Initialize weight V (from input to hidden)
    V = np.random.randn(n_h, n_x) * 0.01

    # Initialize weight U (from hidden to hidden) 
    U = np.random.randn(n_h, n_h) * 0.01

    # Initialize weight W (from hidden to output) 
    W = np.random.randn(n_y, n_h) * 0.01

    # Intialize the beta1 values
    beta1 = np.zeros((n_h, 1))

    # Intialize the beta2 values
    beta2 = np.zeros((n_y, 1))

    # Dictionary to hold V,U,W, beta1 and beta2
    parameters = {"V": V, "U": U, "W": W, "beta1": beta1, "beta2": beta2}

    # Return the parameters dictionary
    return parameters


# Function to update the weights and biases
# Arguments: Parameters dictionary, gradient dictionary and learning rate
# *****
def update_parameters(parameters, gradients, lr):
    parameters['V'] += -lr * gradients['dV']
    parameters['U'] += -lr * gradients['dU']
    parameters['W'] += -lr * gradients['dW']
    parameters['beta1'] += -lr * gradients['dbeta1']
    parameters['beta2'] += -lr * gradients['dbeta2']
    return parameters


# Function to compute the next state h and output
# Arguments: The input x, the previous hidden state, the parameters dictionary (weights)
# *****
def rnn_cell_forward(xt, h_prev, parameters):
    V = parameters["V"]
    U = parameters["U"]
    W = parameters["W"]
    beta1 = parameters["beta1"]
    beta2 = parameters["beta2"]

    # Compute the next state
    h_next = np.tanh(np.dot(V, xt) + np.dot(U, h_prev) + beta1)

    # Compute the output
    yt_pred = softmax(np.dot(W, h_next) + beta2)

    # Store the previous hidden state, the next hidden state, the input and the parameters dictionary as a tuple
    cache = (h_next, h_prev, xt, parameters)

    # Return the next hidden state, the output and the cache tuple
    return h_next, yt_pred, cache


# Function to compute the forward pass of the RNN
# *****
def rnn_forward(X, Y, h0, parameters, vocab_size=27):
    x, h, y_hat = {}, {}, {}
    h[-1] = np.copy(h0)
    loss = 0
    for t in range(len(X)):
        x[t] = np.zeros((vocab_size, 1))
        if (X[t] != None):
            x[t][X[t]] = 1
        h[t], y_hat[t], _ = rnn_cell_forward(x[t], h[t - 1], parameters)
        loss -= np.log(y_hat[t][Y[t], 0])
    cache = (y_hat, h, x)
    return loss, cache


# Function to compute the gradients of the weights and biases
# *****
def rnn_cell_backward(dy, gradients, parameters, x, h, h_prev):
    gradients['dW'] += np.dot(dy, h.T)
    gradients['dbeta2'] += dy
    dh = np.dot(parameters['W'].T, dy) + gradients['dh_next']  # backprop into h
    dhrhw = (1 - h * h) * dh  # backprop through tanh nonlinearity
    gradients['dbeta1'] += dhrhw
    gradients['dV'] += np.dot(dhrhw, x.T)
    gradients['dU'] += np.dot(dhrhw, h_prev.T)
    gradients['dh_next'] = np.dot(parameters['U'].T, dhrhw)
    return gradients


# Function to compute the backward pass of the RNN
# *****
def rnn_backward(X, Y, parameters, cache):
    gradients = {}
    (y_hat, h, x) = cache
    U, V, W, beta2, beta1 = parameters['U'], parameters['V'], parameters['W'], parameters['beta2'], parameters['beta1']
    gradients['dV'], gradients['dU'], gradients['dW'] = np.zeros_like(V), np.zeros_like(U), np.zeros_like(W)
    gradients['dbeta1'], gradients['dbeta2'] = np.zeros_like(beta1), np.zeros_like(beta2)
    gradients['dh_next'] = np.zeros_like(h[0])

    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_cell_backward(
            dy, gradients, parameters, x[t], h[t], h[t - 1])

    return gradients, h


# Function to compute the softmax given the input 
# *****
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# Function to perform gradient clipping
# *****
def clip(gradients, maxValue):
    # Get the gradients from the dictionary
    dU, dV, dW, dbeta1, dbeta2 = gradients['dU'], gradients['dV'], gradients['dW'], gradients['dbeta1'], gradients[
        'dbeta2']

    # For all the gradient values run a loop
    for gradient in [dV, dU, dW, dbeta1, dbeta2]:
        # Clip the gradient value if it goes below or above the threshold
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    # Reinitialize the gradient dictionary
    gradients = {"dU": dU, "dV": dV, "dW": dW, "dbeta1": dbeta1, "dbeta2": dbeta2}

    return gradients


# Function to train the RNN using forward and backprop
# *****
def optimize(X, Y, h_prev, parameters, learning_rate=0.01):
    loss, cache = rnn_forward(X, Y, h_prev, parameters)

    gradients, h = rnn_backward(X, Y, parameters, cache)

    gradients = clip(gradients, 5)

    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, h[len(X) - 1], parameters


# Function that defines the RNN model and outputs a dinosaur name
def rnn(data, ix_to_char, char_to_ix, num_iterations=10001, n_h=50, vocab_size=27):
    # Get size of vocabulary i.e. 27 one for each alphabet and one for new space
    n_x, n_y = vocab_size, vocab_size

    # Call the initialize parameters
    parameters = initialize_parameters(n_h, n_x, n_y)

    # Call the loss function with the vocab_size
    loss = get_initial_loss(vocab_size, 5)

    # Read the dinos.txt file
    with open("dinos.txt") as f:
        # Read the lines in the file
        examples = f.readlines()

    # Get each word in lowercase
    examples = [x.lower().strip() for x in examples]

    # # Set the random seed 
    # np.random.seed(10)

    # Shuffle the input characters
    np.random.shuffle(examples)

    # Initialize the first hidden state as a list of zeroes
    h_prev = np.zeros((n_h, 1))

    # Initialize a dictionary to store the loss at every 1000th iteration
    it = {}

    # Run a loop for the specified number of iterations
    for j in range(num_iterations):
        # For each character in the sample convert to the alphabet to its corresponding interger
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]

        # Add a new line after each output
        Y = X[1:] + [char_to_ix["\n"]]

        # Call the optimize function with X, y (generated above), previous hidden state and the parameters dictionary
        curr_loss, gradients, h_prev, parameters = optimize(X, Y, h_prev, parameters)

        # Call the smooth function with the loss i.e. the previous loss and the current loss
        loss = smooth(loss, curr_loss)

    return parameters


def get_weights(num_iterations=1000, random=1):
    if random == 0:
        # Read the dinos.txt file
        data = open('dinos.txt', 'r').read()

        # Convert the data to lower case
        data = data.lower()

        # Convert the file data into list
        chars = list(set(data))

        # Get length of the file and length of the vocabulary
        data_size, vocab_size = len(data), len(chars)

        # Define dictionary of alphabets:integer
        char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}

        # Define dictionary of integer:alphabets
        ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}

        # Call the rnn_model function
        parameters = rnn(data, ix_to_char, char_to_ix, num_iterations)

    else:

        parameters = initialize_parameters(50, 27, 27)

    return parameters
