input = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

# arbitrary initial values
weight = 0.1 
bias = 0.3
learning_rate = 0.1
epochs = 100

def predict(input):

    return weight * input + bias

# train the neural network
for i in range(epochs):

    preds = [predict(i) for i in input]
    errors = [(p - t) ** 2 for p, t in zip(preds, targets)] 
    cost = sum(errors) / len(targets)

    # --- Mean Squared Error --- 
    error_derivative = [2 * (p - t) for p, t in zip(preds, targets)]
    weight_delta = [e * i for e, i in zip(error_derivative, input)]
    bias_delta = [e * 1 for e in error_derivative]

    weight -= learning_rate * sum(weight_delta) / len(weight_delta)
    bias -= learning_rate * sum(bias_delta) / len(bias_delta)

    print(f"WEIGHT : {weight} | BIAS : {bias}| COST : {cost}")


# test the neural network
test_inputs = [5, 6]
test_targets = [10, 12]

test_preds = [predict(i) for i in test_inputs]

for i, t, p in zip(test_inputs, test_targets, test_preds):

    print(f"INPUT : {i} | TARGET : {t} | PREDICTION : {p}")


