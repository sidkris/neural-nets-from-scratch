input = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

# arbitrary initial values
weight = 0.1 
learning_rate = 0.1
epochs = 100

def predict(input):

    return weight * input


# train the neural network
for i in range(epochs):

    preds = [predict(i) for i in input]

    errors = [t - p for t, p in zip(targets, preds)]

    cost = sum(errors) / len(targets)

    print(f"WEIGHT : {weight} | COST : {cost}")

    weight += learning_rate * cost               # essentially batch gradient descent

# test the neural network
test_inputs = [5, 6]
test_targets = [10, 12]

test_preds = [predict(i) for i in test_inputs]

for i, t, p in zip(test_inputs, test_targets, test_preds):

    print(f"INPUT : {i} | TARGET : {t} | PREDICTION : {p}")


