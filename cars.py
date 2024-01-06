car_age = [0.2, 1.0, 1.4, 1.6, 2.0, 2.2, 2.7, 2.8, 3.2, 3.3, 3.5, 3.7, 4.0, 4.4, 5.0, 5.2]
repair_cost = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290, 870, 1545, 1480, 1750, 1845, 1790, 1955]

weight = 0.1
bias = 0.3
epochs = 100
learning_rate = 0.01

def predict(car_age):

    return weight * car_age + bias 


# training

for epoch in range(epochs):

    preds = [predict(i) for i in car_age]
    errors = [(p - t) ** 2 for p, t in zip(preds, repair_cost)] 
    cost = sum(errors) / len(repair_cost)

    # --- Mean Squared Error --- 
    error_derivative = [2 * (p - t) for p, t in zip(preds, repair_cost)]
    weight_delta = [e * i for e, i in zip(error_derivative, car_age)]
    bias_delta = [e * 1 for e in error_derivative]

    weight -= learning_rate * sum(weight_delta) / len(weight_delta)
    bias -= learning_rate * sum(bias_delta) / len(bias_delta)

    print(f"WEIGHT : {weight} | BIAS : {bias}| COST : {cost}")

# testing
print(predict(0.5))