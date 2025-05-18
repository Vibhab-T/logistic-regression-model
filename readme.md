sigmoid function that returns probability:
y = 1 / (1 + e^ (-x))

this probability value will be our predictions. >0.5 is labelled as 1, else 0.

Gradient descent has been used to properly train the model to find the best weight and bias.

weight = weight - learning rate _ weight gradient
bias = bias - learning rate _ bias gradient
