function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% Add basis to X
X = [ones(m,1), X];

% Calculate hidden layer
hiddenLayer = sigmoid(X * Theta1');

% Add basis to hiddenLayer
hiddenLayer = [ones(m,1), hiddenLayer];

% Calculate output layer
outputLayer = sigmoid(hiddenLayer * Theta2');

% You need to return the following variables correctly 
[maxProbabilities, p] = max(outputLayer, [], 2);

end
