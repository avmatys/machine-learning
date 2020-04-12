function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features + 1

% Calculate unregulated
h = sigmoid(X * theta);
J_unregulized = -1/m * (y'*log(h) + (1-y)'*log(1-h));
gradUnregulized = 1/m * X' * (h-y);

% Regularization cost 
theta1n = theta(2:n); 
regulazationCost = lambda/(2*m) * (theta1n' * theta1n);
regularizationGrad = [0; lambda/m * theta1n];

J = J_unregulized + regulazationCost;
grad = gradUnregulized + regularizationGrad;

grad = grad(:);

end
