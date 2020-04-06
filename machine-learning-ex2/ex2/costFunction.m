function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

h = sigmoid(X * theta); % (m x (n+1)) * ((n+1) x 1) = (m x 1) 
J = -1/m * (y'*log(h) + (1-y)'*log(1-h)); % (1 x 1) * ((1 x m) * (m x 1) + (1 x m) * (m x 1)) = (1 x 1)
grad = 1/m * X' * (h-y); % (((n+1) x m) * (m x 1)) = ((n+1) x 1)


end
