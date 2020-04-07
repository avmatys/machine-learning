function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features + 1

% Compute without regularization
[J, grad] = costFunction(theta, X, y);

% Regularized cost 
theta_sliced = theta(2:n); % theta for regularization computing, excluded first row
J = J + lambda/(2*m) * (theta_sliced' * theta_sliced);% J + coef * sum of squared theta elements (1 x 1)

% Regularized gradient
grad = grad + [0; lambda/m * theta_sliced] 

end
