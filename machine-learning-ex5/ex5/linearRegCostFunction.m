function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h = X * theta;
thetaReg = [0; theta(2:length(theta))]
J = 1/(2*m) * ((h - y)' * (h - y)) + ...
    (lambda/(2*m)) * (thetaReg' * thetaReg);

grad = ((1 / m) * X' * (h - y)) + ...
    lambda / m * thetaReg;

grad = grad(:);

end
