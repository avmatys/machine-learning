function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features + 1
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
   
    h = X * theta; % Calculate hypothesis (m x 1) 
	error = h - y; % Calculate errors (m x 1)
	steps = X' * error; % (n x m) * (m x 1) = (n x 1)
	
	% Calculate new thetas
    theta = theta - alpha * (1/m) * steps; % (n x 1)
	
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
