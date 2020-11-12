function J = computecost(X, y, theta)
m = length(y); % number of training examples
J = 0;

%cost function
J = (1/(2*m))*(X*theta - y).*(X*theta - y);
