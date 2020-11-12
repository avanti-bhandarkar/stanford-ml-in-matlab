function [J, grad] = costfunctionreg(theta, X, y, lambda)

m = length(y); 
J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);
thetawithout0 = theta([2:size(theta,1)]);
regularization = (lambda/(2 * m)) * sum((thetawithout0).^2);

%cost function
J = (1/m) * (-y.* log(h) - (1-y).* log(1 - h)) + regularization;

% gradient with regularization excluding theta = 0

gradientregularization = (lambda/m) * theta;
gradientregularization(1) = 0;
grad = (1/m) * X.* (h - y) + gradientregularization;


