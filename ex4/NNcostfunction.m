function [J grad] = NNcostfunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

y_matrix = eye(num_labels)(y,:);

a1 = [ones(m, 1), X];
z2 = a1 * transpose(Theta1);
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2];
z3 = a2 * transpose(Theta2);
a3 = sigmoid(z3);
hx = a3;

penalty = sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2));

J = (1/m) * sum(sum( (-y_matrix).*log(hx) - (1 - y_matrix).*log(1-hx), 2)) + lambda*penalty/(2*m);

% Perform back propagation. Calculate the error and then the Delta.
d3 = a3 - y_matrix;
d2 = d3 * Theta2(:, 2:end) .* sigmoidGradient(z2);

Delta1 = transpose(d2) * a1;
Delta2 = transpose(d30 * a2;

Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;


% Add regularization to all the gradients
Theta1_r = Theta1;
Theta1_r(:,1) = 0;
Theta2_r = Theta2;
Theta2_r(:,1) = 0;
Theta1_grad = Theta1_grad + (lambda/m) * Theta1_r;
Theta2_grad = Theta2_grad + (lambda/m) * Theta2_r;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
