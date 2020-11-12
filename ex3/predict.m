function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

X = [ones(m, 1), X];

z2 = Theta1 * transpose(X);

a2 = transpose(sigmoid(z2));

% Add a0 of layer = 1
ma2 = size(a2, 1);
a2 = [ones(ma2, 1), a2];

z3 = Theta2 * transpose(a2);
a3 = transpose(sigmoid(z3));

% Get the maximum probability of each index and also get the index so we know which number was predicted.
[k_probability, k_value_predicted] = max( a3, [], 2);

p = k_value_predicted;

