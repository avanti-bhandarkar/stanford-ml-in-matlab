function p = predictOneVSall(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];


predictions_for_each_k = sigmoid( X * all_theta);

[k_probability, k_value_predicted] = max( predictions_for_each_k, [], 2);

p = k_value_predicted;
