function [all_theta] = oneVSall(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

% add ones to the X data matrix
X = [ones(m, 1) X];

for (c = 1:num_labels)
  initial_theta = zeros(n + 1, 1);
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  [theta] = ...
          fmincg (@(t)(lrcostfunction(t, X, (y == c), lambda)), ...
                  initial_theta, options);
  all_theta(c,:) = theta;
end
