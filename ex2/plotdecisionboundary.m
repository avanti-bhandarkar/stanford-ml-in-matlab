function plotdecisionboundary(theta, X, y)

plottingdata(X(:,2:3), y);
hold on;

if size(X, 2) <= 3
    %choosing 2 endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % plot and adjust axes 
    plot(plot_x, plot_y)

    % legend
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapfeature(u(i), v(j))*theta;
        end
    end
    z = z; 
  
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off;

