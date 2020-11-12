function [h, display_array] = displaydata(X, example_width)

% setting example_width automatically if not passed in
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));
end

% grayscale image
colormap(gray);

% compute rows, columns
[m n] = size(X);
example_height = (n / example_width);

% compute number of items to display
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% image padding
pad = 1;

% setup blank display
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

% copy each example into a patch on the display array
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
			break; 
		end
		
		% getting the max value of the patch
		max_val = max(abs(X(curr_ex, :)));
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% display Image
h = imagesc(display_array, [-1 1]);
axis image off

drawnow;
