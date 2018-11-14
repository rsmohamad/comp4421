function [img_marked, corners] = hough_transform(img)

% Implement the Hough transform to detect the target A4 paper
% Input parameter:
% .    img - original input image
% .    (You can add other input parameters if you need. If you have added
% .    other input parameters, please state for what reasons in the PDF file)
% Output parameter:
% .    img_marked - image with marked sides and corners detected by Hough transform
% .    corners - the 4 corners of the target A4 paper

  % Masks
  avg_mask = 1/25 * ones(5);
  sobel_x_mask = [-1 0 1; -2 0 2; -1 0 1];
  sobel_y_mask = [-1 -2 -1; 0 0 0; 1 2 1];
  
  img_gray = double(rgb2gray(img));
  
  % Extract edges
  img_gray = filter2(avg_mask, img_gray);
  img_gray = medfilt2(img_gray, [5, 5]);  
  x_grad = filter2(sobel_x_mask, img_gray);
  y_grad = filter2(sobel_y_mask, img_gray);  
  grad = (x_grad .* x_grad) + (y_grad .* y_grad);
  grad = sqrt(grad);
  edge = (grad > 64) * 255;
  
  % Remove ones at padding
  edge = edge(9+1:end-9, 9+1:end-9);  
  edge = padarray(edge,[9 9],0);  
  
  % Vectorized hough transform
  [y, x] = find(edge);
  numEdges = length(x);  
  angles = 0:pi/720:pi-(5*pi/720);  
  N = length(angles);  
  sinVector = sin(angles);
  cosVector = cos(angles);
  shift = norm(size(img)) + 1;
  [Y, X] = size(img);
  
  rho = floor(([x, y] * [cosVector; sinVector]) + shift);
  map = full(sparse(rho, repmat(1:N, [numEdges, 1]), 1));
  [val, idx] = sort(map(:), 'descend');
  
  [rhos thetas] = ind2sub(size(map), idx(1:72)); 
  houghlines = [rhos, thetas];
  [R T] = size(map)
  
  %{
  [_, houghmeans, d] = kmeans(houghlines, 12);
  total_dist = sum(d);
  i = 11;
  while true
    [_, houghmeans, d] = kmeans(houghmeans, i);
    if abs(total_dist - sum(d))/double(i) > 50
      break
    else
      i = i-1;
      total_dist = sum(d);
    end
  end
  %}
  
  res = [];
  done = false;
  i = 1;
  ndist = 500;
  while ~done
    hough_point = houghlines(end, :);
    res = [res; hough_point];
    min_rho = max(1, hough_point(1)-ndist)
    max_rho = min(R, hough_point(1)+ndist)
    min_the = max(1, hough_point(2)-ndist)
    max_the = min(T, hough_point(2)+ndist)
    
    neighbors = houghlines(all(houghlines(:,1) >= min_rho & houghlines(:,1) <= max_rho & houghlines(:,2) >= min_the & houghlines(:,2) <= max_the, 2), :)
    houghlines = houghlines(~neighbors, :)
    done = isempty(houghlines);    
    
  end

  houghmeans = res;
  houghmeans(:, 1) = ceil(houghmeans(:, 1)-shift);
  houghmeans(:, 2) = houghmeans(:, 2)/double(N) * max(angles);  
  
  
  % Display edges
  
  figure,
  imshow(uint8(img_gray));  
 
  for row = 1:size(houghmeans, 1)
    rho = houghmeans(row, 1);
    theta = houghmeans(row, 2);
    
    isPointOk = zeros([4, 1]);
    
    y1 = rho/sin(theta);
    isPointOk(1) = y1 > 0 && y1 <= Y;
    
    y2 = y1 - X*cot(theta);
    isPointOk(2) = y2 > 0 && y2 <= Y;
    
    x3 = rho/cos(theta);
    isPointOk(3) = x3 > 0 && x3 <= X;
    
    x4 = x3 - Y*tan(theta);
    isPointOk(4) = x4 > 0 && x4 <= X;    
    
    points = (diag(isPointOk) * [1 y1; X y2; x3 1; x4 Y])';
    points = points(:, ~all(points == 0));
    line(points(1, :), points(2, :), 'color', 'g', 'linewidth', 2);    
  end
  
end
  
  
    
  
  
  
  
  