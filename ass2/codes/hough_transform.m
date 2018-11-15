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
  sobel_x_mask = [-1 0 1; -1 0 1; -1 0 1];
  sobel_y_mask = [-1 -1 -1; 0 0 0; 1 1 1];
  
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
  
  angles = 0:pi/720: pi-(8*pi/720);  
  
  N = length(angles);  
  sinVector = sin(angles);
  cosVector = cos(angles);
  
  shift = norm(size(img)) + 1;
  
  rho = floor(([x, y] * [cosVector; sinVector]) + shift);
  map = full(sparse(rho, repmat(1:N, [numEdges, 1]), 1));
  
  
  [_, idx] = sort(map(:), 'descend');  
  [rhos thetas] = ind2sub(size(map), idx(1:72)); 
  
  houghlines = [rhos, thetas];
  
  % Remove local duplicates
  peaks = [];
  rho_dist = shift/8.;
  the_dist = N/9.;
  while ~isempty(houghlines)
    hough_point = houghlines(1, :);
    peaks = [peaks; hough_point];
    min_rho = hough_point(1)-rho_dist;
    max_rho = hough_point(1)+rho_dist;
    min_the = hough_point(2)-the_dist;
    max_the = hough_point(2)+the_dist;
    
    neighbors = all(houghlines(:,1) >= min_rho & 
                    houghlines(:,1) <= max_rho & 
                    houghlines(:,2) >= min_the & 
                    houghlines(:,2) <= max_the, 2);
             
    houghlines = houghlines(~neighbors, :);
  end
  
  % We are detecting a rectangle, if a line is a rectangle edge,
  % it must have another line parallel to it.
  % Remove lines that have not parallel lines.
  peaks_temp = peaks;
  peaks = [];
  for row = 1:size(peaks_temp,1)
    point = peaks_temp(row, :);
    min_the = point(2)-the_dist;
    max_the = point(2)+the_dist;
    
    neighbors = all(peaks_temp(:,2) >= min_the & 
                    peaks_temp(:,2) <= max_the, 2);
            
    if sum(neighbors) > 1
      peaks = [peaks; point];
    end    
  end
  
  % Choose top 4 lines
  if size(peaks, 1) < 4
    return
  end  
  peaks = peaks(1:4, :);
  
  % Get intersects
  corners = [];
  
  % Find one set of parallel lines
  point = peaks(1, :);
  min_the = point(2)-the_dist;
  max_the = point(2)+the_dist;    
  parallels = all(peaks(:,2) >= min_the & 
                  peaks(:,2) <= max_the, 2);
                  
  
  % Convert discrete bins to actual rhos and thetas                
  peaks(:, 1) = ceil(peaks(:, 1)-shift);
  peaks(:, 2) = peaks(:, 2)/double(N) * max(angles);
                  
  % Partition the lines into two parallel sets
  lines1 = peaks(parallels, :);
  lines2 = peaks(~parallels, :);
  
  for i = 1:size(lines1, 1)
    l1 = lines1(i, :);
    
    for j = 1:size(lines2, 1)
      l2 = lines2(j, :);
      
      rho1 = l1(1);
      theta1 = l1(2);
      rho2 = l2(1);
      theta2 = l2(2);
    
      thetaMat = [cos(theta1) sin(theta1); cos(theta2) sin(theta2)];
      rhoVec = [rho1; rho2];
      
      isect = inv(thetaMat) * rhoVec;
      corners = [corners; isect'];
    end
  end
  
  % Display edges  
  [Y, X] = size(img_gray);
  
  figure,
  imshow(uint8(img));
  
  for row = 1:size(peaks, 1)
    rho = peaks(row, 1);
    theta = peaks(row, 2);

    isPointOk = zeros([4, 1]);

    y1 = rho/sin(theta);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    isPointOk(1) = y1 > 0 && y1 <= Y;

    y2 = y1 - X*cot(theta);
    isPointOk(2) = y2 > 0 && y2 <= Y;

    x3 = rho/cos(theta);
    isPointOk(3) = x3 > 0 && x3 <= X;

    x4 = x3 - Y*tan(theta);
    isPointOk(4) = x4 > 0 && x4 <= X;    

    points = [1 y1; X y2; x3 1; x4 Y];
    points = points(find(isPointOk), :);
    
    line(points(:, 1), points(:, 2), 'color', 'r', 'linewidth', 2);   
  end
  
  hold on;
  plot(corners(:, 1), corners(:, 2), 'go', 'MarkerSize', 20, 'MarkerFaceColor', 'g');

end                                                                                                                                                                                                                         
  
  
                                            
  
  
  
  
  