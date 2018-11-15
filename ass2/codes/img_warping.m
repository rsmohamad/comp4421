function img_warp = img_warping(img, corners, n)

% Implement the image warping to transform the target A4 paper into the
% standard A4-size paper
% Input parameter:
% .    img - original input image
% .    corners - the 4 corners of the target A4 paper detected by the Hough transform
% .    (You can add other input parameters if you need. If you have added
% .    other input parameters, please state for what reasons in the PDF file)
% .    n - determine the size of the result image
% Output parameter:
% .    img_warp - the standard A4-size target paper obtained by image warping

  % Sort the corners
  minX = min(corners(:, 1));
  minY = min(corners(:, 2));
  corners(:, 1) -= minX;
  corners(:, 2) -= minY;  
  sumCoord = sum(corners, 2); 
  
  [_ c1ind] = min(sumCoord);
  [_ c4ind] = max(sumCoord); 
  m = ones([4, 1]);
  m(c1ind) = 0;
  m(c4ind) = 0;
  remain_idx = find(m);  
  [_ c2ind] = max(corners(remain_idx, 1));
  [_ c3ind] = max(corners(remain_idx, 2));  
  c2ind = remain_idx(c2ind);
  c3ind = remain_idx(c3ind); 
  
  corners_new = [corners(c1ind, :); corners(c2ind, :); corners(c3ind, :); corners(c4ind, :)];
  corners_new(:, 1) += minX;
  corners_new(:, 2) += minY;  
  corners = corners_new;
  
  % Determine horizontal/vertical
  width = norm(corners(1, :) - corners(2, :)) + norm(corners(3, :) - corners(4, :));
  height = norm(corners(1, :) - corners(3, :)) + norm(corners(2, :) - corners(4, :));
  isHorizontal = width > height;
  
  % Define the target coordinate
  dim = [297 210] * n;
  if isHorizontal 
    dim = fliplr(dim);
  end 
  H = dim(1);
  W = dim(2);
  
  target = [1 1; W 1; 1 H; W H];
  
  % Determine the transformation parameters
  A = [];
  for point = 1:size(corners, 1)
    x = target(point, 1);
    y = target(point, 2);
    A = [A; x y x*y 1];
  end
  
  x_params = inv(A) * corners(:, 1);
  y_params = inv(A) * corners(:, 2);
  
  % Calculate the mapping target -> source
  widths = repmat([1:W], H, 1);
  heights = repmat([1:H]', 1, W);
  wh = widths .* heights;
  one = ones(size(widths));
  
  mapping_x = cat(3, widths*x_params(1), heights*x_params(2), wh*x_params(3), one*x_params(4));
  mapping_x = sum(mapping_x, 3);
  
  mapping_y = cat(3, widths*y_params(1), heights*y_params(2), wh*y_params(3), one*y_params(4));
  mapping_y = sum(mapping_y, 3);
  
  % Bilinear interpolation
  img_warp = zeros([H W 3]);
   
  for row = 1:size(mapping_x, 1)
    for col = 1:size(mapping_x, 2)
      x = mapping_x(row, col);
      y = mapping_y(row, col);
      
      minx = floor(x);
      miny = floor(y);
      maxx = ceil(x);
      maxy = ceil(y);
      
      x -= minx;
      y -= miny;
      
      val = (1-y) * (1-x) * img(miny, minx, :);
      val += y * (1-x) * img(miny, maxx, :);
      val += (1-y) * x * img(maxy, minx, :);
      val += y * x * img(maxy, maxx, :);
      
      img_warp(row, col, :) = val;
      
    end
  end
  
  img_warp = uint8(img_warp);
  
end
  
  
  
  
  
  
  
  
  
