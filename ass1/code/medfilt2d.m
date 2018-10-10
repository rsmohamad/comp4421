function img_result = medfilt2d(img_input, filter_size)
  [img_y, img_x] = size(img_input);  
  img_result = zeros(size(img_input));
  img_input = double(img_input);
  
  f2 = floor(filter_size/2)
  
  for i = 1:1:numel(img_input)
    [y, x] = ind2sub(size(img_input), i);
    y = int16(y);
    x = int16(x);
    y_min = int16(y-f2);
    y_max = int16(y+f2);
    x_min = int16(x-f2);
    x_max = int16(x+f2);
    
    x_min = max(1, x_min);
    y_min = max(1, y_min);
    x_max = min(img_x, x_max);
    y_max = min(img_y, y_max);
    
    sub_img = sort(img_input(y_min:y_max, x_min:x_max)(:));
    mid = int16(floor(numel(sub_img)/2) + 1);
    med_res = sub_img(mid);
    img_result(i) = med_res;  
    
  end
  
  img_result = uint8(img_result);