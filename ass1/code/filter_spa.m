function img_result = filter_spa(img_input, filter)
  [img_y, img_x] = size(img_input);
  filter_size = size(filter)(1);
  f2 = double(floor(filter_size / 2));
  
  img_result = zeros(size(img_input));
  img_input = double(img_input);
  
  for i = 1:1:numel(img_input)
    [y, x] = ind2sub(size(img_input), i);
    y = double(y);
    x = double(x);
    y_min = double(y-f2);
    y_max = double(y+f2);
    x_min = double(x-f2);
    x_max = double(x+f2);
    
    if y_min > 0 && y_max <= img_y && x_min > 0 && x_max <= img_x
      sub_img = img_input(y_min:y_max, x_min:x_max);
      conv_res = sum((sub_img .* filter)(:));
      img_result(i) = conv_res;
    end
    
  end
  
  img_result = uint8(img_result);
  
  