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
    
    capped_x = [max(1, x_min):min(img_x, x_max)];
    capped_y = [max(1, y_min):min(img_y, y_max)];
    
    filter_x = [1+(capped_x(1)-x_min):filter_size-(x_max-capped_x(end))];
    filter_y = [1+(capped_y(1)-y_min):filter_size-(y_max-capped_y(end))];

    filter_trimmed = filter(filter_y, filter_x);
    window = img_input(capped_y, capped_x);
    
    conv_res = sum((window .* filter_trimmed)(:));
    img_result(i) = conv_res;
  end
  
  img_result = uint8(img_result);
  
  