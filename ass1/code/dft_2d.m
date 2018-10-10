function img_result = dft_2d(img_input, flag)
  img_input = double(img_input);
  img_result = zeros(size(img_input));
  [N, M] = size(img_input)
  
  if flag == 'DFT'
    
    for freq_index = 1:1:numel(img_result)
      [v, u] = ind2sub(size(img_result), freq_index);
      
      res = 0;
      for spatial_index = 1:1:numel(img_input)
        [y, x] = ind2sub(size(img_input), spatial_index);
        angle = 2 * pi * ((u*x)/M + (v*y)/N);   
        res = res + (img_input(spatial_index) * cos(angle));
      end
      
      res = res / numel(img_input)
      img_result(freq_index) = res;
      
    end
    
  end
       
    
 