function result = filter2d_fre(img_input, mask)
  img_freq = dft_2d(img_input, 'DFT');
  [Y, X] = size(img_input);
  n = size(mask)(1);
  mask = padarray(mask, [(Y-n), (X-n)], 'post');
  mask = double(mask);
  
  mask_freq = dft_2d(mask, 'DFT');
  
  filtered_img = (img_freq) .* (mask_freq);
  
  result = dft_2d(filtered_img, 'IDFT');
  result = real(result);
  result = mat2gray(result);
  