function img_result = high_freq_emphasis(img_input, a, b, type)
  img_freq = dft_2d(img_input, 'DFT');
  
  cutoff = 0.1 * min(size(img_input));
  
  if strcmp(type, 'butterworth') == 1
    freq_mask = 1 - butterworth(size(img_freq), cutoff, 1);
  elseif strcmp(type, 'gaussian') == 1
    freq_mask = 1 - gaussian(size(img_freq), cutoff);
  end
  
  freq_mask = a + (b * freq_mask);
  img_result = img_freq .* freq_mask;
  img_result = dft_2d(img_result, 'IDFT');
  img_result = abs(img_result);
  img_result = mat2gray(img_result);
    