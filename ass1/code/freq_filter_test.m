function freq_filter_test()
  pkg load image
  img = imread('../73.png');
  averaging_mask = 1/25 * ones(5);
  laplacian_mask = [0 -1 0; -1 4 -1; 0 -1 0];
  sobelX_mask = [-1 0 1; -1 0 1; -1 0 1];

  ave_freq = filter2d_fre(img, averaging_mask);
  laplacian_freq = filter2d_fre(img, sobelX_mask);

  figure,
  subplot(121), imshow(ave_freq), title('Averaging in Frequancy Domain')
  subplot(122), imshow(laplacian_freq), title('Sharpen in Frequency Domain')
  