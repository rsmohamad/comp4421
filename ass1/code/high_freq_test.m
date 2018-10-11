function high_freq_test()
  pkg load image
  img = imread('../73.png');
  a = 0.1;
  b = 0.9;

  butter_result = high_freq_emphasis(img, a, b, 'butterworth');
  gaussian_result = high_freq_emphasis(img, a, b, 'gaussian');

  figure,
  subplot(121), imshow(butter_result), title('Using Butterworth')
  subplot(122), imshow(gaussian_result), title('Using Gaussian')