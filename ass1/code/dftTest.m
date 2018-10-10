function dftTest()
  pkg load image
  img = imread('../73.png');
  dft_img = dft_2d(img, 'DFT');
  fft_img = fft2(img);
  dft_spectrum = log10(dft_img + 1);
  dft_spectrum = mat2gray(dft_spectrum);
  
  difference = sum(abs(dft_img - fft_img)(:))
  
  idft_img = dft_2d(fft_img, 'IDFT');

  % Transform idft_img to a real-value matrix
  real_img = real(idft_img);
  real_img = mat2gray(real_img);
  
  figure,
  subplot(121), imshow(dft_spectrum), title('Fourier Spectrum')
  subplot(122), imshow(real_img), title('Image after IDFT')
  