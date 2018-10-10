function dftTest()
  pkg load image
  img = imread('../73.png');
  dft_img = dft_2d(img, 'DFT');
  dft_spectrum = log10(dft_img + 1);
  dft_spectrum = mat2gray(dft_spectrum);
  figure,
  subplot(121), imshow(dft_spectrum), title('Fourier Spectrum')
  