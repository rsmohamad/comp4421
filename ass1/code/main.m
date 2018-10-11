#!/usr/bin/octave -qf

clc
clear

% For Octave
pkg load image

img = imread('../73.png');

%% --------------- Task 1: Spatial Linear Filtering --------------- 
averaging_mask = 1/25 * ones(5);
sobelX_mask = [-1 0 1; -1 0 1; -1 0 1];
sobelY_mask = [-1 -1 -1; 0 0 0; 1 1 1];
laplacian_mask = [0 -1 0; -1 4 -1; 0 -1 0];

ave_result = filter_spa(img, averaging_mask);
sobelX_result = filter_spa(img, sobelX_mask);
sobelY_result = filter_spa(img, sobelY_mask);
laplacian_result = filter_spa(img, laplacian_mask);

subplot(221), imshow(ave_result), title('Averaging')
subplot(222), imshow(sobelX_result), title('Sobel X')
subplot(223), imshow(sobelY_result), title('sobel Y')
subplot(224), imshow(laplacian_result), title('Laplacian')

path = '../result_images/task1/%s.png';
imwrite (ave_result, sprintf(path, 'img_ave'));
imwrite (sobelX_result, sprintf(path, 'img_dx'));
imwrite (sobelY_result, sprintf(path, 'img_dy'));
imwrite (laplacian_result, sprintf(path, 'img_sharpen'));

%%  --------------- Task 2: Spatial Non-linear Filtering  --------------- 

% add gaussian noises to the original input image
img_gau = noiseGenerate(img, 0, 0, 30);

% add salt-and-pepper noises to the original input image
img_sp = noiseGenerate(img, 1, 0.3, 0.3);

size = 3;

gau_result = medfilt2d(img_gau, size);
sp_result = medfilt2d(img_sp, size);

figure,
subplot(121), imshow(gau_result), title('Median Filter with Gaussian Noises')
subplot(122), imshow(sp_result), title('Median Filter with Salt-and-Pepper Noises')

path = '../result_images/task2/%s.png';
imwrite (img_gau, sprintf(path, 'img_gaussian'));
imwrite (gau_result, sprintf(path, 'med_gaussian'));
imwrite (img_sp, sprintf(path, 'img_sp'));
imwrite (sp_result, sprintf(path, 'med_sp'));


%%  --------------- Task 3: Discrete Fourier Transform  --------------- 
dft_img = dft_2d(img, 'DFT');

% Compute the Fourier Spectrum. Remember to do the enhancement.
dft_spectrum = log10(dft_img + 1);
dft_spectrum = mat2gray(dft_spectrum);

idft_img = dft_2d(dft_img, 'IDFT');

% Transform idft_img to a real-value matrix
real_img = real(idft_img);
real_img = mat2gray(real_img);

figure,
subplot(121), imshow(dft_spectrum), title('Fourier Spectrum')
subplot(122), imshow(real_img), title('Image after IDFT')

path = '../result_images/task3/%s.png';
imwrite (dft_spectrum, sprintf(path, 'dft_spectrum'));
imwrite (real_img, sprintf(path, 'idft_real'));

%% ---------- Task 4: Filtering in the Frequency Domain -----------
averaging_mask = 1/25 * ones(5);
laplacian_mask = [0 -1 0; -1 4 -1; 0 -1 0];

ave_freq = filter2d_fre(img, averaging_mask);
laplacian_freq = filter2d_fre(img, laplacian_mask);

figure,
subplot(121), imshow(ave_freq), title('Averaging in Frequancy Domain')
subplot(122), imshow(laplacian_freq), title('Sharpen in Frequency Domain')

path = '../result_images/task4/%s.png';
imwrite (ave_freq, sprintf(path, 'img_ave_freq'));
imwrite (laplacian_freq, sprintf(path, 'img_sharpen_freq'));

%% Task 5: High-Frequency Emphasis
a = 0.1;
b = 0.9;

butter_result = high_freq_emphasis(img, a, b, 'butterworth');
gaussian_result = high_freq_emphasis(img, a, b, 'gaussian');

figure,
subplot(121), imshow(butter_result), title('Using Butterworth')
subplot(122), imshow(gaussian_result), title('Using Gaussian')

path = '../result_images/task5/%s_%.2f_%.2f.png';
imwrite (butter_result, sprintf(path, 'butter_emphasis', a, b));
imwrite (gaussian_result, sprintf(path, 'gaussian_emphasis', a, b));

% Remove parameters from filename for latex
path = '../result_images/task5/%s.png';
imwrite (butter_result, sprintf(path, 'butter'));
imwrite (gaussian_result, sprintf(path, 'gaussian'));
