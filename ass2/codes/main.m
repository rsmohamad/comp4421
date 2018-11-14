clc
clear

% For octave
%pkg load image

% Assignment 2
% img - original input image
% img_marked - image with marked sides and corners detected by Hough transform
% corners - the 4 corners of the target A4 paper
% img_warp - the standard A4-size target paper obtained by image warping
% n - determine the size of the result image

% define the n by yourself
n = 500;
inputs = [1:6];
for i = 1:length(inputs)
    img_name = [num2str(inputs(i)), '.JPG'];
    img = imread(img_name);
    hough_transform(img);
    %[img_marked, corners] = hough_transform(img);
    %img_warp = img_warping(img, corners, n);
    %figure, 
    %subplot(131),imshow(img);
    %subplot(132),imshow(img_marked);
    %subplot(133),imshow(img_warp);
end