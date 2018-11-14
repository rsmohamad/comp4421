function img_warp = img_warping(img, corners, n)

% Implement the image warping to transform the target A4 paper into the
% standard A4-size paper
% Input parameter:
% .    img - original input image
% .    corners - the 4 corners of the target A4 paper detected by the Hough transform
% .    (You can add other input parameters if you need. If you have added
% .    other input parameters, please state for what reasons in the PDF file)
% Output parameter:
% .    img_warp - the standard A4-size target paper obtained by image warping
% .    n - determine the size of the result image