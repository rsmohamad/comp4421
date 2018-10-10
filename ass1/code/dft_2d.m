function img_result = dft_2d(img_input, flag)
  pkg load signal  
  img_input = double(img_input);
  [N, M] = size(img_input);
  dft_R=zeros(N,M); 
  img_result=zeros(N,M); 
  
  if strcmp(flag, 'DFT') == 1    
    
    % fftshift  
    shift_mat = ones(size(img_input));  
    for i = 1:1:numel(shift_mat)
      [y, x] = ind2sub(size(shift_mat), i);
      if mod(x+y, 2) == 1
        shift_mat(i) = -1;
      end
    end
    img_input = img_input .* shift_mat;
    
    dft_mat = dftmtx(M);  
    for row = 1:1:N
      vec = img_input(row, :)';
      dft_R(row, :) = (dft_mat' * vec)'  ;
    end
    
    dft_mat = dftmtx(N);    
    for col = 1:1:M
      vec = dft_R(:, col);
      img_result(:, col) = dft_mat * vec;
    end
    
  else
    
    % ifftshift
    img_input = [img_input(:, ((M/2)+1):M) img_input(:, 1:M/2)];
    img_input = [img_input(((N/2)+1):N, :); img_input(1:N/2, :)];
    
    dft_mat = dftmtx(M);
    for row = 1:1:N
      vec = img_input(row, :)';
      dft_R(row, :) = (dft_mat * vec)'  ;
    end
    
    dft_mat = dftmtx(N);    
    for col = 1:1:M
      vec = dft_R(:, col);
      img_result(:, col) = dft_mat' * vec;
    end
    
  end
 

 
       
    
 