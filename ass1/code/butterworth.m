function f = butterworth(size, cutoff, n)
  V = size(1);
  U = size(2);  
  f = zeros(size);
  
  for i = 1:1:numel(f);
    [y, x] = ind2sub(size, i);
    dist = sqrt((V/2-y)*(V/2-y) + (U/2-x)*(U/2-x));
    f(i) = 1 / (1 + power(dist/cutoff, 2*n));
  end
