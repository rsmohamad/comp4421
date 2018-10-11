function f = gaussian(size, cutoff)
  V = size(1);
  U = size(2);  
  f = zeros(size);
  
  for i = 1:1:numel(f);
    [y, x] = ind2sub(size, i);
    dist = ((V/2-y)*(V/2-y) + (U/2-x)*(U/2-x));
    f(i) = exp(-abs(dist)/(2*cutoff*cutoff));
  end
  