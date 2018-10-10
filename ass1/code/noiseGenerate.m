function output = noiseGenerate(varargin)

% flags == 0: add gaussian noise
% flags == 1: add salt-and-pepper noise
% G_mean & G_sv: the mean and standard variance of Guassian noise
% S_p & P_p: the probabilities of salt noise and pepper noise respectly
% Parameters list can be: (input,flags)
%                         (input,flags,G_mean,G_sv) with flags == 0
%                         (input,flags,S_p,P_p) with flags == 1

input = varargin{1};
flags = varargin{2};

input = double(input(:,:,1));
[m, n] = size(input);

input = input/255;

if flags == 0
    if nargin == 2
        G_mean = 0;
        G_sv = 30;
    elseif nargin == 4
        G_mean = varargin{3};
        G_sv = varargin{4};
    end
    n_guassian = G_mean + (G_sv/255).*randn(m,n);
    output = n_guassian + input;
elseif flags == 1
    if nargin == 2
        S_p = 0.2;
        P_p = 0.2;
    elseif nargin == 4
        S_p = varargin{3};
        P_p = varargin{4};
    end
    x = rand(m, n);
    output = input;
    output(find(x <= S_p)) = 0;
    output(find(x > S_p & x <(S_p+P_p))) = 1;
end

output = uint8(mat2gray(output) * 255);

end