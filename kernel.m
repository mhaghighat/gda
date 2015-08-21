function k = kernel(u,v,options)

% KERNEL determines kernel function for kernel-based machine learning 
% methods including Support Vector Machines.
% 
% 
% 
% Inputs:
%           u:          First input vector (p dimensional column vector)
%           v:          Second input vector (p dimensional column vector)
%           options:    Struct value in Matlab
%                       The fields in options that can be set:
% 
%                       options.KernelType:	choices are:
% 
%                           'linear'        without kernel function (no parameter)(default value)
%                           'poly'          simple polynomial kernel function (with 1 parameter which is degree of polynomial)
%                           'polyplus'      polynomial kernel function (with 1 parameter which is degree of polynomial)
%                           'sphnorm_poly'  spherically normalized polynomial kernel function (with 2 parameters: 
%                                           the first one is degree of polynomial & the second one is spherical normalization parameter)
%                           'rbf'           Gaussian (RBF) kernel function (with 1 parameyer which is width of RBF (sigma)
%                           'wave'          wavelet kernel function (with 1 parameter which is dilation factor. 
%                       
%                       options.KernelPars: a row vector. Minimum size of it is 0 (when no parameters is needed for kernel function)
%                                           and maximum size of it is 2 (when kernel function has 2 parameters)
%
%
% Output:
%           k:      kernel function value (a real scalar)
% 
% 
% Sample use:    
%       k = kernel(u,v,options);     % computes dot product in feature space
% 
% 
% 
% Bugs: (1) Before using this function, you should determine suitable values for options.KernelType and options.KernelPars 
%           variables otherwise the default values will be used by function.
%       (2) If you use spherically normalized kernels, you'd better already normalize the training data to zero mean and unit variance then set the spherical
%           normalization parameter to 1.
%
% 
% 
%   More details can be found in:
%   
%   S. Meshgini, A. Aghagolzadeh, H. Seyedarabi, "Face recognition using 
%   Gabor-based direct linear discriminant analysis and support vector machine," 
%   Computers & Electrical Engineering, vol. 39, no. 3, pp. 727-745, 2013.
% 
% 
% 
%   (C)	Saeed Meshgini, Ph.D.
%       University of Tabriz



% checking the correct use of input arguments:

if (~exist('options','var'))
   options.KernelType = 'linear';
end

% checking the same dimensionality of input vectors:

p = length(u);
q = length(v);
if p ~= q
    error('dimension of two vectors must be the same.')
end

% kernels:

if ~isfield(options,'KernelType')
    options.KernelType = 'linear';          % default kernel function is linear
end

switch lower(options.KernelType)
    case {lower('linear')}                 
        k = u.'*v;                          % u'*v
    case {lower('poly')}            
        if ~isfield(options,'KernelPars')
            options.KernelPars = 2;         % default value for degree of polynomial is 2
        end
        k = (u.'*v)^options.KernelPars;     % (u'*v)^n
    case {lower('polyplus')}                
        if ~isfield(options,'KernelPars')
            options.KernelPars = 2;         % default value for degree of polynomial is 2
        end
        k = (u.'*v+1)^options.KernelPars;   % (u'*v+1)^n
    case {lower('sphnorm_poly')}      
        if ~isfield(options,'KernelPars')
            options.KernelPars = [2 1];     % default value for degree of polynomial is 2 and default parameter for spherical normalization is 1
        end
        k = (1/2^options.KernelPars(1))*(((u.'*v+options.KernelPars(2)^2)/sqrt((u.'*u+options.KernelPars(2)^2)*(v.'*v+options.KernelPars(2)^2)))+1)^options.KernelPars(1);
                                            % ((u'*v+d^2)/sqrt((u'*u+d^2)(v'*v+d^2))+1)^n/2^n
    case {lower('rbf')}        
        if ~isfield(options,'KernelPars')   % default value for sigma is 1
            options.KernelPars = 1; 
        end
        k = exp((-(u-v).'*(u-v))/(2*options.KernelPars^2));
                                            % e^{-(|u-v|^2)/2(sigma)^2}
    case {lower('wave')}        
        if ~isfield(options,'KernelPars')   % default value for sigma is 1
            options.KernelPars = 2 ;        % default value for dilation factor is 2 and default mother wavelet is morlet function
        end 
%         if strcmp(options.KernelPars(2),'morl')
            pro = 1;
            for i = 1:p
                pro = pro*(cos(1.75*((u(i)-v(i))/options.KernelPars))*exp(-((u(i)-v(i))^2)/(2*options.KernelPars^2)));  
            end
            k = pro;                        % Pro(psi(ui-vi)/a), psi(x) = cos(1.75x)exp(-x^2/2)
%         elseif strcmp(options.KernelPars(2),'mexh')
%             pro = 1;
%             for i = 1:p
%                 pro = pro*(((2/sqrt(3))*pi^(-0.25))*(1-((u(i)-v(i))/options.KernelPars(1))^2)*exp((-((u(i)-v(i))/options.KernelPars(1))^2)/2));
%             end
%             k = pro;
%         elseif strcmp(options.KernelPars(2),'haar')
%             pro = 1;
%             for i = 1:p
%                 if (((u(i)-v(i))/options.KernelPars(1))> = 0)&&(((u(i)-v(i))/options.KernelPars(1))<0.5)
%                     pro = pro*1;
%                 elseif (((u(i)-v(i))/options.KernelPars(1))> = 0.5)&&(((u(i)-v(i))/options.KernelPars(1))<1)
%                     pro = pro*(-1);
%                 else
%                     pro = pro*0;
%                 end
%             end
%             k = pro;
%         else
%             error('use one of these wavelets ("morl" or "mexh" or "haar") as valid wavelet types')
%         end
    otherwise
        error('KernelType does not exist!');
end