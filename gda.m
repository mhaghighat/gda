function mappedData = gda(data,trainData,trainLabel,nDim,options)

% GDA Performs Generalized Discriminant Analysis, a non-linear feature
% dimensionality reduction technique.
% 
% GDA is one of dimensionality reduction techniques, which projects a data 
% matrix from a high-dimensional space into a low-dimensional space by 
% maximizing the ratio of between-class scatter to within-class scatter. 
% 
% 
% Inputs:
%       data:           p-dimensional matrix containing the high-dimensional data to be projected
%                       p:  number of dimensions in high-dimensional space
%
%       trainData:      pxn matrix containing the high-dimensional training data
%                       n:  number of training samples
% 
%       trainLabel:     Row vector of length n containing the class labels for training data
% 
%       nDim:           Numer of dimensions to be retained (nDim < c)
%                       Default:    c-1
%                       c:  number of classes
% 
%       options:        Please see the kernel function (kernel.m).
%         
% 
% Output:
%       mappedData:      nDim-dimensional projected data matrix
% 
% 
% Sample use:
% trainGda  =  gda(trainData,trainData,trainLabel);     % Project the training data matrix into a low-dimensional space
% testGda  =  gda(testData,trainData,trainLabel);       % Project the test data matrix into a low-dimensional space
% 
% 
% 
%   Details can be found in Section 4.3 of:
%   
%   M. Haghighat, S. Zonouz, M. Abdel-Mottaleb, "CloudID: Trustworthy 
%   cloud-based and cross-enterprise biometric identification," 
%   Expert Systems with Applications, vol. 42, no. 21, pp. 7905-7916, 2015.
% 
% 
% 
% (C)	Mohammad Haghighat, University of Miami
%       haghighat@ieee.org
%       PLEASE CITE THE ABOVE PAPER IF YOU USE THIS CODE.
% 
% Thanks to Dr. Saeed Meshgini for his helps.



if(size(data,1) ~= size(trainData,1))
   error('DATA and TRAINDATA must be in the same space with equal dimensions.');
end

if(size(trainData,2) ~= size(trainLabel,2))
   error('The length of the TRAINLABEL must be equal to the number of columns in TRAINDATA.');
end

if (~exist('options','var'))
   options.KernelType='linear';
end


% Separate samples of each class in a cell array

c = max(trainLabel);
dataCell = cell(1,c);
nSample = zeros(1,c);
for i = 1:c
    ind = find(trainLabel==i);
    nSample(i) = length(ind);
    dataCell{1,i} = trainData(:,ind);
end
clear trainLabel


% Create class-specific kernel for the training data

kTrainCell = cell(c,c);
for p = 1:c
    for q = 1:c
        Kpq = zeros(nSample(p),nSample(q));
        classP = dataCell{1,p};
        classQ = dataCell{1,q};
        for i = 1:nSample(p)
            for j = 1:nSample(q)
                Kpq(i,j) = kernel(classP(:,i),classQ(:,j),options);
            end
        end
        kTrainCell{p,q} = Kpq;
    end
end
kTrain = cell2mat(kTrainCell);
clear kTrainCell 


% Make data have zero mean

[~,n] = size(trainData);
One = (1/n) * ones(n,n);
zeroMeanKtrain = kTrain - One*kTrain - kTrain*One+One*kTrain*One;
clear trainData


% Create the block-diagonl W matrix

wTrainCell=cell(c,c);
for p = 1:c
    for q = 1:c
        if p == q
            wTrainCell{p,q}=(1/nSample(p))*ones(nSample(p),nSample(q));
        else
            wTrainCell{p,q}=zeros(nSample(p),nSample(q));
        end
    end
end
wTrain = cell2mat(wTrainCell);
clear wTrainCell


% Decompose zeroMeanKtrain using eigen-decomposition

[P, gamma] = eig(zeroMeanKtrain);
diagonal = diag(gamma);
[~,index] = sort(diagonal,'descend');
gamma = diagonal(index);
P = P(:,index);


% Remove eigenvalues with relatively small value

maxEigVal = max(abs(gamma));
zeroEigIndex = find((abs(gamma)/maxEigVal)<1e-6);
gamma(zeroEigIndex) = [];
P(:,zeroEigIndex) = [];


% Normalize eigenvectors

nEig = length(gamma);
for i = 1:nEig
    P(:,i) = P(:,i)/norm(P(:,i));
end


% Compute eigenvectors (beta) and eigenvalues (lambda)

BB = (P')*(wTrain)*(P);
[beta, lambda] = eig(BB);
diagonal = diag(lambda);
[~, index] = sort(diagonal,'descend');
lambda = diagonal(index);
beta = beta(:,index);
clear BB wTrain


% Remove eigenvalues with relatively small value

maxEigVal = max(abs(lambda));
zeroEigIndex = find((abs(lambda)/maxEigVal)<1e-6);
lambda(zeroEigIndex) = [];
beta(:,zeroEigIndex) = [];


% Compute eigenvectors (alpha) and normalize them

gamma = diag(gamma);
alpha = (P/gamma)*beta;
nEig = length(lambda);
for i = 1:nEig
    scalar = sqrt((alpha(:,i).')*(zeroMeanKtrain)*(alpha(:,i)));
    alpha(:,i)=alpha(:,i)/scalar;
end
clear zeroMeanKtrain P gamma beta


% Dimensionality reduction (if nDim is not given, nEig dimensions are retained):

if (~exist('nDim','var'))
   nDim = nEig;       % nEig is the maximum possible value (the rank of subspace)
elseif (nDim > nEig)
   warning(['Target dimensionality reduced to ' num2str(nEig) '.']);
end

w = alpha(:,1:nDim);    % Projection matrix


% Create class-specific kernel for all data points:

[~,nPrime] = size(data);
kDataCell = cell(c,1);
for p = 1:c
    Kp = zeros(nSample(p),nPrime);
    classP = dataCell{1,p};
    for i = 1:nSample(p)
        for j = 1:nPrime
            Kp(i,j) = kernel(classP(:,i),data(:,j),options);
        end
    end
    kDataCell{p,1} = Kp;
end
kData = cell2mat(kDataCell);
clear data dataCell kDataCell


% Make data zero mean

Oneprime = (1/n)*ones(n,nPrime);
zeroMeanKdata = kData - kTrain*Oneprime - One*kData+One*kTrain*Oneprime;
clear kTrain kData


% Project all data points non-linearly onto a new lower-dimensional subspace (w):

mappedData = (w.') * (zeroMeanKdata);
