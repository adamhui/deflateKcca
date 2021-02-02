function [ KerX ] = mykernel( X, varargin )
%MYKERNEL
%   Objective Function:
%       -
%   Input:
%       -
%   Output
%       -
%   Reference:
%       
%   Author:

%% User defined
global WorkPath
global MatType

if isempty(WorkPath)
    WorkPath = cd;
end
try
    addpath(genpath(WorkPath));
catch
    disp('Please set WorkPath')
end
if isempty(MatType)
    MatType = 'ND';
end
switch MatType
    case 'DN'
        X = X';   
end
Y=X;
type = 'rbf';

%%
if isempty(varargin)==0
if ismatrix(varargin{1})
    Y = varargin{1}; 
end
if ischar(varargin{2})~=0
% KernelType is provided
    type = varargin{2};
    if ischar(type)
        methods = {'linear'; 'polynomial'; 'rbf'; 'sigmoid'; ...
            'gaussian'; 'exponential'; 'laplacian'; 'multiquadric'; ...
            'inverse multiquadric'; 'triangular'; 'generalized T-Student'};
        i = find(strncmpi(type, methods, length(type)));
        if length(i) > 1
            error(message('mykernel:AmbiguousType', type));
        elseif isempty(i)
            error(message('mykernel:UnrecognizedType', type));
        else
            type = methods{i}(1:3);
        end
    end
end
end      


switch type
    case 'lin'
        KerX = X'*Y;        
    case 'pol'
        if nargin<=3
            a = 0.1;
            b = 0.1;
            c = 2;
        else
            a = varargin{end-2};
            b = varargin{end-1};
            c = varargin{end};
        end
        KerX = (a*(X'*X)+b).^c; 
    case 'rbf'
        if nargin <= 3
            sigma = 1;
        else
            sigma = varargin{end};
        end
        rbfKernel = @(X, Y) exp(-sigma .* pdist2(X, Y,'euclidean').^2);
        KerX = rbfKernel(X', Y');
    case 'sig'
        if nargin <=3
            a = 0.1;
            b = 0.1;
        else
            a = varargin{end-1};
            b = varargin{end};
        end
        KerX=tanh(a*(X'*Y)+b);
    case 'gau'
        if nargin <= 3
            sigma =1.9;
        else
            sigma = varargin{end};
        end
        Gaussian = @(X, Y) exp(- pdist2(X, Y,'euclidean').^2/(2*sigma*sigma));
        KerX = Gaussian(X', Y');
    case 'exp'
        if nargin <= 3
            sigma = 1; 
        else
            sigma = varargin{end};
        end
        Exponential = @(X,Y) exp(- pdist2(X,Y,'euclidean').^1/(2*sigma*sigma));
        KerX = Exponential(X', Y');
    case 'lap'
        if nargin <= 3
            sigma = 1;
        else
            sigma = varargin{end};
        end
        Laplacian = @(X,Y) exp(- pdist2(X,Y,'euclidean').^1/(sigma));
        KerX =  Laplacian(X', Y');
    case 'mul'
        if nargin <= 3
            b = 0.1;
        else
            b = varargin{end};
        end
        Multiquadric = @(X,Y)(pdist2(X,Y,'euclidean').^2+b*b).^0.5;
        KerX =  Multiquadric(X', Y');
    case 'inv'
        if nargin <= 3
            c = 2;
        else
            c = varargin{end};
        end
        InvMultiquadric = @(X,Y)(pdist2(X,Y,'euclidean').^2+c*c).^(-0.5);
        KerX =  InvMultiquadric(X', Y');
    case 'tri'
        if nargin <= 3
            c = 2;
        else
            c = varargin{end};
        end
        Triangular = @(X,Y)(-pdist2(X,Y,'euclidean').^c);
        KerX =  Triangular(X', Y');
    case 'gen'
        if nargin <= 3
            c = 2;
        else
            c = varargin{end};
        end
        Generalized = @(X,Y)(1./(1+pdist2(X,Y,'euclidean').^c));
        KerX =  Generalized(X', Y');
end
