%% 基于多投影向量的特征加权诱导CCA算法
%  CCA algorithm with D on the feature solving via eig decomposition
function [Wx]= linear_cca(trX,trY,varargin)

%% default para  settings
p = inputParser;
def_gama =1e-10;
p.addOptional('gama',def_gama,@isnumeric);
p.parse(varargin{:});
gama=p.Results.gama;

X = trX;
Y = trY; 

%% solving eigval and eigvec
eigA = ( X * X'+gama*eye(size(X,1)))\( X *  Y') * (( Y * Y'+gama*eye(size(Y,1)))\( Y * X'));
eigA = ( eigA + eigA')/2;
[eigVector,eigValue] = eig (eigA);

%% Sort eigVector
eigValue = diag(eigValue);
[~, index] = sort(-abs(eigValue));
eigValue = eigValue(index);
eigVector = eigVector(:,index);

%% Normlization
for tmp = 1:size(eigVector,2)
    eigVector(:,tmp) = eigVector(:,tmp) ./ norm(eigVector(:,tmp));
end
Wx=eigVector;


