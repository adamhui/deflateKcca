%% 基于多投影向量的特征加权诱导CCA算法
%  CCA algorithm with D on the feature solving via eig decomposition
function [Wx]= linear_cca_fea_D(X,Y,varargin)

%% default para  settings
p = inputParser;
def_gama =1e-3;
def_delta=1e-6;
p.addOptional('gama',def_gama,@isnumeric);
p.addOptional('delta',def_delta,@isnumeric);
p.parse(varargin{:});
gama=p.Results.gama;
delta = p.Results.delta;

%% solving eigval and eigvec
it_count=0;
max_iteration=500;
Dx=eye(size(X,1));
Dy=eye(size(Y,1));
temp_dxv=0;
while true
    X1=Dx *X;
    Y1=Y;
    eigW = ( X1* X1'+gama*eye(size(X,1)))\( X1 *  Y1') * (( Y1 * Y1'+gama*eye(size(Y,1)))\( Y1 * X1'));
    eigW = max( eigW , eigW');
    W = getSortedAndNormalizedEig(eigW);
    w=W(:,1);
    temp_feature_x = w'*X;
    Dx_vec=abs((X*temp_feature_x')./(norm(temp_feature_x)*sum(X.*X,2).^0.5));
    Dx_vec = 1./(Dx_vec+gama);
    Dx=diag(Dx_vec );
 %% loss calculation      
    Loss =abs(norm(Dx_vec)-norm(temp_dxv))
    if Loss<delta || it_count>max_iteration
%         Loss
%         it_count
        disp( [ 'it convergs,it_count:' num2str(it_count)]);
        Wx=W;
        break;
    end
    it_count=it_count+1;
    temp_dxv=Dx_vec;
end
end

