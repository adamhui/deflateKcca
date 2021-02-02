%% 基于多投影向量的样本加权诱导CCA算法
%  CCA algorithm with D on the feature solving via eig decomposition
function [Wx,Dx]= linear_cca_smp_D(X,Y,varargin)

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
temp_dxv=0;
it_count=0;
max_iteration=500;
Dx=eye(size(X,2));
while true
    X1=X*Dx;
    Y1 =Y;
    eigW = ( X1* X1'+gama*eye(size(X1,1)))\( X1 *  Y1') * (( Y1 * Y1'+gama*eye(size(Y1,1)))\( Y1* X1'));
    eigW = max( eigW , eigW');
    eigV = ( Y1* Y1'+gama*eye(size(Y1,1)))\( Y1 *  X1') * (( X1 * X1'+gama*eye(size(X1,1)))\( X1* Y1'));
    eigV = max( eigV , eigV');
    W = getSortedAndNormalizedEig(eigW);
    V = getSortedAndNormalizedEig(eigV);
    w=W(:,1);
    % Solve Dx and Dy
    Dx_vec=abs((w'*X)./(norm(w)*sum(X.*X,1).^0.5));
    Dx_vec = 1./(Dx_vec+gama); 
    Dx=diag(Dx_vec );
    Loss = abs(norm(Dx_vec)-norm(temp_dxv))
    if Loss<delta || it_count>max_iteration
        Loss
        it_count
        disp( [ 'it convergs,it_count:' num2str(it_count)]);
         Wx=W;      
        break;
    end
    it_count=it_count+1;
    temp_dxv=Dx_vec;
end
end

