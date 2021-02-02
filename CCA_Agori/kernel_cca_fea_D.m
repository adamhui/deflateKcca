%% 基于多 投影向量的特征加权诱导KCCA算法
%KCCA algorithm with D on the feature solving via eig decomposition
function [Alpha]= kernel_cca_fea_D(Kxx,Kyy,varargin)
%trX=(N,dim_x)
%try=(N,dim_x)
%% default para  settings
p = inputParser;
def_delta =1e-6;
%def_r = min(size(trX,2),size(trY,2));
def_gama =1e-15;
%p.addOptional('r',def_r,@isnumeric);
p.addOptional('gama',def_gama,@isnumeric);
p.addOptional('delta',def_delta,@isnumeric);
p.parse(varargin{:});
%r=p.Results.r;
gama=p.Results.gama;
delta=p.Results.delta;

%% solving alpha and beta with D
[dim_x,~] = size(Kxx);
[dim_y,~] = size(Kyy);
I_dx = eye(dim_x);
I_dy = eye(dim_y);
it_count=0;
max_iteration=500;
tempNormA=0;
tempNormB=0;
Dx=I_dx;
Dy=I_dy;
temp_dxv=0;
temp_alpha=0;
tempObj=0;
while true
    % Sovle alpha and beta
    Kxx1=Dx*Kxx;
    Kyy1=Kyy;
    eigA = inv( Kxx1 + gama * I_dx)*  Kyy1 * inv ( Kyy1+ gama * I_dy) * Kxx1 ;
    eigA = max( eigA , eigA');
    eigB = inv( Kyy1+ gama * I_dy)*Kxx1  * inv ( Kxx1 + gama * I_dx) * Kyy1 ;
    eigB = max( eigB , eigB');
    A = getSortedAndNormalizedEig(eigA);
    alpha=A(:,1);
    B = getSortedAndNormalizedEig(eigB);
    beta=B(:,1);
    
    % Solve Dx
    dx=alpha'*Kxx1;
    Dx_vect=dx/norm(alpha)./(sum(Kxx1.^2,1).^0.5);
    Dx_vect= abs(Dx_vect) ;
    Dx_vect = 1./(abs(Dx_vect)+gama);
    Dx=diag(Dx_vect);
    
    % Solve Dy
    dy=beta'*Kyy1;
    Dy_vect=dy/norm(beta)./(sum(Kyy1.^2,1).^0.5);
    Dy_vect= abs(Dy_vect) ;
    Dy_vect = 1./(abs(Dy_vect)+gama);
    Dy=diag(Dy_vect);
    
    % Check convergence
    obj=norm(A'*Kxx1*Kyy1'*B);
    Loss=1/obj
    if Loss<delta||it_count>max_iteration
        Alpha=A;
        Loss
        it_count
        break;
    end
    temp_dxv=Dx_vect;
    temp_alpha=alpha;
    it_count =it_count+1;
    tempObj=obj;
end
end


