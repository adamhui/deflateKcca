%%  KCCAÀ„∑®
%KCCA algorithm   
function [Alpha]= kernel_cca(Kxx,Kyy,varargin)
%trX=(N,dim_x)
%try=(N,dim_x)

%% default para  settings
p = inputParser;
def_delta =1e-6;
%def_r = min(size(trX,2),size(trY,2));
def_gama =1e-10;
%p.addOptional('r',def_r,@isnumeric);
p.addOptional('gama',def_gama,@isnumeric);
p.addOptional('delta',def_delta,@isnumeric);
p.parse(varargin{:});
gama=p.Results.gama;

%% solving eigval and eigvec
[dim_x,~] = size(Kxx);
[dim_y,~] = size(Kyy);
I_dx = eye(dim_x);
I_dy = eye(dim_y);
% n=dim_x;
% J = eye(dim_x) -ones(n,1)*ones(1,n);
% L=1/n*Kxx'*J*Kxx+gama*Kxx;
% N=1/n*Kyy'*J*Kyy+gama*Kyy;
% M=1/n*Kxx'*J*Kyy;
% MatA=(L+gama*I_dx)^-1*M*(N+gama*I_dy)^-1*M';
if exist('usegpu','var')
if usegpu
    Kxx=gpuArray(Kxx);
    Kyy=gpuArray(Kyy);
    I_dx=gpuArray(I_dx);
    I_dy=gpuArray(I_dy);
end
end
MatA = inv( Kxx   + gama * I_dx)*  Kyy  * inv ( Kyy   + gama * I_dy) * Kxx  ;
%MatA =  (Kxx*Kxx'+gama*I_dx)\( Kxx *  Kyy') * (( Kyy*Kyy'+gama*I_dy)\( Kyy * Kxx'));
MatA = max( MatA , MatA');
%MatA = ( MatA + MatA')/2;
[eigVector,eigValue] = eig (MatA);

%% Sort eigVector
eigValue = diag(eigValue);
[~, index] = sort(-abs(eigValue)); 
eigValue = eigValue(index);
eigVector = eigVector(:,index);

%% Normlization
%eigVector=eigVector./sum(eigVector.*eigVector).^0.5;

Alpha=eigVector;

if exist('usegpu','var')
if use_gpu
   Alpha = gather(Alpha); 
end
end
