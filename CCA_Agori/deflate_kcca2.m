function [ alpha,beta ] = deflate_kcca2( Kx,Ky )
%DEFLATE_KCCA 此处显示有关此函数的摘要
%   此处显示详细说明
N=size(Kx,1);
delta=1e-6 ;
gama=1e-10;
reg=eye(N)*gama;
Kxx=Kx*Kx;
Kyy=Ky*Ky;
Kxy=Kx*Ky;
Px=Kx*(Kxx+reg)^-1*Kx;
Py=Ky*(Kyy+reg)^-1*Ky;
Txy=Px*Py;
Tyx=Txy';

it_count=0;
r=N;

alpha = zeros(N,r);
beta=zeros(N,r);
temp_alpha_i=zeros(N,1);
temp_beta_i=zeros(N,1);
for i=1:r
    [U,D,V]=svd(Txy);
    u=U(:,1);
    V=V';
    v=V(:,1);
    while true
        alpha_i=(Kxx+reg)^-1*Kx*Txy*v;
        u=Kx*alpha_i/norm(Kx*alpha_i);
        beta_i = (Kyy+reg)^-1*Ky*Tyx*u;
        v=Ky*beta_i/norm(Ky*beta_i);
        it_count=it_count+1;
        Loss_a=norm(alpha_i-temp_alpha_i);
        Loss_b=norm(beta_i-temp_beta_i);
        Loss=abs(Loss_a);
        if it_count>=200||Loss<delta
            alpha(:,i)=alpha_i;
            beta(:,i)=beta_i;
            temp_alpha_i=zeros(N,1);
            temp_beta_i=zeros(N,1);
            it_count
            Loss
            it_count=0;            
            break;
        end
        temp_alpha_i=alpha_i;
        temp_beta_i=beta_i;
    end
    Txy=Txy-u'*Txy*v*u*v';
end

end

