function [ alpha,beta ] = deflate_kcca( Kx,Ky )
%DEFLATE_KCCA 此处显示有关此函数的摘要
%   此处显示详细说明
N=size(Kx,1);
delta=1e-5;

Kxx=Kx*Kx;
Kyy=Ky*Ky;
Kxy=Kx*Ky;
Kxx_sqrt=((Kxx+ Kxx')/2+eye(N)*1e-10)^0.5;
Kyy_sqrt=((Kyy+Kyy')/2+eye(N)*1e-10)^0.5;
Txy=Kxx_sqrt^-1*Kxy*Kyy_sqrt^-1;
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
        alpha_i=Kxx_sqrt^-1*Txy*v;
        u=Kxx_sqrt*alpha_i/norm(Kxx_sqrt*alpha_i);
        beta_i = Kyy_sqrt^-1*Tyx*u;
        v=Kyy_sqrt*beta_i/norm(Kyy_sqrt*beta_i);
        it_count=it_count+1;
        Loss_a=norm(alpha_i-temp_alpha_i);
        Loss_b=norm(beta_i-temp_beta_i);
        Loss=Loss_a;
        if it_count>=500||Loss<delta
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

