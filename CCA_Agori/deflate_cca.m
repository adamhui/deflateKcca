function [ wx,wy ] = deflate_cca( x,y )
%DEFLATE_CCA 此处显示有关此函数的摘要
%   此处显示详细说明
delta=1e-6;
e=1e-6;
[dim_x,N]=size(x);
[dim_y,N]=size(y);
Cxx=x*x';
Cyy=y*y';
Cxx=(Cxx+Cxx')/2;
Cyy=(Cyy+Cyy')/2;
Cxy=x*y';
Cxx_sqrt=(Cxx+eye(dim_x)*e)^0.5;
Cyy_sqrt=(Cyy+eye(dim_y)*e)^0.5;
Txy=Cxx_sqrt^-1*Cxy*Cyy_sqrt^-1;
Tyx=Txy';
it_count=0;
r=max(dim_x,dim_y);

r=40;
wx = zeros(dim_x,r);
wy=zeros(dim_y,r);
temp_wx_i=zeros(dim_x,1);
temp_wy_i=zeros(dim_y,1);
for i=1:r
    [U,D,V]=svd(Txy);
    u=U(:,1);
    V=V';
    v=V(:,1);
    while true
        wx_i=Cxx_sqrt^-1*Txy*v; 
        u=Cxx_sqrt*wx_i/norm(Cxx_sqrt*wx_i);
        wy_i = Cyy_sqrt^-1*Tyx*u;
        v=Cyy_sqrt*wy_i/norm(Cyy_sqrt*wy_i);
        it_count=it_count+1;
        Loss_a=norm(wx_i-temp_wx_i);
        Loss_b=norm(wy_i-temp_wy_i);
        Loss=abs(Loss_a)+ abs(Loss_b);
        if it_count>=500||Loss<delta
            wx(:,i)=wx_i;
            wy(:,i)=wy_i;
            temp_wx_i=zeros(dim_x,1);
            temp_wy_i=zeros(dim_y,1);
            it_count
            Loss
            it_count=0;
            break;
        end
        temp_wx_i=wx_i;
        temp_wy_i=wy_i;
    end
    Txy=Txy-u'*Txy*v*u*v';
end

end

