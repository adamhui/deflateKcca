function [ wx,wy ] = deflate_cca_fea_D( x,y,varargin )
%DEFLATE_CCA 此处显示有关此函数的摘要
%   此处显示详细说明

%% default para  settings
p = inputParser;
def_gama =1e-9;
def_delta=1e-6;
p.addOptional('gama',def_gama,@isnumeric);
p.addOptional('delta',def_delta,@isnumeric);
p.parse(varargin{:});
gama=p.Results.gama;
delta = p.Results.delta;


e=gama;
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
wx = zeros(dim_x,r);
wy=zeros(dim_y,r);
temp_wx_i=zeros(dim_x,1);
temp_wy_i=zeros(dim_y,1);



for i=1:240
    [U,D,V]=svd(Txy);
    u=U(:,1);
    V=V';
    v=V(:,1);
    Dx=eye(size(x,1));
    Dy=eye(size(y,1));
    while true
        wx_i=Dx^-1*Cxx_sqrt^-1*Txy*v; 
        u=Cxx_sqrt*Dx*wx_i/norm(Cxx_sqrt*Dx*wx_i);
        wy_i =Dy^-1* Cyy_sqrt^-1*Tyx*u;
        v=Cyy_sqrt*wy_i/norm(Cyy_sqrt*wy_i);
        it_count=it_count+1;
        Loss_a=norm(wx_i-temp_wx_i);
        Loss_b=norm(wy_i-temp_wy_i);
        Loss=abs(Loss_a)
        
%         temp_feature_x = wx_i'*x;
%         Dx_vec=abs((x*temp_feature_x')./(norm(temp_feature_x)*sum(x.*x,2).^0.5));
%         %Dx_vec = 1./(Dx_vec+gama);
%         Dx=diag(Dx_vec);
        
        if it_count>=100||Loss<delta
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

