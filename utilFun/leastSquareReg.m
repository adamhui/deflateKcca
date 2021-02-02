function [accuracy] = leastSquareReg(p,trX,teX,trY,teY_label,reduce_dim)
%最小二乘线性回归
projected_trX=p'*trX;
projected_teX=p'*teX;

accuracy=zeros(1,length(reduce_dim));
for r=1:length(reduce_dim)
    dim=reduce_dim(r);
    X=projected_trX(1:dim,:);
    X_t=projected_teX(1:dim,:);
    w=(X*X')\X*trY';
    
    if min(size(trY))==1
        predict=round(w'*X_t);
    else
        [~,predict]=max(w'*X_t,[],1);
    end
    accuracy(r)=length(find(teY_label-predict==0))/length(teY_label);
end
end

