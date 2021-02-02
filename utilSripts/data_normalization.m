% trY = mapminmax(trY, 0, 1);
% trY_vec = mapminmax(trY_vec, 0, 1);
if(max(trX)>1)
%     trX = mapminmax(trX, 0, 1);
%     teX = mapminmax(teX, 0, 1);
    trX=trX/255;
    teX=teX/255;

end

X = (trX - mean(trX,2));
%Y = (trY_vec - mean(trY_vec,2));
Y= (trY - mean(trY,2)); 
X_test=(teX - mean(teX,2));
if smpNoisy_sw==1
    if(max(trX)>1)
    X_sn=X_sn/255;
    X_test_sn=X_test_sn/255;
    end
X_sn =(X_sn - mean(X_sn,2));
X_test_sn =(X_test_sn - mean(X_test_sn,2));
end
if feaNoisy_sw==1
     if(max(trX)>1)
        X_fn=X_fn/255;
        X_test_fn=X_test_fn/255;
    end
X_fn =(X_fn - mean(X_fn,2));
X_test_fn =(X_test_fn - mean(X_test_fn,2));
end 

