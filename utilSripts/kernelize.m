if  kernel_cca_sw==1||kernel_cca_fea_D_sw==1||deflate_kcca_sw==1
% X = mapminmax(X, 0, 1);
% Y = mapminmax(Y, 0, 1);
% X_test = mapminmax(X_test, 0, 1);
Kxx = mykernel( X,X,'gau'); % ºËº¯Êý
Kyy = mykernel( Y,Y ,'gau'); % ºËº¯Êý



Kxx_test = mykernel( X_test,X,'gau'); % ºËº¯Êý

if  smpNoisy_sw==1
    X_sn = mapminmax(X_sn, 0, 1);
    Y = mapminmax(Y, 0, 1);
    X_test_sn = mapminmax(X_test_sn, 0, 1);
    Kxx_sn = mykernel( X_sn,X_sn,'gau'); % ºËº¯Êý
    Kyy = mykernel( Y,Y ,'gau'); % ºËº¯Êý
    Kxx_test_sn = mykernel( X_test_sn,X_sn,'gau'); % ºËº¯Êý
end
if feaNoisy_sw==1
    X_fn = mapminmax(X_fn, 0, 1);
    Y = mapminmax(Y, 0, 1);
    X_test_fn = mapminmax(X_test_fn, 0, 1);
    Kxx_fn = mykernel( X_fn,X_fn,'gau'); % ºËº¯Êý
    Kyy = mykernel( Y,Y ,'gau'); % ºËº¯Êý
    Kxx_test_fn = mykernel( X_test_fn,X_fn,'gau'); % ºËº¯Êý
end
end
