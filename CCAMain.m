%% project settings
global WORKPATH
WORKPATH = cd;
global MAT_TYPE ;
MAT_TYPE= 'DN';
 
addpath(genpath(WORKPATH));
clc;clear;
warning off;

usegpu=0;

%% cca var sw
linear_cca_sw=1;
linear_cca_smp_D_sw=0;
linear_cca_fea_D_sw=0;

deflate_cca_sw=0;
deflate_cca_smp_D_sw=0;
deflate_cca_fea_D_sw=0; 

kernel_cca_sw=1;         
kernel_cca_smp_D_sw=0;
kernel_cca_fea_D_sw=0;   

deflate_kcca_sw=0;
deflate_kcca_smp_D_sw=0;
deflate_kcca_fea_D_sw=0;

%%  noisy sw
smpNoisy_sw=0 ;
feaNoisy_sw =0; 

%% data processing
data=loadDataset('YaleB_32x32');
r=[1:200];
%r=100;
[ trX, trY, teX, teY ] = dataFragment( data, 0.6);%data (ND)
trX = trX';
trY = trY';
teX = teX';
teY = teY';
% teX=teX(:,1:size(trX,2) );
% teY=teY(:,1:size(trX,2) );
% rand_test=randperm(length(teY));
% teX=teX(:,rand_test);
% teY=teY(:,rand_test);

class_count = max(max(trY),max(teY))-min(min(trY),min(teY))+1;%类别数量
trY_vec = double(repmat((1:class_count)',1,size(trY,2))==repmat(trY,class_count,1));  

%% noise injection
corruptRate=0.05;
noiseType={'gaussian';'localvar';'poisson';'salt & pepper';'speckle'};
type=noiseType{4};
if smpNoisy_sw==1
    X_sn =getNoisyData(trX,'s',corruptRate,type);
    X_test_sn =getNoisyData(teX,'s',corruptRate,type);
end
if feaNoisy_sw==1
    X_fn =getNoisyData(trX,'f',corruptRate,type);
    X_test_fn =getNoisyData(teX,'f',corruptRate,type);
end

%% data normalization (compute the centred data)
data_normalization;


% X = trX;
% Y =trY_vec;
% % trY_vec= trY;
% X_test=teX;

%% kernel matrix
kernelize;

%% linear cca
if linear_cca_sw==1
    wx=linear_cca(X, Y);
    %[cor_linear_cca]=leastSquareReg(wx,X,X_test,trY_vec,teY,r);
    [linear_cca_cor] = myClassify(wx, X', X_test', trY', teY', r);
    [accuracy_linear_cca, boundary_linear_cca] = max(linear_cca_cor);
    if smpNoisy_sw==1
        wx=linear_cca (X_sn, Y);
        [cor] = myClassify(wx, X_sn', X_test_sn', trY', teY', r);
        [accuracy, boundary] = max(cor);
        linear_cca_cor=[linear_cca_cor;cor];
        accuracy_linear_cca =[accuracy_linear_cca;accuracy];
        boundary_linear_cca =[boundary_linear_cca;boundary];
    end
    if feaNoisy_sw==1
        wx=linear_cca (X_fn, Y);
        [cor] = myClassify(wx, X_fn', X_test_fn', trY', teY', r);
        [accuracy, boundary]  = max(cor);
        linear_cca_cor=[linear_cca_cor;cor];
        accuracy_linear_cca =[accuracy_linear_cca;accuracy];
        boundary_linear_cca=[boundary_linear_cca;boundary];
    end
end

%% linear cca with sample-wise  D
if linear_cca_smp_D_sw==1
    [wx,Dx]=linear_cca_smp_D(X, Y);
    %     [wx,~,Dx]=CAFJSS(trX, trY_vec);
    dim= size(X,1);
    [linear_cca_smp_D_cor] = myClassify(wx, trX', teX', trY', teY', dim);
    [accuracy_linear_cca_smp_D, boundary_linear_cca_smp_D] = max(linear_cca_smp_D_cor);
    if smpNoisy_sw==1
        [wx,Dx]=linear_cca_smp_D(X_sn, Y);
        dim= size(X,1);
        [cor] = myClassify(wx, X_sn', X_test_sn', trY', teY', dim);
        [accuracy, boundary] = max(cor);
        linear_cca_smp_D_cor=[linear_cca_smp_D_cor;cor];
        accuracy_linear_cca_smp_D=[accuracy_linear_cca_smp_D;accuracy];
        boundary_linear_cca_smp_D=[boundary_linear_cca_smp_D;boundary];
    end
    if feaNoisy_sw==1
        [wx,Dx]=linear_cca_smp_D(X_fn, Y);
        dim= size(X,1);
        [cor] = myClassify(wx, X_fn', X_test_fn', trY', teY', dim);
        [accuracy, boundary]  = max(cor);
        linear_cca_smp_D_cor=[linear_cca_smp_D_cor;cor];
        accuracy_linear_cca_smp_D=[accuracy_linear_cca_smp_D;accuracy];
        boundary_linear_cca_smp_D=[boundary_linear_cca_smp_D;boundary];
    end
end

%% linear cca with feature-wise  D
if linear_cca_fea_D_sw==1
    wx=linear_cca_fea_D(trX, trY_vec);
    dim= size(trX,1);
    [linear_cca_fea_D_cor] = myClassify(wx, trX', teX', trY', teY', dim);
    [accuracy_linear_cca_fea_D, boundary_linear_cca_fea_D] = max(linear_cca_fea_D_cor);
    if smpNoisy_sw==1
        wx=linear_cca_fea_D(X_sn, Y);
        dim= size(trX,1);
        [cor] = myClassify(wx, X_sn', X_test_sn', trY', teY', dim);
        [accuracy, boundary] = max(cor);
        linear_cca_fea_D_cor=[linear_cca_fea_D_cor;cor];
        accuracy_linear_cca_fea_D=[accuracy_linear_cca_fea_D;accuracy];
        boundary_linear_cca_fea_D=[boundary_linear_cca_fea_D;boundary];
    end
    if feaNoisy_sw==1
        wx=linear_cca_fea_D(X_fn, Y);
        dim= size(trX,1);
        [cor] = myClassify(wx, X_fn', X_test_fn', trY', teY', dim);
        [accuracy, boundary]  = max(cor);
        linear_cca_fea_D_cor=[linear_cca_fea_D_cor;cor];
        accuracy_linear_cca_fea_D=[accuracy_linear_cca_fea_D;accuracy];
        boundary_linear_cca_fea_D=[boundary_linear_cca_fea_D;boundary];
    end
end

%% kernel cca
if kernel_cca_sw==1
    Alpha = kernel_cca(Kxx,Kyy);
%     [Alpha, Beta, rc, Kx_c, Ky_c]=kcanonca_reg_ver2(Kxx,Kyy,1e-12,1e-10);
%     Alpha=fliplr(Alpha);
    %[cor_kcca]=leastSquareReg(Alpha,Kxx,Kxx_test',trY_vec,teY,r); 
    [kcca_cor] = myClassify(Alpha, Kxx, Kxx_test, trY', teY', r);
    [kcca_accuracy, kcca_boundary] = max(kcca_cor);
    if smpNoisy_sw==1
        [Alpha_sn] = kernel_cca(Kxx_sn,Kyy);
        [kcca_cor_sn] = myClassify(Alpha_sn, Kxx_sn, Kxx_test_sn, trY', teY', r);
        [kcca_accuracy_sn, kcca_boundary_sn] = max(kcca_cor_sn);
        kcca_cor=[kcca_cor;kcca_cor_sn];
        kcca_accuracy=[kcca_accuracy;kcca_accuracy_sn];
    end
    if feaNoisy_sw==1
        [Alpha_fn] = kernel_cca(Kxx_fn,Kyy);
        [kcca_cor_fn] = myClassify(Alpha_fn, Kxx_fn, Kxx_test_fn, trY', teY', r);
        [kcca_accuracy_fn, kcca_boundary_fn] = max(kcca_cor_fn);
        kcca_cor=[kcca_cor;kcca_cor_fn];
        kcca_accuracy=[kcca_accuracy;kcca_accuracy_fn];
    end
end

%% kernel cca with feature-wise  D
if  kernel_cca_fea_D_sw==1
    [Alpha_fd] = kernel_cca_fea_D(Kxx,Kyy);
    [kcca_f_d_cor] = myClassify(Alpha_fd, Kxx, Kxx_test, trY', teY', r);
    [kcca_f_d_accuracy, kcca_f_d_boundary] = max(kcca_f_d_cor);
    if smpNoisy_sw==1
        [Alpha_fd_sn] = kernel_cca_fea_D(Kxx_sn,Kyy);
        [kcca_f_d_cor_sn] = myClassify(Alpha_fd_sn, Kxx_sn, Kxx_test_sn, trY', teY', r);
        [kcca_f_d_accuracy_sn, kcca_f_d_boundary_sn] = max(kcca_f_d_cor_sn);
        kcca_f_d_cor=[kcca_f_d_cor;kcca_f_d_cor_sn];
        kcca_f_d_accuracy=[kcca_f_d_accuracy;kcca_f_d_accuracy_sn];
    end
    if feaNoisy_sw==1
        [Alpha_fd_fn] = kernel_cca_fea_D(Kxx_fn,Kyy);
        [kcca_f_d_cor_fn] = myClassify(Alpha_fd_fn, Kxx_fn, Kxx_test_fn, trY', teY', r);
        [kcca_f_d_accuracy_fn, kcca_f_d_boundary_fn] = max(kcca_f_d_cor_fn);
        kcca_f_d_cor=[kcca_f_d_cor;kcca_f_d_cor_fn];
        kcca_f_d_accuracy=[kcca_f_d_accuracy;kcca_f_d_accuracy_fn];
    end
end

%% deflate linear cca
if deflate_cca_sw==1
    wx=deflate_cca(X, Y);
    [cor_deflate_cca] = myClassify(wx, X', X_test', trY', teY', r);
    [accuracy_deflate_linear_cca, boundary_deflate_linear_cca] = max(cor_deflate_cca);
    if smpNoisy_sw==1
        wx=deflate_cca (X_sn, Y);
        dim= size(trX,1);
        [cor] = myClassify(wx, X_sn', X_test_sn', trY', teY', dim);
        [accuracy, boundary] = max(cor);
        cor_deflate_cca=[cor_deflate_cca;cor];
        accuracy_deflate_linear_cca =[accuracy_deflate_linear_cca;accuracy];
        boundary_deflate_linear_cca =[boundary_deflate_linear_cca;boundary];
    end
    if feaNoisy_sw==1
        wx=linear_cca (X_fn, Y);
        dim= size(trX,1);
        [cor] = myClassify(wx, X_fn', X_test_fn', trY', teY', dim);
        [accuracy, boundary]  = max(cor);
        cor_deflate_cca=[cor_deflate_cca;cor];
        accuracy_deflate_linear_cca =[accuracy_deflate_linear_cca;accuracy];
        boundary_deflate_linear_cca=[boundary_deflate_linear_cca;boundary];
    end
end

%% deflate linear cca sample-wise  D
if deflate_cca_smp_D_sw==1
    
end

%% deflate linear cca feature-wise  D
if deflate_cca_fea_D_sw==1
    wx=deflate_cca_fea_D(X, Y);
    [cor_deflate_cca_fd] = myClassify(wx, X', X_test', trY', teY', r);
    [accuracy_deflate_cca_fd, boundary_deflate_cca_fd] = max(cor_deflate_cca_fd);
    if smpNoisy_sw==1
        wx=deflate_cca_fea_D (X_sn, Y);
        [cor] = myClassify(wx, X_sn', X_test_sn', trY', teY', r);
        [accuracy, boundary] = max(cor);
        cor_deflate_cca_fd=[cor_deflate_cca_fd;cor];
        accuracy_deflate_cca_fd =[accuracy_deflate_cca_fd;accuracy];
        boundary_deflate_cca_fd =[boundary_deflate_cca_fd;boundary];
    end
    if feaNoisy_sw==1
        wx=linear_cca (X_fn, Y);
        [cor] = myClassify(wx, X_fn', X_test_fn', trY', teY', r);
        [accuracy, boundary]  = max(cor);
        cor_deflate_cca_fd=[cor_deflate_cca_fd;cor];
        accuracy_deflate_cca_fd =[accuracy_deflate_cca_fd;accuracy];
        boundary_deflate_cca_fd=[boundary_deflate_cca_fd;boundary];
    end
end

%% deflate kcca
if deflate_kcca_sw==1
    Alpha = deflate_kcca2(Kxx,Kyy);
    [deflate_kcca_cor] = myClassify(Alpha, Kxx, Kxx_test, trY', teY', r);
    [deflate_kcca_accuracy,  deflate_kcca_boundary] = max( deflate_kcca_cor);
    if smpNoisy_sw==1
        [Alpha_sn] = kernel_cca(Kxx_sn,Kyy);
        [kcca_cor_sn] = myClassify(Alpha_sn, Kxx_sn, Kxx_test_sn, trY', teY', r);
        [kcca_accuracy_sn, kcca_boundary_sn] = max(kcca_cor_sn);
        kcca_cor=[kcca_cor;kcca_cor_sn];
        kcca_accuracy=[kcca_accuracy;kcca_accuracy_sn];
    end
    if feaNoisy_sw==1
        [Alpha_fn] = kernel_cca(Kxx_fn,Kyy);
        [kcca_cor_fn] = myClassify(Alpha_fn, Kxx_fn, Kxx_test_fn, trY', teY', r);
        [kcca_accuracy_fn, kcca_boundary_fn] = max(kcca_cor_fn);
        kcca_cor=[kcca_cor;kcca_cor_fn];
        kcca_accuracy=[kcca_accuracy;kcca_accuracy_fn];
    end
end

%% deflate kcca sample-wise  D
if deflate_kcca_smp_D_sw==1
    
end

%% deflate linear cca feature-wise  D
if deflate_kcca_fea_D_sw==1
    
end
