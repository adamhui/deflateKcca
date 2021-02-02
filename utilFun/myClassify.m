function [ cor ] = myClassify( ...
    w_x,TRAIN, sample, group, label, reduceDim)
%MYCLASSIFY call 1NN classifier
%[ cor ] = myClassify( ...
%    eigvector, trainData, testData, reduceDim)
%  Input:
%       eigvector
%       trainInData (struct): TRAIN & predict
%       testInData  (struct): sample  &  label
%       rdDim:  reduced dimension

%% Init
global workpath

try
    addpath(genpath(workpath));
catch
    disp('Please set workpath for project')
end

if exist('usegpu','var')
    if usegpu &&reduceDim>200
        w_x=gpuArray(w_x);
        TRAIN=gpuArray(TRAIN);
        sample=gpuArray(sample);
    end
end

%% Call 1NN classifier
cor = zeros(1,length(reduceDim));
projected_train_data=w_x'*TRAIN';
projected_test_data=w_x'*sample';

if exist('usegpu','var')
    if usegpu &&reduceDim>200
        projected_train_data=gather(projected_train_data);
        projected_test_data=gather(projected_test_data);   
    end
end


for iDNum = reduceDim
        tmpTrain = projected_train_data(1:iDNum,:);
        tmpTest  = projected_test_data(1:iDNum,:); 
        class = knnclassify(tmpTest', tmpTrain', group);
        accuracy = length(find(class-label==0))/length(label);
        %cor(iDNum)=accuracy;
        cor(find(iDNum==reduceDim))=accuracy;
end
         
end



