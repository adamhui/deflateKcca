function [ X1, Y1, X2, Y2 ] = dataFragment( data, fragRatio )
%Data pre-treatment for dividing into [ fea1, label1, fea2, label2 ]
%  [ X1, Y1, X2, Y2 ] = fragData( data, fragRatio )
%
%  Input:
%       data -matrix with lab in last row or column
%       options -stuct options setting:
%       options.MatType: 
%                   allow  -'ND' (Sample Number * Dimension)
%                        or  -'DN' (Dimension * Sample Number)
%                default  -'ND'
%       options.FragRatio:
%                   allow  -decimal (0 , 1)
%                        or  -integer  (1 , each class number)
%                default  -0.5
%  Output:
%       trainData struct: X1 & Y1
%       testData  struct: X2 & Y2
%
%  Example:
%       fragRatio =  0.5;
%       [ trX, trY, teX, teY ] = fragData( Iris, fragRatio )
%
%  version 2.0 --Sep/2017
%
%  Written by Sixing Liu (QQ:137946009)


%% Init
if nargin < 2
    fragRatio = 0.5;
end

[~, D] = size(data);
% fea = data (:,1:D-1);
% lab = data (:,D);


%% Lab count
labCnt = tabulate(data (:,D));
class = labCnt(:, 1);
labN = size(labCnt, 1);

%% random permutation of Data
tmp = sortrows(data, D);
trainTmp  = {};
testTmp   = {};
for iLab = 1 : labN
    classNum = labCnt(iLab, 2);
    eachClass = tmp(tmp(:, D)==class(iLab), :);
    eachClass(1:size(eachClass,1), :) = eachClass( ...
        randperm(size(eachClass, 1)), :);
    % random permutation of samples in each class
    if fragRatio > 0 && fragRatio < 1
        trainNum = floor(classNum*fragRatio);
    elseif fragRatio < min(labCnt(:, 2))
        trainNum = fragRatio;
    else
        error(message('fragRatio error'))
    end
    trainClass = eachClass(1:trainNum, :);
    trainTmp{iLab} = trainClass;
    testClass = eachClass((trainNum+1) : classNum, :);
    testTmp{iLab} = testClass;
end
% Get trainData & testData, a cell consist of each class

%% trainData & testData
trainTmp = cell2mat(trainTmp');
X1 = trainTmp(:, 1:D-1);
Y1 = trainTmp(:, D);
testTmp  = cell2mat(testTmp');
X2  = testTmp(:, 1:D-1);
Y2  = testTmp(:, D);

end

