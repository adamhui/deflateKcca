function corruptImg = corrupt_image(imgs, corruptRate,type)
%CORRUPT_IMAGE corrupt images with 'salt & pepper' noise
%   -Input:
%       imgs: each row corresponding to a line of image
%       corruptRate: pixel rate corrupted
%   -Output:
%       corruptImg: corrupted images, without normalization

global MAT_TYPE
if isempty(MAT_TYPE)
    MAT_TYPE = 'ND';
end

if strcmp(MAT_TYPE, 'ND')
    [nSmp, nFea] = size(imgs);
    corruptImg = zeros(size(imgs));
    for iNum = 1: nSmp
        corruptImg(iNum, :) = imnoise(imgs(iNum, :), type, corruptRate);
    end
elseif strcmp(MAT_TYPE, 'DN')
    [nFea, nSmp] = size(imgs);
    imgs = imgs';
    corruptImg = zeros(size(imgs));
    for iNum = 1: nSmp
        corruptImg(iNum, :) = imnoise(imgs(iNum, :), type, corruptRate);
    end
    corruptImg = corruptImg';
else
    error(message('MAT_TYPE allow ND or DN'))
end

end

