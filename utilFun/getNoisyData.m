function [  imgNoisy ] = getNoisyData( input_img,nosieWise,corruptRate,noiseType)
%   ���ݼ���
%   nosieWise =��f�� ����������ά����   
%   nosieWise= ��s�� ����������
%   noiseType ='gaussian' ->Gaussian white noise with constant mean and variance
%                   ='localvar' ->Zero-mean Gaussian white noise with an intensity-dependent variance
%                   ='poisson' -> Poisson noise
%                   = 'salt & pepper' ->On and off pixels
%                   = 'speckle' -> Multiplicative noise






if nosieWise=='s'
    imgNoisy =corrupt_image(input_img,corruptRate,noiseType);
end
if nosieWise=='f'
    imgNoisy =corrupt_image_feature(input_img,corruptRate,noiseType);
end

end

