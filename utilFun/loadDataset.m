function [ data ]  = loadDataset(name,varargin)
data=load(name);
switch name
    case 'AR'
        data=data.AR;
        data = data';
    case 'HAPT'
        data=data.all;
        data = data';
    case 'Iris'
        data=data.Iris;
    case 'ORL_32x32'
        data = data.ORL_32x32;
    case 'ORL_64x64'
        data = [data.fea,data.gnd];
    case 'USPS'
        data=data.USPS;
    case 'Yale_32x32'
        data = [data.fea,data.gnd];
    case 'YaleB_32x32'
        data =data.YaleB_32x32;
    case 'Mnist'
        data=[data.data;data.label]';
        data=double(data);
    case 'housing'
    case 'Bodyfat'
    case 'CNAE_9'
        
end