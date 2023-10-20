clc;clear;close all;

classNum = 10;
trainData = [];
for fi=0:2
    file = strcat("./../cifar10/train_by_Misaka10032(",num2str(fi));
    file = strcat(file,").csv");
    trainData = [trainData; readmatrix(file)];
end
trainFeatures = trainData(:,1:size(trainData,2)-1);
trainLabel = trainData(:,size(trainData,2));
clear trainData


featuresMeans = [];
for c=1:classNum
    cIdx = (trainLabel==c-1); 
    cFeatures = trainFeatures(cIdx,:);
    
    mu = mean(cFeatures); 
    featuresMeans = [featuresMeans;mu];
end

[coeff,score,latent,tsquared,explained, mu] = pca(featuresMeans);
csvwrite("cifar4_pca3_mu.csv", mu);
csvwrite("cifar4_pca3_coeff.csv", coeff);
csvwrite("cifar4_pca3_centers.csv", score);
