tic
rand('state', 10000);
% load simulated data (already log-transformed)
simData = load('sim.mat');
data = simData.data;
% only use 7 of the 8 time points
for m = 1 : 2
    for k = 1 : length(data{m}), data{m}{k} = data{m}{k}(:, 1:7); end
end
% the first fold of 4-fold cross validation
trainData = {data{1}(13:50), data{2}(13:50)};
testData = {data{1}(1:12), data{2}(1:12)};
% generative training
nState = 2;
nFeatArr = [100 20 5];
model = tramGenTrain(trainData, nState, nFeatArr, 'replicates',3,'maxiterations',10);
% select genes
fsTrainData = selectFeature(trainData, model{1}.selGenes);
fsTestData  = selectFeature(testData , model{1}.selGenes);


preds = cell(1,2)
for i = 1:2
    preds{i} = zeros(length(fsTestData{i}),2)
end

%compute posteriors and classify 
correct = 0
for i = 1:2
    preds{i} = predict_new(fsTestData{i},2,model)
    correct = correct + sum(preds{i}(:,1) == i)
end

fprintf('Accuracy of generative HMM is %2.0f%%\n',(100 * correct)/24);
