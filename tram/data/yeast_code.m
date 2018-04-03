rand('state', 10000);
data = cell(1,5)

%for i = 1:5 
%data{i} = cell(1,50)
%end
data{1} = cell(1,67)
data{2} = cell(1,135)
data{3} = cell(1,75)
data{4} = cell(1,52)
data{5} = cell(1,55)

for i = 1:67 
data{1,1}{1,i} = earlyg1(i,:)
end

for i = 1:135 
data{2}{1,i} = lateg1(i,:)
end

for i = 1:75
data{3}{1,i} = s(i,:)
end

for i = 1:52
data{4}{1,i} = g2(i,:)
end

for i = 1:55
data{5}{1,i} = m(i,:)
end

train_data = {data{1}(1:57),data{2}(1:125),data{3}(1:65),data{4}(1:42),data{5}(1:45)}

test_data = {data{1}(58:67),data{2}(126:135),data{3}(66:75),data{4}(43:52),data{5}(46:55)}

nState = 5

nFeatArr = [1]

yeast_model = tramGenTrain(train_data, nState, nFeatArr, 'replicates',6,'maxiterations',10)

%no need to select genes

preds = cell(1,5)
for i = 1:5
    preds{i} = zeros(length(test_data{i}),2)
end

%compute posteriors and classify 
correct = 0
for i = 1:5
    preds{i} = predict_new(test_data{i},5,yeast_model)
    correct = correct + sum(preds{i}(:,1) == i)
end

fprintf('Accuracy of generative HMM is %2.0f%%\n',(100 * correct)/50);

% discriminative training and evaluation
disc_yeast_model = tramDiscTrain(train_data, yeast_model, 'mmieIterations', 10);


preds_disc = cell(1,5)
for i = 1:5
    preds_disc{i} = zeros(length(test_data{i}),2)
end

%compute posteriors and classify 
correct_disc = 0
for i = 1:5
    preds_disc{i} = predict_new(test_data{i},5,disc_yeast_model)
    correct_disc = correct_disc + sum(preds_disc{i}(:,1) == i)
end

fprintf('Accuracy of discriminative HMM is %2.0f%%\n',(100 * correct_disc)/50);
