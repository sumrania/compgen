function preds = predict_new(data,num_classes, model, varargin)
%TRAMPREDICT predicts the classes of testing data based on the hidden
%Markov models (HMM).
%   LOGODDS = TRAMPREDICT(DATA, MODEL) returns the log odds of each
%   instance in DATA being a positive example, based on the HMMs given by
%   MODEL. DATA{k} are the time series gene expresion of patient k.
%   MODEL{1} and MODEL{2} are the HMMs for negative and positive examples,
%   respectively. See TRAMGENTRAIN for the data structure. LOGODDS(k) are
%   the log odds of patient k being a positive example.

%lpr = log(model{2}.prior) - log(model{1}.prior);
%logOdds = zeros(length(data),1);
%for q = 1 : length(data)
%    [pStates, loglik2] = mghmmDecode(data{q}, model{2}.tr, model{2}.mu, ...
%                                     model{2}.sigma, varargin{:});
%    [pStates, loglik1] = mghmmDecode(data{q}, model{1}.tr, model{1}.mu, ...
%                                     model{1}.sigma, varargin{:});
%    logOdds(q) = loglik2 - loglik1 + lpr;
%end

preds = zeros(length(data),2)
logLiks = zeros(num_classes,1)
for  i = 1:length(data)
    for j = 1:num_classes
        [pStates, logLiks(j)] = mghmmDecode(data{i}, model{j}.tr, model{j}.mu, model{j}.sigma, varargin{:});
        logLiks(j) = logLiks(j) + log(model{j}.prior)
    end
    [argvalue, argmax] = max(logLiks)
    preds(i,1) = argmax
    preds(i,2) = argvalue
end 

    
        
       
        