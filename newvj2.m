clear; close all; clc;
rng(1); 
flacFolder   = 'C:\Users\S.K.Vijay\Downloads\LA\LA\ASVspoof2019_LA_train\flac\';
protocolFile = 'C:\Users\S.K.Vijay\Downloads\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt';
testFolder   = 'C:\Users\S.K.Vijay\Downloads\checking\';
fid = fopen(protocolFile,'r');
data = textscan(fid,'%s %s %s %s %s');
fclose(fid);

fileNames = data{2};
labelsRaw = data{5};

for i = 1:numel(labelsRaw)
    if strcmpi(labelsRaw{i},'bonafide')
        labelsRaw{i} = 'real';
    else
        labelsRaw{i} = 'fake';
    end
end
realIdx = find(strcmp(labelsRaw,'real'));
fakeIdx = find(strcmp(labelsRaw,'fake'));

numReal = min(100,numel(realIdx));
numFake = min(100,numel(fakeIdx));

fileNames = [fileNames(realIdx(1:numReal)); ...
             fileNames(fakeIdx(1:numFake))];

labels = categorical([labelsRaw(realIdx(1:numReal)); ...
                       labelsRaw(fakeIdx(1:numFake))]);
fsTarget = 16000;
segmentDuration = 2;     
numCoeffs = 13;          
maxFrames = 140;         
X = [];
Y = [];

disp('Extracting MFCC features...');

for i = 1:numel(fileNames)

    filePath = fullfile(flacFolder,[fileNames{i} '.flac']);
    if ~isfile(filePath)
        continue;
    end

    [audio, fs] = audioread(filePath);
    audio = mean(audio,2);

    if fs ~= fsTarget
        audio = resample(audio,fsTarget,fs);
    end

    len = fsTarget * segmentDuration;
    audio = audio(1:min(end,len));
    audio(end+1:len) = 0;

    mf = mfcc(audio,fsTarget,'NumCoeffs',numCoeffs)';
    
    mfccImg = zeros(numCoeffs,maxFrames);
    numC = min(numCoeffs,size(mf,1));
    numF = min(maxFrames,size(mf,2));
    mfccImg(1:numC,1:numF) = mf(1:numC,1:numF);

    X(:,:,:,end+1) = mfccImg;
    Y(end+1) = labels(i);
end

Y = categorical(Y);
N = numel(Y);
idx = randperm(N);

numTrain = round(0.8 * N);
trainIdx = idx(1:numTrain);
valIdx   = idx(numTrain+1:end);

XTrain = X(:,:,:,trainIdx);
YTrain = Y(trainIdx);

XVal = X(:,:,:,valIdx);
YVal = Y(valIdx);
layers = [
    imageInputLayer([numCoeffs maxFrames 1],'Normalization','zscore')

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    averagePooling2dLayer([6 6])
    flattenLayer

    bilstmLayer(64,'OutputMode','last')

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',8, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XVal,YVal}, ...
    'Plots','training-progress', ...
    'Verbose',true);
disp('Training CNN + BiLSTM Network...');
net = trainNetwork(XTrain,YTrain,layers,options);
[YValPred,valScores] = classify(net,XVal);
valAccuracy = mean(YValPred == YVal);
fprintf('\nValidation Accuracy: %.2f%%\n',valAccuracy*100);

figure;
confusionchart(YVal,YValPred);
title('Validation Confusion Matrix');
[fp,tp,~,auc] = perfcurve(YVal,valScores(:,2),'real');

figure;
plot(fp,tp,'LineWidth',2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ' num2str(auc,'%.3f') ')']);
grid on;

fnr = 1 - tp;
eer = fp(find(abs(fp - fnr) == min(abs(fp - fnr)),1));
fprintf('EER: %.4f\n',eer);
wavFiles  = dir(fullfile(testFolder,'*.wav'));
flacFiles = dir(fullfile(testFolder,'*.flac'));
testFiles = [wavFiles; flacFiles];

disp('Testing on unseen data...');

for i = 1:numel(testFiles)

    [audio, fs] = audioread(fullfile(testFolder,testFiles(i).name));
    audio = mean(audio,2);

    if fs ~= fsTarget
        audio = resample(audio,fsTarget,fs);
    end

    audio = audio(1:min(end,len));
    audio(end+1:len) = 0;

    mf = mfcc(audio,fsTarget,'NumCoeffs',numCoeffs)';
    mfccImg = zeros(numCoeffs,maxFrames);

    numC = min(numCoeffs,size(mf,1));
    numF = min(maxFrames,size(mf,2));
    mfccImg(1:numC,1:numF) = mf(1:numC,1:numF);

    input = reshape(mfccImg,[numCoeffs maxFrames 1 1]);
    [pred,score] = classify(net,input);

    fprintf('%s -> %s (%.2f)\n', ...
        testFiles(i).name, string(pred), max(score));
end

disp('ALL OUTPUTS GENERATED SUCCESSFULLY ✅');
