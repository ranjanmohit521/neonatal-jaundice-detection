clc; close all; clear;

trainData = imageDatastore('train', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
valData   = imageDatastore('validation', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

labelCounts = countEachLabel(trainData);
totalSamples = sum(labelCounts.Count);
numClasses = height(labelCounts);
classWeights = totalSamples ./ (numClasses * labelCounts.Count);

inputSize = [224 224 3];

augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandRotation', [-15 15], ...
    'RandScale', [0.8 1.2], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10]);

augTrain = augmentedImageDatastore(inputSize, trainData, 'DataAugmentation', augmenter);
augVal   = augmentedImageDatastore(inputSize, valData);

layers = [
    imageInputLayer(inputSize, 'Normalization', 'rescale-zero-one')

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    flattenLayer
    fullyConnectedLayer(256)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer('Classes', labelCounts.Label, 'ClassWeights', classWeights)
];

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 1e-3, ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 10, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true);

net = trainNetwork(augTrain, layers, options);

save('jaundice_model.mat', 'net');

fprintf('Training complete. Model saved to jaundice_model.mat\n');

evaluate_model(net, augVal, valData.Labels);
