% Train a CNN for Open_Eyes / Closed_Eyes classification.
% Expected folders:
% dataset/Closed_Eyes/ and dataset/Open_Eyes/
clear; clc; close all;

datasetFolder = fullfile(pwd, "dataset");
if ~isfolder(datasetFolder)
    error("Create a dataset folder with Open_Eyes and Closed_Eyes subfolders first.");
end

imds = imageDatastore(datasetFolder, "IncludeSubfolders", true, "LabelSource", "foldernames");
disp(countEachLabel(imds));
[inputTrain, inputVal] = splitEachLabel(imds, 0.8, "randomized");
inputSize = [64 64 3];
augTrain = augmentedImageDatastore(inputSize, inputTrain, "ColorPreprocessing", "gray2rgb");
augVal = augmentedImageDatastore(inputSize, inputVal, "ColorPreprocessing", "gray2rgb");
numClasses = numel(categories(inputTrain.Labels));

layers = [
    imageInputLayer(inputSize, "Normalization", "zerocenter")
    convolution2dLayer(3, 16, "Padding", "same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, "Stride", 2)
    convolution2dLayer(3, 32, "Padding", "same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, "Stride", 2)
    convolution2dLayer(3, 64, "Padding", "same")
    reluLayer
    globalAveragePooling2dLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

opts = trainingOptions("adam", "MaxEpochs", 10, "MiniBatchSize", 32, ...
    "Shuffle", "every-epoch", "ValidationData", augVal, ...
    "ValidationFrequency", 10, "Verbose", true, "Plots", "training-progress");

trainedDrowsinessNet = trainNetwork(augTrain, layers, opts);
save("trainedDrowsinessNet.mat", "trainedDrowsinessNet", "inputSize");
disp("Saved trainedDrowsinessNet.mat");
