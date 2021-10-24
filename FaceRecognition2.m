function RecFaceID = FaceRecognition2(TrainDirectory, TestDirectory) 
% Face recognition method using transfer learning with alexnet 
%    trainPath - directory that contains the given training face images
%    testPath  - directory that constains the test face images
%    outputLabel - predicted face label for all tested images 

% Retrieving images
TrainingImages = imageDatastore(TrainDirectory, 'IncludeSubFolders', true, 'LabelSource', 'folderNames');
TestImages = imageDatastore(TestDirectory, 'IncludeSubFolders', true);
% Resizing images to fit into alexnet
TrainingImages.ReadFcn = @(loc)imresize(imread(loc),[227,227]);
TestImages.ReadFcn = @(loc)imresize(imread(loc),[227,227]);

% Specifying alexnet layers and paramerers
cnn = alexnet;
layers = cnn.Layers;
fullyConnectLayer = fullyConnectedLayer(100);
classLayer = classificationLayer;
layers(23) = fullyConnectLayer;
layers(25) = classLayer; 
learningRate = 0.00001;
opts = trainingOptions("rmsprop","InitialLearnRate",learningRate,'MiniBatchSize',25, 'MaxEpochs',1000);

% transfer learning with alexnet
[trainedNet,information] = trainNetwork(TrainingImages, layers, opts);

[predictedLabels,scores] = classify(trainedNet,TestImages);
RecFaceID = char(predictedLabels);

end

