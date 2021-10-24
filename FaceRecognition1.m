function RecFaceID = FaceRecognition1(TrainDirectory, TestDirectory)
% Face recognition method using HoG features and a SVM
%    trainPath - directory that contains the given training face images
%    testPath  - directory that constains the test face images
%    outputLabel - predicted face label for all tested images 

% Retrieving training images and labels
folderNames=ls(TrainDirectory);
trainImgSet=zeros(600,600,3,length(folderNames)-2);
labelImgSet=folderNames(3:end,:);

newImageSize = [75 75];
sampleImage = imresize(trainImgSet(:,:,:,1), newImageSize);
cellSize = [4 4];
blockSize = [8 8];
numBins = 11;
blockOverlap = ceil(blockSize/1.2);
[hog2x2, vis2x2] = extractHOGFeatures(sampleImage,'CellSize',cellSize, 'BlockSize', blockSize, 'BlockOverlap', blockOverlap, 'NumBins', numBins);
hogFeatureSize = length(hog2x2);
numImages = size(folderNames,1) - 2;

% To store HoG features of training images
trainingHOGFeatures = zeros(numImages, hogFeatureSize, 'single'); 

% Extracting HoG features of training images
for i=3:length(folderNames)
    imgName=ls([TrainDirectory, folderNames(i,:),'\*.jpg']);
    img = imread([TrainDirectory, folderNames(i,:), '\', imgName]);
    img = rgb2gray(uint8(img));
    img = imresize(img, newImageSize);
    trainingHOGFeatures(i-2, :)= extractHOGFeatures(img, 'CellSize', cellSize, 'BlockSize', blockSize, 'BlockOverlap', blockOverlap, 'NumBins', numBins);
end

% Training SVM model with HoG features as input
SVMmodel = fitcecoc(trainingHOGFeatures, labelImgSet, 'Coding', 'onevsall');

testImgNames=ls([TestDirectory,'*.jpg']);
testFeatures = zeros(size(testImgNames,1), hogFeatureSize, 'single');

% Extracting HoG features of test images
for i=1:size(testImgNames,1)
    img = imread([TestDirectory, testImgNames(i,:)]);
    img = rgb2gray(uint8(img));
    img = imresize(img, newImageSize);
    testFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize, 'BlockSize', blockSize, 'BlockOverlap', blockOverlap, 'NumBins', numBins);  
end

%predicting output label using trained svm model
RecFaceID = predict(SVMmodel, testFeatures);

end
