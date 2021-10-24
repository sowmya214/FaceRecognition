clear all;
close all;
trainPath='FaceDatabase\Train\';
testPath='FaceDatabase\Test\';


%% Method 1 - HOG Features + SVM
tic;
   outputLabel1=FaceRecognition1(trainPath, testPath);
method1Time=toc

 load testLabel
 correctP=0;
 for i=1:size(testLabel,1)
    if strcmp(outputLabel1(i,:),testLabel(i,:))
        correctP=correctP+1;
    end
 end

 recAccuracy=correctP/size(testLabel,1)*100  %Recognition accuracy%


%% Method 2 - Transfer Learning with AlexNet
tic;
   outputLabel2=FaceRecognition2(trainPath, testPath);
method2Time=toc

 load testLabel
 correctP=0;
 for i=1:size(testLabel,1)
    if strcmp(outputLabel2(i,:),testLabel(i,:))
        correctP=correctP+1;
    end
 end
 recAccuracy=correctP/size(testLabel,1)*100  %Recognition accuracy%
