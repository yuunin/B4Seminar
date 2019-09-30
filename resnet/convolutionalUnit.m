function layers = convolutionalUnit(numF,stride,tag)
layers = [
    convolution2dLayer(3,numF,'Padding','same','Stride',stride,'Name',[tag,'conv1'])
    dropoutLayer(0.2, 'Name', [tag,'Dropout1'])
    batchNormalizationLayer('Name',[tag,'BN1'])
    reluLayer('Name',[tag,'relu1'])
    convolution2dLayer(3,numF,'Padding','same','Name',[tag,'conv2'])
    dropoutLayer(0.3, 'Name', [tag,'Dropout2'])
    batchNormalizationLayer('Name',[tag,'BN2'])];
end