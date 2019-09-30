dataset_path = '/Users/Hiroto Kai/Documents/MATLAB/cifar-10-batches-mat';

img_size = [32 32 3];

cd(dataset_path)
% load image
dataBatch_1 = load('data_batch_1.mat');
dataBatch_2 = load('data_batch_2.mat');
dataBatch_3 = load('data_batch_3.mat');
dataBatch_4 = load('data_batch_4.mat');
dataBatch_5 = load('data_batch_5.mat');
testBatch = load('test_batch.mat');
% 50000 x 3072
Data = [dataBatch_1.data;dataBatch_2.data;dataBatch_3.data;dataBatch_4.data;dataBatch_5.data];
labels_data = [dataBatch_1.labels;dataBatch_2.labels;dataBatch_3.labels;dataBatch_4.labels;dataBatch_5.labels];
testData = testBatch.data;
labels_test = testBatch.labels;
%% split [train, val] = [80, 20]

percent = 20;
numVal = floor((percent/100)*size(Data,1));
indices = randperm(size(Data, 1), numVal);
trainData = zeros(size(Data,1)-numVal, size(Data,2),'uint8');
valData = zeros(numVal, size(Data,2),'uint8');

val_index = 1;
train_index = 1;
for i = 1:size(Data,1)
    if ismember(i, indices)
        valData(val_index,:) = Data(i, :);
        labels_val(val_index,1) = labels_data(i);
        val_index = val_index + 1;
    else
        trainData(train_index,:) = Data(i,:);
        labels_train(train_index,1) = labels_data(i);
        train_index = train_index + 1;
    end
end
%%
labels_train = categorical(labels_train);
labels_test = categorical(labels_test);
labels_val = categorical(labels_val);

%%
percent = 30;
numElem = floor((percent/100)*img_size(1)*img_size(2)*3);
rps_pix_train = zeros(img_size(1), img_size(2), 3, size(trainData, 1), 'uint8');
rps_pix_val = zeros(img_size(1), img_size(2), 3, size(valData, 1), 'uint8');

for i = 1:size(trainData, 1)
    img_1 = trainData(i,:);
    indices = randperm(img_size(1)*img_size(2)*3, numElem);
    img_2 = zeros(1,img_size(1)*img_size(2)*3);
    img_2(indices) = img_1(indices);
    %img_2(indices+img_size(1)*img_size(2)) = img_1(indices+img_size(1)*img_size(2));
    %img_2(indices+img_size(1)*img_size(2)*2) = img_1(indices+img_size(1)*img_size(2)*2);
    img_3 = uint8(reshape(img_2, [img_size(1), img_size(2),3]));
    img_4 = permute(img_3, [2 1 3]);
    rps_pix_train(:,:,:,i) = img_4;  %(32,32,3,40000)
end

for i = 1:size(valData, 1)
    img_1 = valData(i,:);
    indices = randperm(img_size(1)*img_size(2)*3, numElem);
    img_2 = zeros(1,img_size(1)*img_size(2)*3);
    img_2(indices) = img_1(indices);
    %img_2(indices+img_size(1)*img_size(2)) = img_1(indices+img_size(1)*img_size(2));
    %img_2(indices+img_size(1)*img_size(2)*2) = img_1(indices+img_size(1)*img_size(2)*2);
    img_3 = uint8(reshape(img_2, [img_size(1), img_size(2),3]));
    img_4 = permute(img_3, [2 1 3]);
    rps_pix_val(:,:,:,i) = img_4;  %(32,32,3,10000)
end
%%
percent = 30;
numElem = floor((percent/100)*img_size(1)*img_size(2)*3);
rps_pix_test = zeros(img_size(1), img_size(2), 3, size(testData, 1), 'uint8');
for i = 1:size(testData, 1)
    img_1 = testData(i,:);
    indices = randperm(img_size(1)*img_size(2)*3, numElem);
    img_2 = zeros(1,img_size(1)*img_size(2)*3);
    img_2(indices) = img_1(indices);
    %img_2(indices+img_size(1)*img_size(2)) = img_1(indices+img_size(1)*img_size(2));
    %img_2(indices+img_size(1)*img_size(2)*2) = img_1(indices+img_size(1)*img_size(2)*2);
    img_3 = uint8(reshape(img_2, [img_size(1), img_size(2),3]));
    img_4 = permute(img_3, [2 1 3]);
    rps_pix_test(:,:,:,i) = img_4;  %(32,32,3,10000)
end
%% resnet-18

netWidth = 32;
layers = [
    imageInputLayer([32 32 3],'Name','input')
    
    convolution2dLayer(3,netWidth,'Padding','same','Name','convInp')
    batchNormalizationLayer('Name','BNInp')
    reluLayer('Name','reluInp')
    
    convolutionalUnit(netWidth,1,'S1U1')
    additionLayer(2,'Name','add11')
    reluLayer('Name','relu11')
    convolutionalUnit(netWidth,1,'S1U2')
    additionLayer(2,'Name','add12')
    reluLayer('Name','relu12')
    convolutionalUnit(netWidth,1,'S1U3')
    additionLayer(2,'Name','add13')
    reluLayer('Name','relu13')
    
    convolutionalUnit(2*netWidth,2,'S2U1')
    additionLayer(2,'Name','add21')
    reluLayer('Name','relu21')
    convolutionalUnit(2*netWidth,1,'S2U2')
    additionLayer(2,'Name','add22')
    reluLayer('Name','relu22')
    convolutionalUnit(2*netWidth,1,'S2U3')
    additionLayer(2,'Name','add23')
    reluLayer('Name','relu23')
    
    convolutionalUnit(4*netWidth,2,'S3U1')
    additionLayer(2,'Name','add31')
    reluLayer('Name','relu31')
    convolutionalUnit(4*netWidth,1,'S3U2')
    additionLayer(2,'Name','add32')
    reluLayer('Name','relu32')
    convolutionalUnit(4*netWidth,1,'S3U3')
    additionLayer(2,'Name','add33')
    reluLayer('Name','relu33')
   %{ 
    convolutionalUnit(6*netWidth,2,'S4U1')
    additionLayer(2,'Name','add41')
    reluLayer('Name','relu41')
    convolutionalUnit(6*netWidth,1,'S4U2')
    additionLayer(2,'Name','add42')
    reluLayer('Name','relu42')
    convolutionalUnit(6*netWidth,1,'S4U3')
    additionLayer(2,'Name','add43')
    reluLayer('Name','relu43')
    %}
    averagePooling2dLayer(8,'Name','globalPool')
    fullyConnectedLayer(10,'Name','fcFinal')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];

lgraph = layerGraph(layers);

lgraph = connectLayers(lgraph,'reluInp','add11/in2');
lgraph = connectLayers(lgraph,'relu11','add12/in2');
lgraph = connectLayers(lgraph,'relu12','add13/in2');

skip1 = [
    convolution2dLayer(1,2*netWidth,'Stride',2,'Name','skipConv1')
    batchNormalizationLayer('Name','skipBN1')];
lgraph = addLayers(lgraph,skip1);
lgraph = connectLayers(lgraph,'relu13','skipConv1');
lgraph = connectLayers(lgraph,'skipBN1','add21/in2');

lgraph = connectLayers(lgraph,'relu21','add22/in2');
lgraph = connectLayers(lgraph,'relu22','add23/in2');

skip2 = [
    convolution2dLayer(1,4*netWidth,'Stride',2,'Name','skipConv2')
    batchNormalizationLayer('Name','skipBN2')];
lgraph = addLayers(lgraph,skip2);
lgraph = connectLayers(lgraph,'relu23','skipConv2');
lgraph = connectLayers(lgraph,'skipBN2','add31/in2');

lgraph = connectLayers(lgraph,'relu31','add32/in2');
lgraph = connectLayers(lgraph,'relu32','add33/in2');
%{
skip3 = [
    convolution2dLayer(1,6*netWidth,'Stride',2,'Name','skipConv3')
    batchNormalizationLayer('Name','skipBN3')];
lgraph = addLayers(lgraph,skip3);
lgraph = connectLayers(lgraph,'relu33','skipConv3');
lgraph = connectLayers(lgraph,'skipBN3','add41/in2');

lgraph = connectLayers(lgraph,'relu41','add42/in2');
lgraph = connectLayers(lgraph,'relu42','add43/in2');
%}
figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
plot(lgraph)
%% training option 
miniBatchSize = 128;
learnRate = 0.01*miniBatchSize/128;
valFrequency = floor(size(rps_pix_train,4)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'InitialLearnRate',learnRate, ...
    'MaxEpochs',80, ...
    'MiniBatchSize',miniBatchSize, ...
    'VerboseFrequency',valFrequency, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{rps_pix_val, labels_val}, ...
    'ValidationFrequency',valFrequency, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',30);

%% 62.14%
[net, trainInfo] = trainNetwork(rps_pix_train, labels_train, lgraph, options);

%%
[test, score] = classify(net, rps_pix_test);
acc = 0;
for i = 1:10000
    if labels_test(i) == test(i)
        acc = acc + 1;
    end
end
accuracy = acc / 10000



















