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
%trainData = zeros(size(Data,1)-numVal, size(Data,2),'uint8');
valData = zeros(numVal, size(Data,2),'uint8');

val_index = 1;
train_index = 1;
for i = 1:size(Data,1)
    if ismember(i, indices)
        valData(val_index,:) = Data(i, :);
        labels_val(val_index,1) = labels_data(i);
        val_index = val_index + 1;
    else
        trainData(:,:,:,train_index) = uint8(reshape(Data(i,:), [img_size(1), img_size(2),3]));
        labels_train(train_index,1) = labels_data(i);
        train_index = train_index + 1;
    end
end
%% Data Augmentation 
augmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandScale',[0.5 1.5])
auimds = augmentedImageDatastore(img_size,trainData,labels_train,'DataAugmentation',augmenter)

%%
labels_train = categorical(labels_train);
labels_test = categorical(labels_test);
labels_val = categorical(labels_val);

%% selecting pixels(RGB)
percent = 10;
numElem = floor((percent/100)*img_size(1)*img_size(2));
rps_pix_train = zeros(img_size(1), img_size(2), 3, size(trainData, 1), 'uint8');
rps_pix_val = zeros(img_size(1), img_size(2), 3, size(valData, 1), 'uint8');

for i = 1:size(trainData, 1)
    img_1 = trainData(i,:);
    indices = randperm(img_size(1)*img_size(2), numElem);
    img_2 = zeros(1,img_size(1)*img_size(2)*3);
    img_2(indices) = img_1(indices);
    img_2(indices+img_size(1)*img_size(2)) = img_1(indices+img_size(1)*img_size(2));
    img_2(indices+img_size(1)*img_size(2)*2) = img_1(indices+img_size(1)*img_size(2)*2);
    img_3 = uint8(reshape(img_2, [img_size(1), img_size(2),3]));
    img_4 = permute(img_3, [2 1 3]);
    rps_pix_train(:,:,:,i) = img_4;  %(32,32,3,40000)
end

for i = 1:size(valData, 1)
    img_1 = valData(i,:);
    indices = randperm(img_size(1)*img_size(2), numElem);
    img_2 = zeros(1,img_size(1)*img_size(2)*3);
    img_2(indices) = img_1(indices);
    img_2(indices+img_size(1)*img_size(2)) = img_1(indices+img_size(1)*img_size(2));
    img_2(indices+img_size(1)*img_size(2)*2) = img_1(indices+img_size(1)*img_size(2)*2);
    img_3 = uint8(reshape(img_2, [img_size(1), img_size(2),3]));
    img_4 = permute(img_3, [2 1 3]);
    rps_pix_val(:,:,:,i) = img_4;  %(32,32,3,10000)
end
%% selecting RGB 
percent = 30;
numElem = floor((percent/100)*img_size(1)*img_size(2)*3);
rps_pix_train = zeros(img_size(1), img_size(2), 3, size(trainData, 1), 'uint8');
rps_pix_val = zeros(img_size(1), img_size(2), 3, size(valData, 1), 'uint8');

for i = 1:size(trainData, 1)
    img_1 = trainData(i,:);
    indices = randperm(img_size(1)*img_size(2)*3, numElem);
    img_2 = zeros(1,img_size(1)*img_size(2)*3);
    img_2(indices) = img_1(indices);
    img_3 = uint8(reshape(img_2, [img_size(1), img_size(2),3]));
    img_4 = permute(img_3, [2 1 3]);
    rps_pix_train(:,:,:,i) = img_4;  %(32,32,3,40000)
end

for i = 1:size(valData, 1)
    img_1 = valData(i,:);
    indices = randperm(img_size(1)*img_size(2)*3, numElem);
    img_2 = zeros(1,img_size(1)*img_size(2)*3);
    img_2(indices) = img_1(indices);
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
    img_3 = uint8(reshape(img_2, [img_size(1), img_size(2),3]));
    img_4 = permute(img_3, [2 1 3]);
    rps_pix_test(:,:,:,i) = img_4;  %(32,32,3,10000)
end
%%
imshow(rps_pix_val(:,:,:,1))
%%
layers = [
    imageInputLayer([img_size(1) img_size(2) 3])
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Stride', 1) 
    reluLayer
    batchNormalizationLayer
    convolution2dLayer(3, 32, 'Padding', 'same', 'Stride', 1) 
    reluLayer
    batchNormalizationLayer
    maxPooling2dLayer(2, 'Stride', 2)
    dropoutLayer(0.3)

    convolution2dLayer(3, 64, 'Padding', 'same', 'Stride', 1) 
    reluLayer
    batchNormalizationLayer
    convolution2dLayer(3, 64, 'Padding', 'same', 'Stride', 1) 
    reluLayer
    batchNormalizationLayer
    maxPooling2dLayer(2, 'Stride', 2)
    dropoutLayer(0.3)
    
    convolution2dLayer(3, 128, 'Padding', 'same', 'Stride', 1) 
    reluLayer
    batchNormalizationLayer
    convolution2dLayer(3, 128, 'Padding', 'same', 'Stride', 1) 
    reluLayer
    batchNormalizationLayer
    maxPooling2dLayer(2, 'Stride', 2)
    dropoutLayer(0.4)
    
    convolution2dLayer(3, 256, 'Padding', 'same', 'Stride', 1) 
    reluLayer
    batchNormalizationLayer
    convolution2dLayer(3, 256, 'Padding', 'same', 'Stride', 1) 
    reluLayer
    batchNormalizationLayer
    maxPooling2dLayer(2, 'Stride', 2)
    dropoutLayer(0.4)
    
    convolution2dLayer(3, 512, 'Padding', 'same', 'Stride', 1) 
    reluLayer
    batchNormalizationLayer
    convolution2dLayer(3, 512, 'Padding', 'same', 'Stride', 1) 
    reluLayer
    batchNormalizationLayer
    maxPooling2dLayer(2, 'Stride', 2)
    dropoutLayer(0.5)

    reluLayer

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
%

% NN options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',30,...
    'LearnRateDropFactor',0.1,...
    'L2Regularization', 0.0001,...
    'MaxEpochs',50, ...
    'MiniBatchSize', 128, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'ValidationData', {rps_pix_val, labels_val},...
    'ValidationFrequency', floor(size(rps_pix_train,4)/128),...
    'Plots','training-progress');

%train network
[net, trainInfo] = trainNetwork(rps_pix_train, labels_train, layers, options);

%%
[test, score] = classify(net, rps_pix_test);
%% 80.59% achieved for 100%pixels 
acc = 0;
for i = 1:10000
    if labels_test(i) == test(i)
        acc = acc + 1;
    end
end
accuracy = acc / 10000

%%
%{
if training accuracy is higher than validation accuracy, the model is too
complex or overfitting is the cause(making the model more simple or data augmentation is a solution). if its lower, underfitting is the
cause. 
%}
%%
train = rps_pix_train;
test = rps_pix_test;

size1 = size(train, 1);
size2 = size(train, 1);

X = zeros(size(train, 4), size1 * size2, 'double');
Xtest = zeros(size(test, 4), size1 * size2, 'double');

for data_num = 1:size(train, 4)
    for i = 1:size1
        for j = 1:size2
            X(data_num, (j - 1) * size1 + i) = train(i, j, data_num);
        end
    end
end

for data_num = 1:size(test, 3)
    for i = 1:size1
        for j = 1:size2
            Xtest(data_num, (j - 1) * size1 + i) = test(i, j, data_num);
        end
    end
end

X_zscored = zscore(X, 0, 1);
Xtest_zscored = zscore(Xtest, 0, 1);
%% ???K???E???V???A??????SVM
label = labels_test;
t = templateSVM('Standardize', 0, 'KernelFunction', 'gaussian', 'KernelScale', 'auto');
rng(1)
svm = fitcecoc(X_zscored, labels_train, 'Coding', 'onevsall', 'learners', t);
Loss_nonEtC = loss(svm, Xtest_zscored, label)
%%
predicted_label = predict(svm, Xtest_zscored);

Accuracy = 0;
for i = 1:size(test, 3)
     if predicted_label(i) == label(i)
         Accuracy = Accuracy + 1;
    end
end
Accuracy * 100 / size(test, 3)
%%
% ?????????`
%{
templateSVM('Standardize', 0, 'KernelFunction', 'linear', 'KernelScale', 'auto');
rng(1)
svm = fitcecoc(X_zscored, label_train, 'Coding', 'onevsall', 'learners', t);
Loss_nonEtC = loss(svm, Xtest_zscored, label_test)
predicted_label = predict(svm, Xtest_zscored);
Accuracy = 0;
for i = 1:38
    for j = 1:32
     if predicted_label((i-1)*32+j) == label_test((i-1) * 32 + j, 1)
         Accuracy = Accuracy + 1;
     end
    end
end
Accuracy * 100 / 1216
%}
% ??????????????????
%
t = templateSVM('Standardize', 0, 'KernelFunction', 'polynomial', 'KernelScale', 'auto');
rng(1)
svm = fitcecoc(X_zscored, train_label, 'Coding', 'onevsall', 'learners', t);
Loss_nonEtC = loss(svm, Xtest_zscored, label)

predicted_label = predict(svm, Xtest_zscored);

Accuracy = 0;
for i = 1:size(test, 3)
     if predicted_label(i) == label(i)
         Accuracy = Accuracy + 1;
    end
end
Accuracy * 100 / size(test, 3)






