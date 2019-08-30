dataset_path = '/home/Yuunin/Documents/MATLAB/CNN/example/CroppedYaleDataset';
DOG_path = '/home/Yuunin/Documents/MATLAB/DOG';

train_number_per_class = 54;
seed = 42;
img_size = [64 64];
blocksize = [32 32];
%% [train, test] = [0.85, 0.15]
cd('/home/Yuunin/Documents/MATLAB/CNN/');
[img_train, label_train, img_test, label_test] = ...
    baseline(dataset_path, train_number_per_class, img_size);
%% need to use image dataset(save img_train as image folders)
%[data_Train,data_Validation] = splitEachLabel(img_train,0.7, 0.3,'randomize');
%% [train, validation] = [0.8, 0.2]
sub_train_per_class = 43;
valid_per_class = 11;

train = zeros(img_size(1), img_size(2), 1634);
validation = zeros(img_size(1), img_size(2), 418);
%rng(42)
for i = 1:38
    imds = img_train(:,:,(i-1)*54+1:i*54);
    indices = randperm(train_number_per_class);
    train(:,:,(i-1)*43+1:i*43) = imds(:,:,indices(1:43));
    train_label((i-1)*43+1:i*43) = i;
    validation(:,:,(i-1)*11+1:i*11) = imds(:,:,indices(44:54));
    valid_label((i-1)*11+1:i*11) = i;
end


%% random pixel selection(RPS-vector)
percent = 5;
%rng(42)
numElem = floor(percent/100*img_size(1)*img_size(2));
indices = randperm(img_size(1)*img_size(2), numElem);
rps_vec = zeros(numElem, size(img_train, 3));

for i = 1:size(img_train, 3)
    img = img_train(:,:,i);
    img_2 = reshape(img', 1,4096);
    rps_vec(:,i) = img_2(indices);  %(409, 1216)
end
%% random pixel selection(RPS-pixel) [train , validation]
percent = 10;
numElem = floor(percent/100*img_size(1)*img_size(2));
indices = randperm(img_size(1)*img_size(2), numElem);
rps_pix_train = zeros(img_size(1), img_size(2), sub_train_per_class * 38, 'uint8');
rps_pix_valid = zeros(img_size(1), img_size(2), valid_per_class * 38, 'uint8');
for i = 1:size(train, 3)
    img_1 = train(:,:,i);
    img_2 = reshape(img_1', 1,4096);
    img_3 = zeros(1,4096);
    img_3(indices) = img_2(indices);
    img_4 = uint8(reshape(img_3, 64,64))';
    rps_pix_train(:,:,i) = img_4;  %(64, 64, 1634)
end

for i = 1:size(validation, 3)
    img_1 = validation(:,:,i);
    img_2 = reshape(img_1', 1,4096);
    img_3 = zeros(1,4096);
    img_3(indices) = img_2(indices);
    img_4 = uint8(reshape(img_3, 64,64))';
    rps_pix_valid(:,:,i) = img_4;  %(64, 64, 418)
end

Train = reshape(rps_pix_train, [img_size(1), img_size(2), 1, size(rps_pix_train, 3)]);
Valid = reshape(rps_pix_valid, [img_size(1), img_size(2), 1, size(rps_pix_valid, 3)]);
TrainLabel = categorical(train_label);
ValidLabel = categorical(valid_label);

%% random pixel selection(RPS-pixel) [test]
percent = 10;
numElem = floor(percent/100*img_size(1)*img_size(2));
indices = randperm(img_size(1)*img_size(2), numElem);
rps_pix_test = zeros(img_size(1), img_size(2), size(img_test , 3), 'uint8');
for i = 1:size(img_test, 3)
    img_1 = img_test(:,:,i);
    img_2 = reshape(img_1', 1,4096);
    img_3 = zeros(1,4096);
    img_3(indices) = img_2(indices);
    img_4 = uint8(reshape(img_3, 64,64))';
    rps_pix_test(:,:,i) = img_4;  %(64, 64, 1634)
end

Test = reshape(rps_pix_test, [img_size(1), img_size(2), 1, size(rps_pix_test, 3)]);
TestLabel = categorical(label_test);
%%

%CNN structure

layers = [
    imageInputLayer([64 64 1])
    
    convolution2dLayer(3, 8, 'Padding', 'same') 
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(38)
    softmaxLayer
    classificationLayer];

% NN options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',16, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{Valid, ValidLabel}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');


%train network
net = trainNetwork(Train, TrainLabel, layers, options);

%%
% validation 90.19%
[ValidPred, scores] = classify(net, Valid);
Accuracy = 0;
for i = 1:418
    if ValidPred(i) == ValidLabel(i)
        Accuracy = Accuracy + 1;
    end
end
%%
[TestPred, scores] = classify(net, Test);
Accuracy = 0;
for i = 1:380
    if TestPred(i) == TestLabel(i)
        Accuracy = Accuracy + 1;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
Train = reshape(train, [img_size(1), img_size(2), 1, size(train, 3)]);
Valid = reshape(validation, [img_size(1), img_size(2), 1, size(validation, 3)]);
TrainLabel = categorical(train_label);
ValidLabel = categorical(valid_label);
%% random pixel selection(RPS-vector)
percent = 10;
numElem = floor(percent/100*img_size(1)*img_size(2));
rng(42);
indices = randperm(img_size(1)*img_size(2), numElem);
rps_vec = zeros(numElem, size(img_train, 3));

for i = 1:size(img_train, 3)
    img = img_train(:,:,i);
    img_2 = reshape(img', 1,4096);
    rps_vec(:,i) = img_2(indices);  %(409, 1216)
end
%% random pixel selection(RPS-pixel)
percent = 50;
numElem = floor(percent/100*img_size(1)*img_size(2));
rng(42);
indices = randperm(img_size(1)*img_size(2), numElem);
rps_pix = zeros(img_size(1), img_size(2), train_number_per_class * 38, 'uint8');

for i = 1:size(img_train, 3)
    img_1 = img_train(:,:,i);
    img_2 = reshape(img_1', 1,4096);
    img_3 = zeros(1,4096);
    img_3(indices) = img_2(indices);
    img_4 = uint8(reshape(img_3, 64,64))';
    rps_pix(:,:,i) = img_4;  %(64, 64, 1216)
end

%% block images
cd('/home/Yuunin/Documents/MATLAB/CNN/');
%%
train = rps_pix_train;
test = rps_pix_valid;
size1 = size(train, 1);
size2 = size(train, 2);

X = zeros(size(train, 3), size1 * size2, 'double');
Xtest = zeros(size(test, 3), size1 * size2, 'double');

for data_num = 1:size(train, 3)
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
%% �K�E�V�A��SVM

t = templateSVM('Standardize', 0, 'KernelFunction', 'gaussian', 'KernelScale', 'auto');
rng(1)
svm = fitcecoc(X_zscored, train_label, 'Coding', 'onevsall', 'learners', t);
Loss_nonEtC = loss(svm, Xtest_zscored, valid_label)

predicted_label = predict(svm, Xtest_zscored);

Accuracy = 0;
for i = 1:size(test, 3)
     if predicted_label(i) == valid_label(i)
         Accuracy = Accuracy + 1;
    end
end
Accuracy * 100 / size(test, 3)

% ���`
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
% ������
%
t = templateSVM('Standardize', 0, 'KernelFunction', 'polynomial', 'KernelScale', 'auto');
rng(1)
svm = fitcecoc(X_zscored, train_label, 'Coding', 'onevsall', 'learners', t);
Loss_nonEtC = loss(svm, Xtest_zscored, valid_label)

predicted_label = predict(svm, Xtest_zscored);

Accuracy = 0;
for i = 1:size(test, 3)
     if predicted_label(i) == valid_label(i)
         Accuracy = Accuracy + 1;
    end
end
Accuracy * 100 / size(test, 3)
