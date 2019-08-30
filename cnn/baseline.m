function [img_train, label_train, img_test, label_test] = ...
    baseline(dataset_path, train_number_per_class, img_size)

random_seed = 1;
rng(random_seed)
class_seed = randperm(38);
train_index = zeros(38, train_number_per_class, 'uint8');

for i = 1:38
    rng(class_seed(i))
    perm = randperm(64);
    train_index(i, :) = perm(1, 1:train_number_per_class);
end

back = cd(dataset_path);
YaleList = dir('yaleB*');
img_train = zeros(img_size(1), img_size(2), train_number_per_class * 38, 'uint8');
img_test = zeros(img_size(1), img_size(2), (64 - train_number_per_class) * 38, 'uint8');

train_count = 0;
test_count = 0;
for y = 1:size(YaleList, 1)
    nowYale = YaleList(y).name
    back2 = cd(nowYale);
    imageList = dir();
    for listNum = 1:size(imageList, 1)
        isdir(listNum) = imageList(listNum).isdir;
    end
    imageList = imageList(find(isdir == 0));
    
    for imNum = 1:size(imageList, 1)
        img = imresize(imread(imageList(imNum).name), img_size);
        if isempty(find(train_index(y, :) == imNum))
            test_count = test_count + 1;
            label_test(test_count) = y;
            [img_test(:,:,test_count)] = img;
        else
            train_count = train_count + 1;
            label_train(train_count) = y;
            [img_train(:,:,train_count)] = img;
        end
    end
    cd(back2);
end

cd(back);


