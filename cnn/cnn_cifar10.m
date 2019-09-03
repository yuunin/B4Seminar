dataset_path = '/home/Yuunin/Documents/MATLAB/CNN/example/cifar-10-batches-mat/';

img_size = [32 32 3];

cd(dataset_path)
%% load image
dataBatch = load('data_batch_1.mat')
%%
%dtst = zeros(img_size(1), img_size(2), img_size(3), size(dataBatch.data, 1));
for i = 1:size(dataBatch.data, 1)
    img = reshape(dataBatch.data(i,:), [32, 32, 3]);
    dtst(i) = permute(img, [2,1,3]);
    imshow(dtst(:,:,:,i))
end
%%
imshow(dtst(:,:,:,1))