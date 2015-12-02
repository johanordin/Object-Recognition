%% Import the files

% data 10000x3072 array of uint8.
% Each row in the array stores a 32x32 colour image.
% The first 1024 entries contain the red channel values, etc.
% The image is stored in row-major order, so that the first 32 entries are
% the red channel values of the first row of the image.

% notice: the instances are transposed, so the data of one image is one col
load(strcat('cifar-10-batches-mat/', 'data_batch_1.mat'));
data1 = double(data);
labels1 = double(labels);
load(strcat('cifar-10-batches-mat/', 'data_batch_2.mat'));
data2 = double(data);
labels2 = double(labels);
load(strcat('cifar-10-batches-mat/', 'data_batch_3.mat'));
data3 = double(data);
labels3 = double(labels);
load(strcat('cifar-10-batches-mat/', 'data_batch_4.mat'));
data4 = double(data);
labels4 = double(labels);
load(strcat('cifar-10-batches-mat/', 'data_batch_5.mat'));
data5 = double(data);
labels5 = double(labels);
train_data = [data1; data2; data3; data4; data5]';
train_labels = [labels1; labels2; labels3; labels4; labels5]'; 

load(strcat('cifar-10-batches-mat/', 'test_batch.mat'));
test_data = double(data');
test_labels = double(labels');


%% Create target matrix for the training data for the Matlab nprtool
Targets=zeros(10, size(train_data,2));

for i = 1:size(train_data,2)
   j=train_labels(i)+1;
   Targets(j, i)=1;
end


%% Preprocessing of the data 



%% 

% 
inputs  = double(train_data);
targets = Targets;

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net, inputs, targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs)

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotconfusion(targets,outputs)
% figure, ploterrhist(errors)




%% Reconstruct image - check that the data is correct after transformation.

% Get R, G, and B from the first ROW of data.
R=data(1,1:1024);
G=data(1,1025:2048);
B=data(1,2049:3072);

% Get R, G, and B from the first COLUMN of inputs.
R1=inputs(1:1024, 1);
G1=inputs(1025:2048, 1);
B1=inputs(2049:3072, 1);

% Create a 32x32 color image.
image1(:,:,1)=reshape(R,32,32);
image1(:,:,2)=reshape(G,32,32);
image1(:,:,3)=reshape(B,32,32);
image2(:,:,1)=reshape(R1,32,32);
image2(:,:,2)=reshape(G1,32,32);
image2(:,:,3)=reshape(B1,32,32);
% Display the color image.
imshow(image1);figure;
imshow(image2)

