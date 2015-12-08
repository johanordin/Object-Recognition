%% Import the files

% data 10000x3072 array of uint8.
% Each row in the array stores a 32x32 colour image.
% The first 1024 entries contain the red channel values, etc.
% The image is stored in row-major order, so that the first 32 entries are
% the red channel values of the first row of the image.

clear all
clc

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
%train_data = [data1; data2; data3; data4; data5];
%train_labels = [labels1; labels2; labels3; labels4; labels5]; 

%only use the first batch
train_data=data1;
train_labels=labels1; 

% note: test data is transposed
load(strcat('cifar-10-batches-mat/', 'test_batch.mat'));
test_data = double(data');
test_labels = double(labels');


%% Create target matrix for the training data for the Matlab nprtool
Targets=zeros(10, size(train_data,1));

for i = 1:size(train_data,1)
   j=train_labels(i)+1;
   Targets(j, i)=1;
end

%% Preprocessing of the data 

% centering the data (saving the means to apply it on the test data)
% images still stored row-wise
train_data_t=train_data';
train_data_t_means=mean(train_data_t);

train_data_tc=bsxfun(@minus,train_data_t,train_data_t_means);

%%
% performing PCA 
[coeff,score,eigenvalues] = pca(train_data_tc');

% determing the number of components to keep to maintan >99% of variation
var_sum=sum(eigenvalues);
var_kept=0.0;
numb_comp=0;
while var_kept<0.99
    numb_comp=numb_comp+1;
    var_kept=sum(eigenvalues(1:numb_comp))/var_sum;
end

fprintf('Number of components: %i \n', numb_comp)

%% 
% creating train_data_red as the reduced-dimension representation of the data
train_data_red=coeff(:,1:numb_comp)'*train_data_t;

% projecting the reduced images back to the full-dimensional input space
% for comparing the images
train_data_app=coeff(:,1:numb_comp)*train_data_red;

%% 

% close all plots windows
close all
clc

inputs  = train_data_t;
%inputs  = train_data_tc;
%inputs  = train_data_red;
targets = Targets;

[pn, ps1]    = mapstd(inputs); % standardizing 
[inputs,ps2] = processpca(pn,0.001); % uncorrelate every row, drop rows with low variation


%% 
% Create a Pattern Recognition Network
%hiddenLayerSize = [l1 l2];
hiddenLayerSize = 100;
net = patternnet(hiddenLayerSize);
%net = feedforwardnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% % Performance function
% net.performFcn = 'mse';
net.performFcn = 'crossentropy';
%net.performFcn = 'sse'; %Sum squared error performance function
%net.performFcn = 'sae';  %Sum absolute error performance function

%net.performParam.regularization = 0.01;

% % Transfer functions
%net.layers{1}.transferFcn = 'tansig';
%net.layers{1}.transferFcn = 'logsig';
%net.layers{2}.transferFcn = 'purelin';
%net.layers{2}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'softmax';

% % Train function 
%net.trainFcn = 'trainrp';
net.trainFcn = 'traingdx'  % Gradient descent with momentum and adaptive learning rate backpropagation
%net.trainFcn = 'traingdm'
%net.trainFcn = 'trainscg';
%net.trainFcn = 'trainlm'; % Levenberg-Marquardt
%net.trainFcn = 'trainbfg';

% % Train parameters trainscg
net.trainParam.max_fail = 1000;            % default 6
net.trainParam.min_grad = 1e-5;          % default 1e-6
% net.trainParam.lambda= 1;           % default 5.0e-7
%net.trainParam.sigma=5.0e-1;            % default 5.0e-5
%net.trainParam.goal;                    % default 0

% % Tran parameters trainrp
%net.trainParam.lr=0.5;                         % default 0.01
%net.trainParam.delt_inc=1.2;                 % default 1.2
%net.trainParam.delt_dec=0.5;                 % default 0.5
%net.trainParam.delta0=0.07;                  % default 0.07
%net.trainParam.deltamax=50;                  % default 50

% Train parameters traingdx
net.trainParam.lr=0.3;   %default 	0.01
net.trainParam.mc= 0.90;	  %Momentum constant 0.9
net.trainParam.epochs = 4000;


% Initialize the network
net = init(net);

n=1; % <--- how many different networks should be created? 

results     = double.empty; % to save outputs 
performance = double.empty; % to save the performance 
testIndezes = int8.empty;   % to save the instances used for testing 
confusions  = double.empty; % to save the fraction of misclassified samples

for i=1:n
    % Train the Network
    [trained_net, stats] = train(net, inputs, targets);
    % Test the Network
    outputs = trained_net(inputs);
    results = cat(1, results,outputs);    
    % errors = gsubtract(targets, outputs);
    % performance = perform(trained_net, targets, outputs);
    performance = cat(1,performance, perform(trained_net, targets, outputs));
    confusions  = cat(1,confusions , confusion(targets(:,stats.testInd), outputs(:,stats.testInd)));
    testIndezes = cat(1,testIndezes, stats.testInd); 
end

% calculating the average performance of the ANNs 
fprintf('Results over all networks: \n')
fprintf('Average network performance: %4.2f \n'   , mean(performance))
fprintf('Average Test Correct Class: %4.2f%%  \n' , 100*(1-mean(confusions)))
fprintf('Best: %4.2f%%  \n' , 100*(1-min(confusions)))
fprintf('Worst: %4.2f%%  \n' , 100*(1-max(confusions)))

% View the Network
view(trained_net);

% plot confusion table for the last network 
% plotconfusion(targets,outputs, 'Overall')
% plotconfusion(targets(:,stats.testInd),outputs(:,stats.testInd),'Test')

%%
% Print Percentage
[ct, cmt] = confusion(targets(:,stats.testInd), outputs(:,stats.testInd));
fprintf('Test  Correct Class  :%4.2f%%   \n'  , 100*(1-ct));
%fprintf('Test  Incorrect Class:%4.2f%%   \n'  , 100*ct);

[c, cm] = confusion(targets, outputs);
fprintf('Total Correct Class  :%4.2f%%   \n'  , 100*(1-c));
%fprintf('Total Incorrect Class:%4.2f%%   \n'  , 100*c);

fprintf('Test best performance:%4.2f%%   \n'  , stats.best_tperf);

% Plot the confusion matrix
% plotconfusion(targets,outputs, 'Overall', targets(:,stats.testInd),outputs(:,stats.testInd),'test')
figure;plotconfusion(targets(:,stats.testInd),  outputs(:,stats.testInd),  'Test')
                      %targets, outputs, 'Total')

% Plot the performance on linear scale
figure;             % Create a new figure.
ax = axes;          % Get a handle to the figure's axes
hold on;            % Set the figure to not overwrite old plots.
grid on;            % Turn on the grid.
plot(ax, stats.tperf)

% 
classes = vec2ind(outputs);

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotconfusion(targets,outputs)
% figure, ploterrhist(errors)


%% Equation for number of neurons in two layers
N = size(inputs,1);
m = 10;

l1 = round(sqrt(m+2)*N + 2*sqrt(N/(m+2)));
l2 = round(m*sqrt(N/(m+2)));

%% Reconstruct image - check that the data is correct after transformation.

% Get R, G, and B from the first ROW of data.
% R=train_data(1,1:1024);
% G=train_data(1,1025:2048);
% B=train_data(1,2049:3072);

% Get R, G, and B from the first ROW of the reduced dimensional data.
R=train_data_app(1:1024,1);
G=train_data_app(1025:2048,1);
B=train_data_app(2049:3072,1);

% Get R, G, and B from the first COLUMN of centered data.
R1=train_data_tc(1:1024, 1);
G1=train_data_tc(1025:2048, 1);
B1=train_data_tc(2049:3072, 1);
 
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

