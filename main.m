%% Notice: 
% the data files("data_batch_1.mat", etc.) have to be in a folder called
% "cifar-10-batches-mat" in the location of this MATLAB script 

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
train_data = [data1; data2; data3; data4; data5];
train_labels = [labels1; labels2; labels3; labels4; labels5]; 

% outcomment the following 2 lines, if only the use of the second data batch is wanted 
% train_data=data2;
% train_labels=labels2; 

% note: test data is transposed
load(strcat('cifar-10-batches-mat/', 'test_batch.mat'));
test_data = double(data');
test_labels = double(labels');

% Create target matrix for the training data for the Matlab nprtool
Targets=zeros(10, size(train_data,1));
for i = 1:size(train_data,1)
   j=train_labels(i)+1;
   Targets(j, i)=1;
end

% Create target matrix for the test data 
Test_targets=zeros(10, size(test_data,2));
for i = 1:size(test_data,2)
   j=test_labels(i)+1;
   Test_targets(j, i)=1;
end

%% Preprocessing of the images 
% each of the following sections is dedicated to one method of preprocessing 
% running a section will make it possible to use it to train the ANN later
% there only the corresponding line has to be commented 

%% 1. Centering the data
% centering the data (saving the means to apply it on the test data)
% images still stored row-wise
train_data_t=train_data'; 
train_data_t_means=mean(train_data_t);
train_data_tc=bsxfun(@minus, train_data_t, train_data_t_means); 

%% 2. Standardization of the data 
[train_data_std, setting_std]= mapstd(train_data'); 

%% 3. Principal Component Analysis (PCA) 
% standardized data ("train_data_std") is used for the following methods 

% 3.1 using pca 
[coeff, score, eigenvalues] = pca(train_data_std');
% score, which is the representation of X in the principal component space

% determing the number of components to keep to maintan >99% of the variation
var_sum=sum(eigenvalues);
var_kept=0.0;
numb_comp=0;
while var_kept<0.99
    numb_comp=numb_comp+1;
    var_kept=sum(eigenvalues(1:numb_comp))/var_sum;
end
fprintf('Number of components: %i \n', numb_comp)

% creating a reduced-dimension representation of the data 
train_data_red=coeff(:,1:numb_comp)'*train_data_std;

% projection of the reduced images back to the full-dimensional input space 
% was used for comparing the images with the original images 
% train_data_app=coeff(:,1:numb_comp)*train_data_red;

%%
% 3.2 using processpca 
% uncorrelate every row (pixel), drop rows with low variation 
[train_data_ppca,setting_processpca] = processpca(train_data_std,0.004); 

%% Initializing and training of the network(s)
% You can/should set these parameters to the prefered values before running
% this section: 
% - inputs, targets: the data and the corresponding target matrix 
% - hiddenLayserSize: number of hidden neurons 
% - n: number of networks that are trained to get an estimate of the performance 
% - net.performParam.regularization: the value for the regularization 

% Defining inputs and targets for the network
% depending on what data you want to use as input data uncomment the
% correspondig line: 

% inputs = train_data_tc;  % centered data 
% inputs = train_data_std; % standardized data 
% inputs = score';         % principal components from the pca method 
% inputs = train_img';        % image processed data
% inputs = train_w;           % Whitening data
inputs = train_data_ppca;  % components from processpca, number of components depending on the variation stated at point 3.2 

targets = Targets;

% Create the Neural Network
hiddenLayerSize = 100;      % setting the number of hidden neurons 
n=1;                        %  how many different networks should be created,  
                            %  the best, average and worst performance of
                            %  the built networks will be printed out 
             
net = patternnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% % Performance function
net.performFcn = 'crossentropy';
net.performParam.regularization = 0.1;      % setting the regularization 

% % Train parameters trainscg
net.trainParam.max_fail = 50;            % default 6

% Initialize the network
net = init(net);

close all % close all plots windows
clc

% training and testing the network with the set parameters
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
% view(trained_net);

% plot confusion table for the last network 
% plotconfusion(targets,outputs, 'Overall')
% plotconfusion(targets(:,stats.testInd),outputs(:,stats.testInd),'Test')

%% plots (can be disregarded)
% % Print Percentage
% [ct, cmt] = confusion(targets(:,stats.testInd), outputs(:,stats.testInd));
% fprintf('Test  Correct Class  :%4.2f%%   \n'  , 100*(1-ct));
% [c, cm] = confusion(targets, outputs);
% fprintf('Total Correct Class  :%4.2f%%   \n'  , 100*(1-c));
% fprintf('Test best performance:%4.2f%%   \n'  , stats.best_tperf);
% 
% % Plot the confusion matrix
% % plotconfusion(targets,outputs, 'Overall', targets(:,stats.testInd),outputs(:,stats.testInd),'test')
% figure;plotconfusion(targets(:,stats.testInd),  outputs(:,stats.testInd),  'Test')
%                       %targets, outputs, 'Total')
% 
% % Plot the performance on linear scale
% figure;             % Create a new figure.
% ax = axes;          % Get a handle to the figure's axes
% hold on;            % Set the figure to not overwrite old plots.
% grid on;            % Turn on the grid.
% plot(ax, stats.tperf)
% 
% classes = vec2ind(outputs);

%% Testing the neural network on the test data 
% preprocessing the test data as the train data: 
test_data_std = mapstd('apply',test_data,setting_std);% standardization of the data
test_data_trans = processpca('apply',test_data_std,setting_processpca_004);%processpcaoutputs=trained_net(test_data_trans);% feeding the network with the test data

[val ind]=max(outputs,[],1); % highest class probability and position per instance
ind=ind-1; % position - 1 = class 
classification_rate=sum(ind == test_labels)/length(ind) % Classification rate 
%% Notice 2: This is the code for image processing that was implemented but
%            not used in the end.
%            This can be tested if you uncomment the rows until next notice
%            Then you also need to uncomment train_img at row 115.
% 
% %% Preprocessing - methods applied on images 
% %% Convert to dataset to rgb images
% train_data_int = uint8(train_data);
% R=train_data_int(:,1:1024);
% G=train_data_int(:,1025:2048);
% B=train_data_int(:,2049:3072);
% 
% for i=1:10000
%     img_org(:,:,1,i)=reshape(R(i,:),32,32);
%     img_org(:,:,2,i)=reshape(G(i,:),32,32);
%     img_org(:,:,3,i)=reshape(B(i,:),32,32);
% end
% class(img_org)% check uint8, no data loss
% 
% %% Process images
% for i=1:10000
%     for j=1:size(img_org,3)
%         %denoised(:,:,j,i) = medfilt2(img_org(:,:,j,i),[3,3]); % only median filter
%         adjusted(:,:,j,i) = histeq(img_org(:,:,j,i));          % only histogram equalization
%         %adjusted(:,:,j,i) = histeq(denoised(:,:,j,i));        % both
%     end
% end
% %% Plot - Show contrast normalization results visually
% figure;
% subplot(1,2,1),imshow(uint8(img_org(:,:,:,8888))),title('Original RGB image')
% subplot(1,2,2),imshow(uint8(adjusted(:,:,:,8888))),title('Adjusted contrast RGB image')
% 
% %% Plot -Show median filter results visually
% figure;
% subplot(2,2,1),imshow(uint8(img_org(:,:,:,10000-5))), title('Original image');
% subplot(2,2,2),imshow(uint8(denoised(:,:,:,10000-5))),title('Median filter image');
% subplot(2,2,3),imshow(uint8(img_org(:,:,:,10000-10))), title('Original image');
% subplot(2,2,4),imshow(uint8(denoised(:,:,:,10000-10))),title('Median filter image');
% 
% %% Convert rgb images back to original data representation.
% train_img = train_data;
% denoised=adjusted;
% for i=1:10000
%         red_ch = denoised(:,:,1,i);
%         green_ch = denoised(:,:,2,i);
%         blue_ch = denoised(:,:,3,i);
%         red_ch_lin = red_ch(:)';
%         green_ch_lin = green_ch(:)';
%         blue_ch_lin = blue_ch(:)';
%         train_img(i,:) = [red_ch_lin,red_ch_lin,red_ch_lin];
% end

%% Notice 3: This is the code of the Whitening transformation that we tried
%            to implement but not used in the end.
%            This can be tested if you uncomment the rows until next notice
%            Then you also need to uncomment train_w at row 116.
% %% Whitening transformation
% n_samples = size(train_data_tc,2); %10000
% 
% covar_matrix = (train_data_tc*train_data_tc')*(1/(n_samples-1));
% %imagesc(covar_matrix) % check if this was done correct
% 
% [V,D]=eig(covar_matrix);
% train_data_tc_rot=V'*train_data_tc;
% 
% eps = 0.1;
% train_data_w = diag(1./sqrt(diag(D) + eps )) * train_data_tc_rot;
% %AC = cov(train_data_w);
% 
%% Notice 4: This is the code for translating the images in the data set
%            to the image domain and look at them visually.
%          
% %% Reconstruct images from the data
% % used to check if methods have been correctly applied to the data
% 
% % Get R, G, and B from the first ROW of data.
% % R=train_data(1,1:1024);
% % G=train_data(1,1025:2048);
% % B=train_data(1,2049:3072);
% 
% % Get R, G, and B from the first ROW of the reduced dimensional data.
% R=train_data_app(1:1024,1);
% G=train_data_app(1025:2048,1);
% B=train_data_app(2049:3072,1);
% 
% % Get R, G, and B from the first COLUMN of centered data.
% R1=train_data_tc(1:1024, 1);
% G1=train_data_tc(1025:2048, 1);
% B1=train_data_tc(2049:3072, 1);
%  
% % Create a 32x32 color image.
% image1(:,:,1)=reshape(R,32,32);
% image1(:,:,2)=reshape(G,32,32);
% image1(:,:,3)=reshape(B,32,32);
% image2(:,:,1)=reshape(R1,32,32);
% image2(:,:,2)=reshape(G1,32,32);
% image2(:,:,3)=reshape(B1,32,32);
% % Display the color image.
% imshow(image1);figure;
% imshow(image2)

