%% Import the file
newData = load('-mat', 'C:\Users\Sebastian\Google Drive\CI - Project\Data\cifar-10-batches-mat\data_batch_1.mat');

% Create new variables in the base workspace from those fields.
%vars = fieldnames(newData);

vars={'data' 'labels'};
for i = 1:length(vars)
    assignin('base', vars{i}, newData.(vars{i}));
end

%%
% create the right target vector for the Matlab nprtool
% labels=abs(labels);
T=zeros(10000,10);

for i = 1:10000
   j=labels(i)+1;
   T(i,j)=1;
end

names = cell(5,2);
for i=1:5
    load(strcat('cifar-10-batches-mat/', 'data_batch_', num2str(i),'.mat'));
    data1=double(data);
    labels1=double(labels);
end



%% take 6 batches data into one unite set
load('data_batch_1.mat');
data1 = double(data);
labels1 = double(labels);
load('data_batch_2.mat');
data2 = double(data);
labels2 = double(labels);
load('data_batch_3.mat');
data3 = double(data);
labels3 = double(labels);
load('data_batch_4.mat');
data4 = double(data);
labels4 = double(labels);
load('data_batch_5.mat');
data5 = double(data);
labels5 = double(labels);
load('test_batch.mat');

testData = double(data);
testLabels = double(labels);
data = [data1; data2; data3; data4; data5; testData];
labels = [labels1; labels2; labels3; labels4; labels5; testLabels]; 


