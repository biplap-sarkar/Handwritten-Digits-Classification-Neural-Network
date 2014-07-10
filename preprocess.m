function [train_data, train_label, validation_data, ...
    validation_label, test_data, test_label] = preprocess()
% preprocess function loads the original data set, performs some preprocess
%   tasks, and output the preprocessed train, validation and test data.

% Input:
% Although this function doesn't have any input, you are required to load
% the MNIST data set from file 'mnist_all.mat'.

% Output:
% train_data: matrix of training set. Each row of train_data contains 
%   feature vector of a image
% train_label: vector of label corresponding to each image in the training
%   set
% validation_data: matrix of training set. Each row of validation_data 
%   contains feature vector of a image
% validation_label: vector of label corresponding to each image in the 
%   training set
% test_data: matrix of training set. Each row of test_data contains 
%   feature vector of a image
% test_label: vector of label corresponding to each image in the testing
%   set

% Some suggestions for preprocessing step:
% - divide the original data set to training, validation and testing set
%       with corresponding labels
% - convert original data set from integer to double by using double()
%       function
% - normalize the data to [0, 1]
% - feature selection

S = load('mnist_all.mat');
c = struct2cell(S);
v = cell2mat(c(1,1));
%v = v(randperm(size(v,1)),:);
temp = v(1:5000,1:784);
train_label = zeros(50000,1);
validation_label = zeros(10000,1);
validation_data = zeros(10000,784);
row = length(v);
index = row - 5000;
dataindex = 1;
validation_data(dataindex:index,1:784) = v(5001:row,1:784);
dataindex = dataindex+index;
v = cell2mat(c(3,1));
%v = v(randperm(size(v,1)),:);
temp1 = v(1:5000,1:784);
for j = 5000:10000
    train_label(j,1) = 1;
end
row = length(v);
validation_data(dataindex:row - 5001 + dataindex,1:784) = v(5001:row,1:784);
dataindex = dataindex + row - 5000;
for j = 5001:row
    validation_label(index,1) = 1;
    index = index+1;
end
v = cell2mat(c(5,1));
%v = v(randperm(size(v,1)),:);
temp2 = v(1:5000,1:784);
for j = 10000:15000
    train_label(j,1) = 2;
end
row = length(v);
validation_data(dataindex:row - 5001 + dataindex,1:784) = v(5001:row,1:784);
dataindex = dataindex + row - 5000;
for j = 5001:row
    validation_label(index,1) = 2;
    index = index+1;
end
v = cell2mat(c(7,1));
%v = v(randperm(size(v,1)),:);
temp3 = v(1:5000,1:784);
for j = 15000:20000
    train_label(j,1) = 3;
end
row = length(v);
validation_data(dataindex:row - 5001 + dataindex,1:784) = v(5001:row,1:784);
dataindex = dataindex + row - 5000;
for j = 5001:row
    validation_label(index,1) = 3;
    index = index+1;
end
v = cell2mat(c(9,1));
%v = v(randperm(size(v,1)),:);
temp4 = v(1:5000,1:784);
for j = 20000:25000
    train_label(j,1) = 4;
end
row = length(v);
validation_data(dataindex:row - 5001 + dataindex,1:784) = v(5001:row,1:784);
dataindex = dataindex + row - 5000;
for j = 5001:row
    validation_label(index,1) = 4;
    index = index+1;
end
v = cell2mat(c(11,1));
%v = v(randperm(size(v,1)),:);
temp5 = v(1:5000,1:784);
for j = 25000:30000
    train_label(j,1) = 5;
end
row = length(v);
validation_data(dataindex:row - 5001 + dataindex,1:784) = v(5001:row,1:784);
dataindex = dataindex + row - 5000;
for j = 5001:row
    validation_label(index,1) = 5;
    index = index+1;
end
v = cell2mat(c(13,1));
%v = v(randperm(size(v,1)),:);
temp6 = v(1:5000,1:784);
for j = 30000:35000
    train_label(j,1) = 6;
end
row = length(v);
validation_data(dataindex:row - 5001 + dataindex,1:784) = v(5001:row,1:784);
dataindex = dataindex + row - 5000;
for j = 5001:row
    validation_label(index,1) = 6;
    index = index+1;
end
v = cell2mat(c(15,1));
%v = v(randperm(size(v,1)),:);
temp7 = v(1:5000,1:784);
for j = 35000:40000
    train_label(j,1) = 7;
end
row = length(v);
validation_data(dataindex:row - 5001 + dataindex,1:784) = v(5001:row,1:784);
dataindex = dataindex + row - 5000;
for j = 5001:row
    validation_label(index,1) = 7;
    index = index+1;
end
v = cell2mat(c(17,1));
%v = v(randperm(size(v,1)),:);
temp8 = v(1:5000,1:784);
for j = 40000:45000
    train_label(j,1) = 8;
end
row = length(v);
validation_data(dataindex:row - 5001 + dataindex,1:784) = v(5001:row,1:784);
dataindex = dataindex + row - 5000;
for j = 5001:row
    validation_label(index,1) = 8;
    index = index+1;
end
v = cell2mat(c(19,1));
%v = v(randperm(size(v,1)),:);
temp9 = v(1:5000,1:784);
for j = 45000:50000
    train_label(j,1) = 9;
end
row = length(v);
validation_data(dataindex:row - 5001 + dataindex,1:784) = v(5001:row,1:784);
for j = 5001:row
    validation_label(index,1) = 9;
    index = index+1;
end
train_data = vertcat(temp,temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9);
test_data = vertcat(c{[2 4 6 8 10 12 14 16 18 20]});
test_label = zeros(10000,1);
v = cell2mat(c(2,1));
row = length(v);
v = cell2mat(c(4,1));
row1 = length(v);
for j = row+1:row1+row
    test_label(j,1) = 1;
end
row = row+row1;
v = cell2mat(c(6,1));
row1 = length(v);
for j = row+1:row1+row
    test_label(j,1) = 2;
end
row = row+row1;
v = cell2mat(c(8,1));
row1 = length(v);
for j = row+1:row1+row
    test_label(j,1) = 3;
end
row = row+row1;
v = cell2mat(c(10,1));
row1 = length(v);
for j = row+1:row1+row
    test_label(j,1) = 4;
end
row = row+row1;
v = cell2mat(c(12,1));
row1 = length(v);
for j = row+1:row1+row
    test_label(j,1) = 5;
end
row = row+row1;
v = cell2mat(c(14,1));
row1 = length(v);
for j = row+1:row1+row
    test_label(j,1) = 6;
end
row = row+row1;
v = cell2mat(c(16,1));
row1 = length(v);
for j = row+1:row1+row
    test_label(j,1) = 7;
end
row = row+row1;
v = cell2mat(c(18,1));
row1 = length(v);
for j = row+1:row1+row
    test_label(j,1) = 8;
end
row = row+row1;
v = cell2mat(c(20,1));
row1 = length(v);
for j = row+1:row1+row
    test_label(j,1) = 9;
end

train_data = double(train_data);
double(train_label);
validation_data = double(validation_data);
double(validation_label);
test_data = double(test_data);
double(test_label);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

