function label = nnPredict(w1, w2, data)
% nnPredict predicts the label of data given the parameter w1, w2 of Neural
% Network.

% Input:
% w1: matrix of weights of connections from input layer to hidden layers.
%     w1(i, j) represents the weight of connection from unit j in input 
%     layer to unit j in hidden layer.
% w2: matrix of weights of connections from hidden layer to output layers.
%     w2(i, j) represents the weight of connection from unit j in input 
%     layer to unit j in hidden layer.
% data: matrix of data. Each row of this matrix represents the feature 
%       vector of a particular image
       
% Output: 
% label: a column vector of predicted labels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

input_count = size(data,1);
% adding bias in input
data = [ones(input_count,1) data];

% feed forward
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a = data * w1';
z = sigmoid(a);
z = [ones(input_count,1) z];

b = z * w2';
y = sigmoid(b);

[val, label] = max(y, [], 2);
label=label-1;

end
