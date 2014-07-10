function [obj_val obj_grad] = nnObjFunction(params, n_input, n_hidden, ...
                                    n_class, training_data,...
                                    training_label, lambda)
% nnObjFunction computes the value of objective function (negative log 
%   likelihood error function with regularization) given the parameters 
%   of Neural Networks, thetraining data, their corresponding training 
%   labels and lambda - regularization hyper-parameter.

% Input:
% params: vector of weights of 2 matrices w1 (weights of connections from
%     input layer to hidden layer) and w2 (weights of connections from
%     hidden layer to output layer) where all of the weights are contained
%     in a single vector.
% n_input: number of node in input layer (not include the bias node)
% n_hidden: number of node in hidden layer (not include the bias node)
% n_class: number of node in output layer (number of classes in
%     classification problem
% training_data: matrix of training data. Each row of this matrix
%     represents the feature vector of a particular image
% training_label: the vector of truth label of training images. Each entry
%     in the vector represents the truth label of its corresponding image.
% lambda: regularization hyper-parameter. This value is used for fixing the
%     overfitting problem.
       
% Output: 
% obj_val: a scalar value representing value of error function
% obj_grad: a SINGLE vector of gradient value of error function
% NOTE: how to compute obj_grad
% Use backpropagation algorithm to compute the gradient of error function
% for each weights in weight matrices.
% Suppose the gradient of w1 is 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reshape 'params' vector into 2 matrices of weight w1 and w2
% w1: matrix of weights of connections from input layer to hidden layers.
%     w1(i, j) represents the weight of connection from unit j in input 
%     layer to unit i in hidden layer.
% w2: matrix of weights of connections from hidden layer to output layers.
%     w2(i, j) represents the weight of connection from unit j in hidden 
%     layer to unit i in output layer.
w1 = reshape(params(1:n_hidden * (n_input + 1)), ...
                 n_hidden, (n_input + 1));

w2 = reshape(params((1 + (n_hidden * (n_input + 1))):end), ...
                 n_class, (n_hidden + 1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
input_count = size(training_data,1);
% adding bias in input
training_data = [ones(input_count,1) training_data];

% feed forward
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a = training_data * w1';
z = sigmoid(a);
z1 = z;
z = [ones(input_count,1) z];

b = z * w2';
y = sigmoid(b);

% creating matrix to keep training lables in 1 of k coding scheme
tk = zeros(input_count,n_class);
for i=1:input_count,
   tk(i,training_label(i)+1) = 1; 
end

% calculate the error
obj_val = -(1/input_count) * sum( sum(tk .* log(y) + (1-tk).* log(1-y)));

% regularize
wt1 = w1(:,2:size(w1,2));
wt2 = w2(:,2:size(w2,2));

reg = lambda * (sum(sum(wt1.^2)) + sum(sum(wt2.^2)))/(2*input_count);

obj_val = obj_val + reg;


% calculate gradient
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

grad_w1 = zeros(size(w1));
grad_w2 = zeros(size(w2));

%siga = sigmoid(a).*(1-sigmoid(a));
del_k = y - tk;
del_p = (del_k * w2);
del_p = del_p(:,2:end);
del_p = del_p .* (z1.*(1-z1));%siga;

grad_w1 = grad_w1 + del_p' * training_data;
grad_w2 = grad_w2 + del_k' * z;

w1tmp = w1;
w2tmp = w2;
w1tmp(:,1) = 0;
w2tmp(:,1) = 0;

grad_w1 = grad_w1/input_count + (lambda/input_count)*w1tmp;
grad_w2 = grad_w2/input_count + (lambda/input_count)*w2tmp;

% Suppose the gradient of w1 and w2 are stored in 2 matrices grad_w1 and grad_w2
% Unroll gradients to single column vector
obj_grad = [grad_w1(:) ; grad_w2(:)];

end
