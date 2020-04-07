function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values

% number of training examples
m = size(y,1);
% You need to return the following variables correctly 



t = [0,theta(2:end)'];
J=(1/m)*(-y'*log(sigmoid(X*theta)) - (1-y)'*log(1-sigmoid(X*theta))) + (lambda/(2*m))*(t*t');
grad0 = (1/m) * (sigmoid(X*theta)'*X - y'*X);

grad1 = (1/m) * (sigmoid(X*theta)'*X - y'*X) + (lambda/m)*t;
grad = [grad0(1),grad1(2:end)];
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
