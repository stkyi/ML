function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_vec =  [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
val = 1;
for i=1:size(sigma_vec,1)
    for j=1:size(C_vec,1)
        model = svmTrain(X, y, C_vec(j), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(i))); 
        
        val_t = mean(double(svmPredict(model,Xval) ~= yval));
        
        if val_t <= val
            val = val_t;
            C = C_vec(j);
            sigma = sigma_vec(i);
            
        end
    end
end

end
