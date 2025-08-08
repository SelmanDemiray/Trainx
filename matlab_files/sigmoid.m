function phi = sigmoid(x)
% SIGMOID Calculates the sigmoid function on each element of a matrix.
%
% 	PHI = SIGMOID(X) calculates 1/(1+exp(-x)) for each element of the matrix X
%	and returns it as PHI.
%
%	See also BACKWARD_PASS.
%
%	Code for BIO/NROD08 Assignment 2, Winter 2019
%	Author: Blake Richards, blake.richards@utoronto.ca

% calculate the sigmoid
phi = 1./(1+exp(-x)); % TO-DO: FILL IN THE EQUATION FOR THE SIGMOID ACTIVATION FUNCTION

% function end
end
