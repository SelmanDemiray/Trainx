function [train_images, train_labels, test_images, test_labels] = load_data()
% LOAD_DATA Loads the MNIST dataset for training and testing.
%
% 	[TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS] = LOAD_DATA() loads the training and testing data
%	from the MNIST dataset. Returns the images as floats scaled from 0-1, and labels as one-hot vectors.
%
%	Code for BIO/NROD08 Assignment 2, Winter 2019
%	Author: Blake Richards, blake.richards@utoronto.ca

% open the files for reading
f_train_images = fopen('../mnist_data/train-images-idx3-ubyte','r','ieee-be');
f_train_labels = fopen('../mnist_data/train-labels-idx1-ubyte','r','ieee-be');
f_test_images = fopen('../mnist_data/t10k-images-idx3-ubyte','r','ieee-be');
f_test_labels = fopen('../mnist_data/t10k-labels-idx1-ubyte','r','ieee-be');

% set each file to the start (just to be sure)
fseek(f_train_images,0,-1);
fseek(f_train_labels,0,-1);
fseek(f_test_images,0,-1);
fseek(f_test_labels,0,-1);

%%%%%% - READ IN THE TRAINING IMAGES AND LABELS

% read the headers
magicnum = fread(f_train_images,1,'int32');
nimages  = fread(f_train_images,1,'int32');
nrows    = fread(f_train_images,1,'int32');
ncolumns = fread(f_train_images,1,'int32');
magicnum = fread(f_train_labels,1,'int32');
nimages  = fread(f_train_labels,1,'int32');

% initialize the data holders
train_images = zeros(nrows*ncolumns,nimages);
train_labels = zeros(10,nimages);

% read the images and labels
for i = 1:nimages
	current_image     = fread(f_train_images,784,'uint8');
	current_label     = fread(f_train_labels,1,'uint8');
	train_images(:,i) = double(current_image)./255.0;
	train_labels(current_label+1,i) = 1.0;
end

% close the files
fclose(f_train_images);
fclose(f_train_labels);

%%%%%% - READ IN THE TEST IMAGES AND LABELS

% read the headers
magicnum = fread(f_test_images,1,'int32');
nimages  = fread(f_test_images,1,'int32');
nrows    = fread(f_test_images,1,'int32');
ncolumns = fread(f_test_images,1,'int32');
magicnum = fread(f_test_labels,1,'int32');
nimages  = fread(f_test_labels,1,'int32');

% initialize the data holders
test_images = zeros(nrows*ncolumns,nimages);
test_labels = zeros(10,nimages);

% read the images and labels
for i = 1:nimages
	current_image    = fread(f_test_images,784,'uint8');
	current_label    = fread(f_test_labels,1,'uint8');
	test_images(:,i) = double(current_image)./255.0;
	test_labels(current_label+1,i) = 1.0;
end

% close the files
fclose(f_test_images);
fclose(f_test_labels);

% function end
end
