function show_images(images,labels)
% SHOW_IMAGES Displays all the receptive field structures as gray images.
%
% 	SHOW_IMAGES(IMAGES,LABELS) displays a subset of the images in the IMAGES matrix, 10 from each 
%	category as specified in LABELS.
%
%	See also SHOW_WEIGHTS.
%
%	Code for BIO/NROD08 Assignment 2, Winter 2019
%	Author: Blake Richards, blake.richards@utoronto.ca

% check the images argument
if ~isnumeric(images) || size(images,1) ~= 784
	error('You must provide the IMAGES matrix, which is a 784 x nimages matrix.');
end

% check the labels argument
if ~isnumeric(labels) || size(labels,1) ~= 10
	error('You must provide the LABELS matrix, which is a 10 x nimages matrix.');
end

% create a figure
figure();

% set the colormap to black and white
colormap('gray');

% select a random subset of ten images from each image category
selection = zeros(10,10);
for cat = 1:10
	[m,categories]   = max(labels,[],1);
	thiscategory     = find(categories == cat);
	selection(cat,:) = thiscategory(randperm(length(thiscategory),10));
end

% step through all categories and selected images and plot them
for r = 1:10
	for c = 1:10
		imagenumber = (r-1)*10 + c;                              
		subplot(10,10,imagenumber,'align');           
		imagesc(reshape(images(:,selection(r,c)),28,28)');   
		axis equal; axis off;                             
	end
end

% set the title for the image
set(gcf,'numbertitle','off','name','Sample images');

% function end
end
