%%
clear all
clc

load(strcat('cifar-10-batches-mat/', 'data_batch_1.mat'));
data1 = uint8(data);
labels1 = labels;

% Select an image for testing
img_idx=151; % <--- the image you want to perform tests on. 

R=data1(img_idx, 1:1024);
G=data1(img_idx, 1025:2048);
B=data1(img_idx, 2049:3072);
img_rgb(:,:,1)=reshape(R,32,32);
img_rgb(:,:,2)=reshape(G,32,32);
img_rgb(:,:,3)=reshape(B,32,32);


%% 
% Brighten up the lights
img = img_rgb;
kernel = fspecial('prewitt');
edges = imfilter(img,kernel);
brighter = img + edges;
subplot(1,2,1),imshow(uint8(img)),title('Original image');
subplot(1,2,2),imshow(uint8(brighter)),title('Brightened image');

%% LoG filter - edge detection
color = img_rgb;
lg = fspecial('log');
c2 = imfilter(color,lg);
subplot(1,2,1),imshow(uint8(color)),title('Original image');
subplot(1,2,2),imshow(uint8(c2)),   title('Color LoG result');

%% contrast enhancement --> increase the range of pixel values.
%If the image has very little contrast and its color bands are highly correlated with each other.
%To enhance the image and acquire a more decent and life-like result, we can adjust the contrast of all the three channels.
%adjusted = img_rbg;
img = img_rgb;
for i=1:size(img_rgb,3)
    adjusted(:,:,i) = imadjust(img(:,:,i)); 
end
subplot(1,2,1),imshow(uint8(img)),       title('Original RGB image')
subplot(1,2,2),imshow(uint8(adjusted)),  title('Adjusted RGB image')

%% Median filter --> noise removal
%This difference causes the process of median filtering to be less sensitive to outliers
img = img_rgb;
for i=1:size(img_rgb,3)
    denoised(:,:,i) = medfilt2(img(:,:,i),[5,5]);
end
figure;
subplot(1,2,1),imshow(uint8(img)),      title('Original RGB image')
subplot(1,2,2),imshow(uint8(denoised)), title('Median filter RGB image')

%% analyze different color representations
%img=color;
img = img_rgb;
%we generate the HSV image:
img_hsv = rgb2hsv(img);
%convert our image to CIE-L*a*b*:
cform = makecform('srgb2lab'); % Make the transformstructure
img_lab = applycform(img,cform); % Apply transform

img=uint8(img);
img_hsv=uint8(img_hsv);
img_lab=uint8(img_lab);

subplot(3,4,1),imshow(img),title('RGB image')
subplot(3,4,2),imshow(img(:,:,1)),title('R channel')
subplot(3,4,3),imshow(img(:,:,2)),title('G channel')
subplot(3,4,4),imshow(img(:,:,3)),title('B channel')
subplot(3,4,5),imshow(img_hsv),title('HSV image')
subplot(3,4,6),imshow(img_hsv(:,:,1)),title('H channel')
subplot(3,4,7),imshow(img_hsv(:,:,2)),title('S channel')
subplot(3,4,8),imshow(img_hsv(:,:,3)),title('V channel')
subplot(3,4,9),imshow(img_lab),title('CIE-L*a*b* image')
subplot(3,4,10),imshow(img_lab(:,:,1)),title('L* channel')
subplot(3,4,11),imshow(img_lab(:,:,2)),title('a* channel')
subplot(3,4,12),imshow(img_lab(:,:,3)),title('b* channel')

%%

color = uint8(img_rgb);
red   = im2bw(color(:,:,1));          % Threshold red channel
green = im2bw(color(:,:,2));          % Threshold green channel
blue  = im2bw(color(:,:,3));          % Threshold blue channel
bin_image_or = red | green | blue;    % Find union using OR
bin_image_and = red & green & blue;   % Find intersection using AND
subplot(1,3,1),imshow(uint8(color)),              title('Original Image')
subplot(1,3,2),imshow(uint8(bin_image_or)),       title('Binary Union Image')
subplot(1,3,3),imshow(uint8(bin_image_and)),      title('Binary Intersection Image')

%% Put togeter mask with orginal image.
%bin_image_or(1:100,:) = 0;
mask = imdilate(bin_image_or, strel('disk', 1));

R = img(:,:,1); % store R channel in new matrix
G = img(:,:,2); % store G channel in new matrix
B = img(:,:,3); % store B channel in new matrix

img_gray = rgb2gray(img);
R(mask == 0) = img_gray(mask == 0);
G(mask == 0) = img_gray(mask == 0);
B(mask == 0) = img_gray(mask == 0);
% R(mask == 1) = img_gray(mask == 1);
% G(mask == 1) = img_gray(mask == 1);
% B(mask == 1) = img_gray(mask == 1);

img_final = cat(3,R,G,B);
subplot(2,2,1),imshow(uint8(img)),         title('Original image');
subplot(2,2,2),imshow(uint8(mask)),        title('Mask');
subplot(2,2,3),imshow(uint8(img_final)),   title('Processed image');

%%

