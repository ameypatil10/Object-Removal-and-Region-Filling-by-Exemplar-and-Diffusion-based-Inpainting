%% Image InPainting based on Exemplar method
clear
clc
psz = 9;       

FolderPath = 'Images';
ImageName = 'blue-green.jpg';
img = imread(fullfile(FolderPath,ImageName));
mask = ones(416, 555);
mask(150:180, 430:460) = 0;
mask(280:310, 320:350) = 0;
maskedImg = img .* uint8(repmat(mask == 0, 1, 1, 3));

[imgInpainted, Confidence, Data] = inpainting(img, logical(~mask), psz);
imwrite(imgInpainted, 'Images/EXIN-blue-green.jpg');
imwrite(~mask, 'Images/MS-blue-green.jpg');

%%
% clear
% clc
% psz = 9;       
% 
% FolderPath = 'Images';
% ImageName = 'dustbin.jpg';
% img = imread(fullfile(FolderPath,ImageName));
% mask = ones([size(img,1), size(img,2)]);
% mask(190:220, 210:300) = 0;
% mask(275:320, 130:170) = 0;
% maskedImg = img .* uint8(repmat(mask == 0, 1, 1, 3));
% 
% [imgInpainted, Confidence, Data] = inpainting(img, logical(~mask), psz);


%%
clear
clc
psz = 9;       

FolderPath = 'Images';
ImageName = 'pole.jpg';
MaskName = 'mask2.jpg';
img = imread(fullfile(FolderPath,ImageName));
mask = imread(fullfile(FolderPath,MaskName)); mask = im2bw(mask);
maskedImg = img .* uint8(repmat(mask == 0, 1, 1, 3));

[imgInpainted, Confidence, Data] = inpainting(img, logical(~mask), psz);
imwrite(imgInpainted, 'Images/EXIN-pole.jpg');
imwrite(~mask, 'Images/MS-pole.jpg');

%%
clear
clc
psz = 9;       

FolderPath = 'Images';
ImageName = 'lena.bmp';
MaskName = 'mask.bmp';
img = imread(fullfile(FolderPath,ImageName));
mask = imread(fullfile(FolderPath,MaskName)); mask = im2bw(mask);

maskedImg = img .* uint8(repmat(mask == 0, 1, 1, 3));

[imgInpainted, Confidence, Data] = inpainting(img, logical(~mask), psz);
imwrite(imgInpainted, 'Images/EXIN-lena.jpg');
imwrite(~mask, 'Images/MS-lena.jpg');

%%
% clear
% clc
% psz = 9;       
% 
% FolderPath = 'Images';
% ImageName = 'blue-green.jpg';
% 
% img = imread(fullfile(FolderPath,ImageName));
% mask = ones(416, 555);
% mask(190:240, 95:130) = 0;
% mask(190:240, 430:460) = 0;
% maskedImg = img .* uint8(repmat(mask == 0, 1, 1, 3));
% 
% [imgInpainted, Confidence, Data] = inpainting(img, logical(~mask), psz);

%%
% clear
% clc
% psz = 25;       
% 
% FolderPath = 'Images';
% ImageName = 'square.png';
% 
% img = imread(fullfile(FolderPath,ImageName));
% mask = ones([size(img,1), size(img,2)]);
% mask(113:344, 152:387) = 0;
% maskedImg = img .* uint8(repmat(mask == 0, 1, 1, 3));
% [imgInpainted, Confidence, Data] = inpainting(img, logical(~mask), psz);

%%
clear 
clc
f = phantom(256);
my_theta = 0:1:177;
Rf = radon(f,my_theta);
BRf = iradon(Rf,my_theta,'linear','none',1,256);
w_max = 1;
RRMSE = @(A,B) [sqrt(sum(sum((A-B) .^ 2))) ./ sqrt(sum(sum(A .^ 2)))];
psz = 15;
%figure(1); show_colormap(Rf);title('Radon Transform for phantom 256');hold off;
%figure(2); show_colormap(BRf);title('Back Projection of Radon Transform for phantom 256');hold off;
img = repmat(Rf, 1, 1, 3);
multiplier = 255 / max(max(Rf));
img = multiplier * img;
mask = ones([size(img,1), size(img,2)]);
mask(50:310, 150:160) = 0;
figure(3);
[imgInpainted, Confidence, Data] = inpainting(img, logical(~mask), psz);
imgInpainted = imgInpainted(:,:,1) / multiplier;
imwrite(imgInpainted, 'Images/EXIN-sinogram.jpg');
imwrite(~mask, 'Images/MS-sinogram.jpg');
Inpainted_IR = iradon(imgInpainted,my_theta,'linear','none',1,256);
imwrite(Inpainted_IR, 'Images/EXIN-phantom.jpg');

%% Image InPainting using Diffusion Method
clear
clc

FolderPath = 'Images';
ImageName = 'lena.bmp';
MaskName = 'mask.bmp';
img = imread(fullfile(FolderPath,ImageName));
mask = imread(fullfile(FolderPath,MaskName)); mask = im2bw(mask);

maskedImg = img .* uint8(repmat(mask == 0, 1, 1, 3));
imgInpainted = diffusion_inpaint(img, logical(~mask));

imwrite(imgInpainted, 'Images/DFIN-lena.jpg');

%%
clear
clc

FolderPath = 'Images';
ImageName = 'pole.jpg';
MaskName = 'mask2.jpg';
img = imread(fullfile(FolderPath,ImageName));
mask = imread(fullfile(FolderPath,MaskName)); mask = im2bw(mask);

maskedImg = img .* uint8(repmat(mask == 0, 1, 1, 3));

imgInpainted = diffusion_inpaint(img, logical(~mask));
imwrite(imgInpainted, 'Images/DFIN-pole.jpg');

%%
clear 
clc
f = phantom(256);
my_theta = 0:1:177;
Rf = radon(f,my_theta);
BRf = iradon(Rf,my_theta,'linear','none',1,256);
RRMSE = @(A,B) [sqrt(sum(sum((A-B) .^ 2))) ./ sqrt(sum(sum(A .^ 2)))];
%figure(1); show_colormap(Rf);title('Radon Transform for phantom 256');hold off;
%figure(2); show_colormap(BRf);title('Back Projection of Radon Transform for phantom 256');hold off;
img = Rf;
mask = ones([size(img,1), size(img,2)]);
mask(60:300, 150:160) = 0;

imgInpainted = diffusion_inpaint(img, logical(~mask));