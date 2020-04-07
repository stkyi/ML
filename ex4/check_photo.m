Image3DmatrixRGB = imread("myOwnPhoto_2.jpg");
Image3DmatrixYIQ = rgb2ntsc(Image3DmatrixRGB);
Image2DmatrixBW = Image3DmatrixYIQ(:,:,1);
%imshow(Image2DmatrixBW, [-1, 1] );
% Get the size of your image
oldSize = size(Image2DmatrixBW);
% Obtain crop size toward centered square (cropDelta)
% ...will be zero for the already minimum dimension
% ...and if the cropPercentage is zero, 
% ...both dimensions are zero
% ...meaning that the original image will go intact to croppedImage
cropDelta = floor((oldSize - min(oldSize)) .* (20/100));
% Compute the desired final pixel size for the original image
finalSize = oldSize - cropDelta;
% Compute each dimension origin for croping
cropOrigin = floor(cropDelta / 2) + 1;
% Compute each dimension copying size
copySize = cropOrigin + finalSize - 1;
% Copy just the desired cropped image from the original B&W image
croppedImage = Image2DmatrixBW( ...
                    cropOrigin(1):copySize(1), cropOrigin(2):copySize(2));
% Resolution scale factors: [rows cols]
scale = [20 20] ./ finalSize;
% Compute back the new image size (extra step to keep code general)
newSize = max(floor(scale .* finalSize),1); 
% Compute a re-sampled set of indices:
rowIndex = min(round(((1:newSize(1))-0.5)./scale(1)+0.5), finalSize(1));
colIndex = min(round(((1:newSize(2))-0.5)./scale(2)+0.5), finalSize(2));
% Copy just the indexed values from old image to get new image
newImage = croppedImage(rowIndex,colIndex,:);
imshow(newImage, [-1, 1] );
% Rotate if needed: -1 is CCW, 0 is no rotate, 1 is CW
rotStep = 0;
newAlignedImage = rot90(newImage, rotStep);
% Invert black and white Invert black and white because it is easier to draw black digits over white background in our photos but the classifier needs white digits.
invertedImage = - newAlignedImage;

% Find min and max grays values in the image
maxValue = max(invertedImage(:));
minValue = min(invertedImage(:));
% Compute the value range of actual grays
delta = maxValue - minValue;

% Normalize grays between 0 and 1
normImage = (invertedImage - minValue) / delta;

% Add contrast. Multiplication factor is contrast control.
contrastedImage = sigmoid((normImage -0.5) * 5);
% Show image as seen by the classifier
%imshow(contrastedImage, [-1, 1] );

% Output the matrix as a unrolled vector
vectorImage = reshape(normImage, 1, newSize(1) * newSize(2));


