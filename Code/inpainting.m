% Inpainting via Criminski Algorithm

function [inpaintedImg,C,D] = inpainting(origImg,mask,psz)

% error check
if ~ismatrix(mask); error('Invalid mask'); end
if sum(sum(mask~=0 & mask~=1))>0; error('Invalid mask'); end
if mod(psz,2)==0; error('Patch size psz must be odd.'); end

fillRegion = mask;

origImg = im2double(origImg);
img = origImg;
ind = img2ind(img);
sz = [size(img,1) size(img,2)];
sourceRegion = ~fillRegion;

% Initialize isophote values
[Ix(:,:,3), Iy(:,:,3)] = gradient(img(:,:,3));
[Ix(:,:,2), Iy(:,:,2)] = gradient(img(:,:,2));
[Ix(:,:,1), Iy(:,:,1)] = gradient(img(:,:,1));
Ix = sum(Ix,3)/(3*255); Iy = sum(Iy,3)/(3*255);
temp = Ix; Ix = -Iy; Iy = temp;  % Rotate gradient 90 degrees

% Initialize confidence and data terms
C = im2double(sourceRegion);
D = repmat(-.1,sz);
iter = 1;

% Seed 'rand' for reproducible results (good for testing)
rand('state',0);

% Loop until entire fill region has been covered
while any(fillRegion(:))
  % Find contour & normalized gradients of fill region
  fillRegionD = im2double(fillRegion);
  dR = find(conv2(fillRegionD,[1,1,1;1,-8,1;1,1,1],'same')>0);
  
  [Nx,Ny] = gradient(im2double(~fillRegion));
  N = [Nx(dR(:)) Ny(dR(:))];
  N = normr(N);  
  N(~isfinite(N))=0; % handle NaN and Inf
  
  % Compute confidences along the fill front
  for k=dR'
    Hp = getpatch(sz,k,psz);
    q = Hp(~(fillRegion(Hp))); % fillRegion
    C(k) = sum(C(q))/numel(Hp);
  end
  
  % Compute patch priorities = confidence term * data term
  D(dR) = abs(Ix(dR).*N(:,1)+Iy(dR).*N(:,2)) + 0.001;
  priorities = C(dR).* D(dR);
  
  % Find patch with maximum priority, Hp
  [~,ndx] = max(priorities(:));
  p = dR(ndx(1));
  [Hp,rows,cols] = getpatch(sz,p,psz);
  toFill = fillRegion(Hp);
  
  % Find exemplar that minimizes error, Hq
  Hq = bestexemplar(img,img(rows,cols,:),toFill',sourceRegion);
  
  % Update fill region
  toFill = logical(toFill);                 
  fillRegion(Hp(toFill)) = false;
  
  % Propagate confidence & isophote values
  C(Hp(toFill))  = C(p);
  Ix(Hp(toFill)) = Ix(Hq(toFill));
  Iy(Hp(toFill)) = Iy(Hq(toFill));
  
  % Copy image data from Hq to Hp
  ind(Hp(toFill)) = ind(Hq(toFill));
  img(rows,cols,:) = ind2img(ind(rows,cols),origImg);  
  
  iter = iter+1;
  
  imshow(img .* (repmat(fillRegion,1,1,3)==0));
  p = 100*sum(sum(fillRegion))/ (size(fillRegion,1) * size(fillRegion, 2))
  'Percent Inpainting Remaining = '
  p
end
'Done ImPainting'
inpaintedImg = img;


%---------------------------------------------------------------------
% Scans over the entire image (with a sliding window)
%---------------------------------------------------------------------
function Hq = bestexemplar(img,Ip,toFill,sourceRegion)
m=size(Ip,1); mm=size(img,1); n=size(Ip,2); nn=size(img,2);
best = bestexemplarhelper(mm,nn,m,n,img,Ip,toFill,sourceRegion);
Hq = sub2ndx(best(1):best(2),(best(3):best(4))',mm);


%---------------------------------------------------------------------
% Returns the indices for a psz x psz patch centered at pixel p.
%---------------------------------------------------------------------
function [Hp,rows,cols] = getpatch(sz,p,psz)
w=(psz-1)/2; p=p-1; y=floor(p/sz(1))+1; p=rem(p,sz(1)); x=floor(p)+1;
rows = max(x-w,1):min(x+w,sz(1));
cols = (max(y-w,1):min(y+w,sz(2)))';
Hp = sub2ndx(rows,cols,sz(1));

%---------------------------------------------------------------------
% Converts the (rows,cols) subscript-style indices to Colunm major form
%---------------------------------------------------------------------
function N = sub2ndx(rows,cols,nTotalRows)
X = rows(ones(length(cols),1),:);
Y = cols(:,ones(1,length(rows)));
N = X+(Y-1)*nTotalRows;


%---------------------------------------------------------------------
% Converts an indexed image into an RGB image
%---------------------------------------------------------------------
function img2 = ind2img(ind,img)
for i=3:-1:1, temp=img(:,:,i); img2(:,:,i)=temp(ind); end;


%---------------------------------------------------------------------
% Converts an RGB image into a indexed image
%---------------------------------------------------------------------
function ind = img2ind(img)
s=size(img); ind=reshape(1:s(1)*s(2),s(1),s(2));

