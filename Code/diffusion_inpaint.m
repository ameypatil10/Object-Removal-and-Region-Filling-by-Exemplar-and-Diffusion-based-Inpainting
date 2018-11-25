function [imgInpainted] = diffusion_inpaint(img, mask)  

% FolderPath = 'Images';
% ImageName = 'pole.jpg';
% MaskName = 'mask3.jpg';
% 
% img = imread(fullfile(FolderPath,ImageName));
% mask = imread(fullfile(FolderPath,MaskName)); 
% mask = im2bw(mask);
 mask = ~mask;
delta = 0.1;
if size(img,3) > 1
    imgMasked = im2double(img) .* im2double(repmat(mask,1,1,3)==1); 
    imgInpainted = rgb2gray(imgMasked);
else
    imgMasked = img .* im2double(mask); 
    imgInpainted = imgMasked;
end

u = imgInpainted;
dt = 0.5;
m = size(u,1);
n = size(u,2);

geps = diff(mask,dt);
u = anisodiff(u,dt,geps,1);
iter = 0;
loop = 1;
while loop && iter < 200
    iter = iter + 1
   for k = 1:40
       lap = laplacian(u);
       [lapIx, lapIy] = grad(lap);
       [uIx, uIy] = grad(u);
       temp = uIx; uIx = -uIy; uIy = temp;
       lapIx = reshape(lapIx,m,n);
       lapIy = reshape(lapIy,m,n);
       uIx = reshape(uIx,m,n);
       uIy = reshape(uIy,m,n);
       dI = (lapIx .* uIx) + (lapIy .* uIy);
       dIpos = dI>0;
       dIpos = reshape(dIpos,m*n,1);
       [Dif, Djf, Dib, Djb] = D(m,n);
       uxf = Dif*reshape(u,m*n,1);
       uxb = Dib*reshape(u,m*n,1);
       uyf = Djf*reshape(u,m*n,1);
       uyb = Djb*reshape(u,m*n,1);
       
       slopeLim = dIpos .* sqrt(min(uxb,0).^2 + max(uxf,0).^2 + min(uyb,0).^2 + max(uyf,0).^2)...
                + (~dIpos) .* sqrt(max(uxb,0).^2 + min(uxf,0).^2 + max(uyb,0).^2 + min(uyf,0).^2);
                 
       slopeLim = reshape(slopeLim,m,n);
       update = dI .* slopeLim;
       u = u + (dt * ~mask .* update);
   end
   
    un = anisodiff(u,dt,geps,1);
    u = ~mask.*un + mask.*u;  
    imshow(u);
    sum(sum(abs(un - u)))
    if(sum(sum(abs(un - u))) < delta)
        loop = 0;
    end
end



end   

function [Ix, Iy] = grad(I)
    m = size(I,1);
    n = size(I,2);

    % CENTERED
    d1i_centered  = spdiags([-ones(m,1),ones(m,1)],[-1,1],m,m)/(2);
    d1j_centered  = spdiags([-ones(n,1),ones(n,1)],[-1,1],n,n)/(2);
    d1i_centered([1 end],:) = 0;
    d1j_centered([1 end],:) = 0;
    
    Dic  = kron(speye(n),d1i_centered);
    Djc  = kron(d1j_centered,speye(m));
    
    Ix = Dic * reshape(I,m*n,1);
    Iy = Djc * reshape(I,m*n,1);
    reshape(Ix,m,n); reshape(Iy,m,n);
    D = sqrt(Ix.^2 + Iy.^2 + 1e-10);
    Ix = Ix ./ D;
    Iy = Iy ./D;
end

function [ laplacianImage ] = laplacian(I)
    m = size(I,1);
    n = size(I,2);
    d2i = toeplitz(sparse([1,1],[1,2],[-2,1],1,m));
    d2j = toeplitz(sparse([1,1],[1,2],[-2,1],1,n));
    % PERIODIC BOUNDARY CONDITIONS
    d2i(1,end) = 1;
    d2i(end,1) = 1;
    d2j(end,1) = 1;
    d2j(1,end) = 1;
    laplacianImage = (kron(speye(n),d2i)+kron(d2j,speye(m))) * reshape(I,m*n,1);
    laplacianImage = reshape(laplacianImage,m,n);
    
    
end

function [ diffGeps ] = diff(mask, dt)
    maskeps = 1-mask;
    diffGeps = maskeps;
    lambdaeps = imdilate(maskeps,strel('ball',6,0,0))-maskeps;
    L = laplacian(diffGeps);
    for t=1:5
        % just interpolate g_{epsilon} with a few steps of linear diffusion within the strip
        diffGeps = diffGeps + (dt .* L) + lambdaeps .* (maskeps - diffGeps);
    end
end

function [ u ] = anisodiff(u,dt,geps,N)
    gamma = 1;
    m = size(u,1);
    n= size(u,2);
for i=1:N
    % image gradients in NSEW direction
    uN=[u(1,:); u(1:m-1,:)]-u;
    uS=[u(2:m,:); u(m,:)]-u;
    uE=[u(:,2:n) u(:,n)]-u;
    uW=[u(:,1) u(:,1:n-1)]-u;

    cN=1./(1+(abs(uN)/gamma).^2);
    cS=1./(1+(abs(uS)/gamma).^2);
    cE=1./(1+(abs(uE)/gamma).^2);
    cW=1./(1+(abs(uW)/gamma).^2);

    u=u+dt*geps.*(cN.*uN + cS.*uS + cE.*uE + cW.*uW);
end
end

function [Dif, Djf, Dib, Djb] = D(m,n)
     % FORWARD AND BACKWARD
    d1i_forward  = spdiags([-ones(m,1),ones(m,1)],[0,1],m,m);
    d1j_forward  = spdiags([-ones(n,1),ones(n,1)],[0,1],n,n);
    d1i_backward = spdiags([-ones(m,1),ones(m,1)],[-1,0],m,m);
    d1j_backward = spdiags([-ones(n,1),ones(n,1)],[-1,0],n,n);

    % FOR BOUNDARY CONDITIONS
    d1i_forward(end,:) = 0;
    d1j_forward(end,:) = 0;
    d1i_backward(1,:)  = 0;
    d1j_backward(1,:)  = 0;

    Dif  = kron(speye(n),d1i_forward);
    Djf  = kron(d1j_forward,speye(m));
    Dib  = kron(speye(n),d1i_backward);
    Djb  = kron(d1j_backward,speye(m));
end
