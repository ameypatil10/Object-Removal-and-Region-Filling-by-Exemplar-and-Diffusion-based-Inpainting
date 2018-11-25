function show_colormap(Img, t, theta)
    hold on;
    if ~exist('theta','var') || ~exist('t','var')
          theta = 1:size(Img,2);
          t = 1:size(Img,1);
    end
    imagesc(t,theta,Img)
    colormap(gca,hot), colorbar
end