function [fighandle,imxy,imyz,imxz] = imshow_ortho(vol,scaling,figtitle)

if nargin < 2
    scaling = [min(vol(:)),max(vol(:))];
    figtitle = '';
elseif nargin < 3
    figtitle = '';
end


imxy = vol(:,:,round(end/2));
imyz = rot90(squeeze(vol(:,round(end/2),:)),-1);
imxz = rot90(squeeze(vol(round(end/2),:,:)),-1);

% fighandle = figure;
% subplot(131), imagesc(imxy,scaling), axis equal tight off, colormap gray, title('xy')
% subplot(132), imagesc(imyz,scaling), axis equal tight off, colormap gray, title('yz')
% subplot(133), imagesc(imxz,scaling), axis equal tight off, colormap gray, title('xz')
% sgtitle(figtitle)

fighandle = figure; ha = tight_subplot(1,3,0.05,0.05,0.05);
axes(ha(1)), imagesc(imxy,scaling), axis equal tight off, colormap gray, title('xy')
axes(ha(2)), imagesc(imyz,scaling), axis equal tight off, colormap gray, title('yz')
axes(ha(3)), imagesc(imxz,scaling), axis equal tight off, colormap gray, title('xz')
sgtitle(figtitle); 
fighandle.Position = [fighandle.Position(1),fighandle.Position(2),fighandle.Position(3)*2,fighandle.Position(4)*1.25];






end

