%% Histology to microCT registration
% Semi-automatic approach with manually defined correspondences
% for rat jaw, see publications: 10.3390/app12126286
%
% 13/10/2022 CT flag extraFigs, use filenames, include registration
%               commands, output processing steps to user

extraFigs = 0;  % 1: show additional figures
rerun = 0;      % 1: rerun non-rigid registration even if it exists already

%% 1. Toolboxes and directories
% % Add toolboxes
addpath(genpath('./utils/'))

% % directories where data is located
synch_dir = './example/data/';
histo_dir = './example/data/';

%% 2. Metadata for uCT and histology datasets
% % uCT
pixsize_raw = 5.114*1e-6; % isotropic pixels
bf = 2;
pixsize = pixsize_raw*bf; % [m]

% % histology
pixsize_h_raw = 0.243094*1e-6; % [m]
bf_h_raw = 1;
pixsize_h = pixsize_h_raw*bf_h_raw;

%% 3. Import data, crop, coarse rotation, and scaling
disp('3. Import data - takes some time')
% % uCT
% Note about uCT gray scale:
% uint8 28 is equal to -0.00018016 (absorption per pixel length)
% uint8 97 is equal to 0.00109926 (absorption per pixel length)
% this function converts 1-255 gray scale to absorption in units of 1/cm:
i8_2_1ocm = @(x) ((((0.00109926+0.00018016)*(x-28)/(97-28))-0.00018016)/pixsize)/100;

% % load uCT data
h = hdr_read_header([synch_dir 'Tier1R_Cal_Synchrotron_2010_bmc02_bin2.hdr']);
vol = double(hdr_read_volume(h));

% % Coarse pre-alignment
% This could also be done interactively with ITKSnap registration tool, then transformix
%   we will use matlab here, note that I want to use only one interpolation step
interpmethod = 'linear'; %'cubic';
rx = 0; ry = 45; rz = -20; % I found these values manually
rotm = eul2tform(deg2rad([rx,ry,rz]),'XYZ'); % rotation matrix
tform = affine3d(rotm);
vol = permute(vol,[3,2,1]);
vol = fliplr(vol);
% apply transformation
tmp = size(vol);
vol = imwarp(vol,tform,interpmethod);
win = centerCropWindow3d(size(vol),tmp);
r = win.YLimits(1):win.YLimits(2);
c = win.XLimits(1):win.XLimits(2);
p = win.ZLimits(1):win.ZLimits(2);
vol = vol(r,c,p);
% display result
%imshow_ortho(vol,[0,255],'');

% % Course cropping of uCT
ccx = [100,450];
ccy = [1,425];
ccz = [310,525];
vol = vol(ccy(1):ccy(2),ccx(1):ccx(2),ccz(1):ccz(2));

% % histology
% % load slide
histo_full = imread([histo_dir '5752_13_Wholeslide_Default_Extended_compressed.tif']);

% % resizing to approximately match pixel size
histo = imresize(histo_full,pixsize_h/pixsize);

% % course cropping of histology
hcx = [100,800];
hcy = [1,1905];
histo = histo(hcy(1):hcy(2),hcx(1):hcx(2),:);

% % stain normalization -- code is not mine, comes from https://doi.org/10.1111/jmi.12001
[Inorm, H, E] = normalizeStaining(histo);
Hgs = rgb2gray(H);
Egs = rgb2gray(E);
histo_gs = rgb2gray(Inorm); % grayscale normalized histology

%% 4. Matching points between histology and uCT
% % Domain expert can find matched features and document coordinates in a
% % spreadsheet, see manual_matched_points_matlabimport.xlsx

% % An alternative would be to use an automatic feature finder to find 
% % matching points, e.g. SURF or SIFT
disp('4. Manually match points')
% % points were find by using the following commands and data pointers
figure, imshow3D(vol,[75,245])
%figure, imagesc(histo), axis equal
figure, imshow(histo,'border', 'tight')   % alternative to show larger
%% 5. Load matched points
matchedPointsFile = [histo_dir 'manual_matched_points_matlabimport.xlsx'];
if ~exist(matchedPointsFile,'file')
    disp('define point correspondences')
    return
end
disp('5. Load manually matched points')
% import manual_matched_points_matlabimport.xlsx
opts = spreadsheetImportOptions("NumVariables", 7);
opts.Sheet = "Sheet1";
opts.DataRange = "A2:G51";
opts.VariableNames = ["mpi", "his_x", "his_y", "uct_x", "uct_y", "uct_z", "uct_w"];
opts.SelectedVariableNames = ["mpi", "his_x", "his_y", "uct_x", "uct_y", "uct_z", "uct_w"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double"];
manualmatchedpointsmatlabimport = readtable(matchedPointsFile, opts, "UseExcel", false);
clear opts
matchedpoints = table2array(manualmatchedpointsmatlabimport);
% clean up data 
matchedpoints(46,:) = []; % exclude bad points
matchedpoints(43,:) = [];
matchedpoints(30,:) = [];
matchedpoints(9,:) = [];
%
mpi = matchedpoints(:,1); % point index
his_x = matchedpoints(:,2); % histology x
his_y = matchedpoints(:,3);
uct_x = matchedpoints(:,4); % uct x
uct_y = matchedpoints(:,5);
uct_z = matchedpoints(:,6);
uct_w = matchedpoints(:,7); uct_w = 1./uct_w; % point weight uct (1: good, 0.5: medium)
% clean up data
uct_x(39-2) = 243; uct_y(39-2) = 140; % fix one point

%% 6. Extract surface fit to these points
disp('6. Extract surface based on points')

% % fit a surface to these points
[uct_sy,uct_sx,uct_sz] = size(vol);
[his_sy,his_sx] = size(histo);
% poly11
%[fit11,gof11] = fit([uct_x,uct_y],uct_z,'poly11','Weight',uct_w);
% poly22
%[fit22,gof22] = fit([uct_x,uct_y],uct_z,'poly22','Weight',uct_w);
% poly33
%[fit33,gof33] = fit([uct_x,uct_y],uct_z,'poly33','Weight',uct_w);
% thin-plate-spline:
[fit_tps,gof_tps] = fit([uct_x,uct_y],uct_z,'thinplateinterp','Normalize','on');

% % extract surface from fit
[uct_XX,uct_YY] = meshgrid(1:uct_sx,1:uct_sy);
%uct_ZZ = fit11(uct_XX,uct_YY);
%uct_ZZ = fit22(uct_XX,uct_YY);
uct_ZZ = fit_tps(uct_XX,uct_YY);

exslice = interp3(vol,uct_XX,uct_YY,uct_ZZ,'linear');
exslice(isnan(exslice)) = 0;

if extraFigs
    % show surface fitted to landmarks
    figure
    mesh(uct_XX, uct_YY, uct_ZZ)
    hold on
    plot3(uct_x,uct_y,uct_z,'k*')
    hold off
    xlabel('x');ylabel('y');zlabel('z')
    axis equal tight
    title(['TPS surface fitted to ' num2str(length(uct_x)) ' landmarks'])
    % show extracted slice
    figure
    imshow(exslice/max(exslice(:)),'Border','tight'); 
end

%% 7. 2D-2D pre-alignment (feature-based affine)
% % Ideally, one would now do a true 2d-3d registration, where the volume
% % is warped. In this case, we extract the surface and warp the 2d extracted
% % surface
disp('7. 2D-2D pre-alignment based on points')

% % find nearest points in uct extracted slice
[K,dist] = dsearchn([uct_XX(:),uct_YY(:),uct_ZZ(:)],[uct_x,uct_y,uct_z]);
uct_x_p = uct_XX(K);
uct_y_p = uct_YY(K);
uct_z_p = uct_ZZ(K);

if extraFigs
    % checking if this is correct for a single point
    i = 20;
    point = [uct_x(i),uct_y(i),uct_z(i)];
    dist2plane = sqrt((point(1)-uct_XX).^2 + (point(2)-uct_YY).^2 + (point(3)-uct_ZZ).^2);
    figure
    imagesc(dist2plane); axis equal tight
    hold on
    plot(point(1),point(2),'y*')
    hold off
    title('checking distance point->surface')
end

% % pre-align with point matching
% use these points (in case want to filter out certain points)
his_x2 = his_x;
his_y2 = his_y;
uct_x2 = uct_x_p;
uct_y2 = uct_y_p;

% % align these points with the histo points
% % point-based registration based on geometrical transform
% % register uCT -> histology

% rigid:
transformationType = 'nonreflectivesimilarity';
tform = fitgeotrans([uct_x2,uct_y2],[his_x2,his_y2],transformationType);
exslice_rigid = imwarp(exslice,tform,'OutputView',imref2d(size(histo)));
[uct_x2_rigid,uct_y2_rigid] = transformPointsForward(tform,uct_x2,uct_y2);

% affine:
transformationType = 'affine';
tform = fitgeotrans([uct_x2,uct_y2],[his_x2,his_y2],transformationType);
exslice_affine = imwarp(exslice,tform,'OutputView',imref2d(size(histo)));
[uct_x2_affine,uct_y2_affine] = transformPointsForward(tform,uct_x2,uct_y2);

% measure distance between feature points
dist_rigid = sqrt((uct_x2_rigid-his_x2).^2+(uct_y2_rigid-his_y2).^2);
dist_affine = sqrt((uct_x2_affine-his_x2).^2+(uct_y2_affine-his_y2).^2);

if extraFigs
    % optional plotting of landmark pairs
    if (1==1)
        % only show for landmark region
        maxY=ceil(max([uct_y2_rigid; his_y2]))+10;
        minY=ceil(min([uct_y2_rigid; his_y2]))-10;
        figure
        imshowpair(exslice_rigid(minY:maxY,:),histo(minY:maxY,:,:))
        hold on, plot(uct_x2_rigid,uct_y2_rigid-minY+1,'o')
        plot(his_x2,his_y2-minY+1,'o')
        plot([uct_x2_rigid,his_x2]',[uct_y2_rigid-minY+1,his_y2-minY+1]','k-')
        legend({'uct','histo'})
        title(['nonreflectivesimilarity, ' num2str(mean(dist_rigid),'%.1f') 'px'])
        drawnow
        figure
        imshowpair(exslice_affine(minY:maxY,:),histo(minY:maxY,:,:))
        hold on, plot(uct_x2_affine,uct_y2_affine-minY+1,'o')
        plot(his_x2,his_y2-minY+1,'o')
        plot([uct_x2_affine,his_x2]',[uct_y2_affine-minY+1,his_y2-minY+1]','k-')
        legend({'uct','histo'})
        title(['affine, ' num2str(mean(dist_affine),'%.1f') 'px'])
        % visualize motion via comparing transformed moving image
        figure
        imshowpair(exslice_rigid(minY:maxY,:),exslice_affine(minY:maxY,:))
        title('n.r.similarity vs. affine')
    else
        % full images
        figure, subplot(121), imshowpair(exslice_rigid,histo)
        hold on, plot(uct_x2_rigid,uct_y2_rigid,'o')
        plot(his_x2,his_y2,'o')
        plot([uct_x2_rigid,his_x2]',[uct_y2_rigid,his_y2]','k-')
        legend({'uct','histo'})
        title('nonreflectivesimilarity')
        subplot(122), imshowpair(exslice_affine,histo)
        hold on, plot(uct_x2_affine,uct_y2_affine,'o')
        plot(his_x2,his_y2,'o')
        plot([uct_x2_affine,his_x2]',[uct_y2_affine,his_y2]','k-')
        legend({'uct','histo'})
        title('affine')
    end
end

if (1==0)
    % % register histology -> uCT
    transformationType = 'affine';
    itform = fitgeotrans([his_x2,his_y2],[uct_x2,uct_y2],transformationType);
    histo_gs = double(rgb2gray(histo));
    histo_affine = imwarp(histo_gs,itform,'OutputView',imref2d([size(vol,1),size(vol,2)]),'FillValues',256);
    [his_x2_affine,his_y2_affine] = transformPointsForward(itform,his_x2,his_y2);
    dist_histo_affine = sqrt((uct_x2-his_x2_affine).^2+(uct_y2-his_y2_affine).^2);
    if extraFigs
        figure
        imshowpair(exslice,histo_affine)
        hold on, plot(uct_x2,uct_y2,'o')
        plot(his_x2_affine,his_y2_affine,'o')
        plot([uct_x2,his_x2_affine]',[uct_y2,his_y2_affine]','k-')
        legend({'uct','histo'})
        title(['hist->CT affine, ' num2str(mean(dist_histo_affine),'%.1f') 'px'])
    end
end

%% 8. 2D-2D non-rigid registration (with elastix) 
disp('8. 2D-2D non-rigid registration (with elastix)')

% define a crop for relevant area to be registered
cyi = 315;
cyf = 1165;
cxi = 2;
cxf = 701;

% % prep and export images and points
reghist = uint8(255-histo_gs(cyi:cyf,cxi:cxf)); % inverting colors
reguct = uint8(exslice_affine(cyi:cyf,cxi:cxf));

if extraFigs
    figure
    montage([reghist reguct])
    title('affine aligned slices to be registered')
end

regDir = './example/elastix2/';
histoFname = [regDir 'histology.mha'];   % fixed image
uCTfname = [regDir 'microct.mha'];       % moving image

writemetaimagefile(histoFname,reghist,[1,1],[0,0]);
writemetaimagefile(uCTfname,reguct,[1,1],[0,0]);

% % Run elastix 
% I used a very simple registration, could be adapted for improved results
% $ cd ./example/elastix/
% $ elastix -f ./histology.mha -m ./microct.mha -out ./ -p ./param_nonrigid_mi.txt
% here as unix command assuming known by name elastix
parFname = [regDir 'param_nonrigid_mi.txt']; % parameter file for non-rigid registration

% The resulting transformation
tparam = [regDir 'TransformParameters.0.txt'];
if ~exist(tparam,'file') || rerun
    % do non-rigid registration
    %[status,res]=unix(['elastix -f ' histoFname ' -m ' uCTfname ' -out ' regDir ' -p ' parFname]);
    [status,result] = callelastix(histoFname,uCTfname,regDir,parFname);
end

% in case you need to get spatial jacobian or want to warp another image, use these: 
if (1==0)
    calltransformix(uCTfname,tparam,regDir); % warps mha, output result.mha
    calltransformix('',tparam,regDir); % produces spatialJacobian.mha
end

% read non-rigid registration result
uCTRegFname = [regDir 'result.0.mha'];   % non-rigid registration result
reguct_nr = mha_read_volume(uCTRegFname); % result of non-rigid registration

% % transforming points with elastix
% if desired, could use points in registration -- I didn't do this
[sy,sx] = size(reguct);
uctpts = [uct_x2_affine-cxi,uct_y2_affine-cyi];

matlab2itk = @(coords) [coords(:,2)-1,coords(:,1)-1];
itk2matlab = @(coords) [coords(:,2)+1,coords(:,1)+1];

pointsFile = './example/elastix/microct_pts.txt';
fileID = fopen(pointsFile,'w');
fprintf(fileID,'index\n');
fprintf(fileID,'%d\n',length(uctpts));
fprintf(fileID,'%d %d\r\n',matlab2itk(uctpts)');

% % note: we should use inverse for sending points from moving to fixed
% to get inverse, run following elastix registration:
% $ cd ./example/elastix/
% $ elastix -f ./histology.mha -m ./histology.mha -out ./inverse/ -p ./inverse/param_inverse.txt -t0 ./TransformParameters.0.txt
% then edit the resulting TransformParameters file and remove the initial
% transform, replacing with "NoInitialTransform"
invRegDir = [regDir 'inverse/'];
invParFname = [invRegDir 'param_inverse.txt'];
invtparam = [invRegDir 'TransformParameters.0.txt'];

if ~exist(invtparam,'file') || rerun
    % get inverse transformation
    [status,res]=unix(['elastix -f ' histoFname ' -m ' histoFname ' -out ' invRegDir ' -p ' parFname ' -t0 ' tparam]);

    disp(['NOW edit ' invtparam])
    disp('replace filename in InitialTransformParametersFileName with "NoInitialTransform" to read')
    disp('(InitialTransformParametersFileName "NoInitialTransform")')
    input('when finished, press any key to continue ','s');
end

calltransformix(pointsFile,invtparam,regDir); % transform points, outputponts.txt

pointsFileTransformed = [regDir 'outputpoints.txt'];
t = readtable(pointsFileTransformed);
regpts1 = itk2matlab([t.Var28,t.Var29]);
uct_x2_nonrigid = regpts1(:,1);
uct_y2_nonrigid = regpts1(:,2);

%% 9. Figures and visualizations
disp('9. Figures and visualization')

figdir = './example/figures/';

prepimage = @(im,vr) uint8(255*(double(im)-vr(1))/(vr(2)-vr(1)));

% % grayscale range for displaying images
vr_synch = [80,215];

% % define colors
bmc_red = [198,19,24]/255;
bmc_green = [14,152,46]/255;

% % plot histology and extracted microCT with matched points
hcrop = [1,700,286,1135]; % crop for histology slice
ms = 6; % marker size

figure, subplot(121), imagesc(exslice,vr_synch), axis equal tight off, colormap gray
hold on
plot(uct_x,uct_y,'s','Color',bmc_red,'MarkerSize',ms)
subplot(122) 
yax = hcrop(3):hcrop(4); xax = hcrop(1):hcrop(2);
imagesc(xax,yax,Inorm(yax,xax,:)), axis equal tight off
hold on
plot(his_x,his_y,'h','Color',bmc_green,'MarkerSize',ms)
f = gcf; f.Position = [100,100,800,400];

% % looking at feature matching and 2d image extraction from 3d volume
extractedSlice = uint8(prepimage(exslice,vr_synch));

figure, plot3(uct_x,uct_y,uct_z,'rs')
hold on
%surf(uct_XX,uct_YY,uct_ZZ,'FaceAlpha',0.1,'LineStyle','none')
h = warp(uct_XX,uct_YY,uct_ZZ,extractedSlice);
h.FaceAlpha = 0.75;
axis equal
grid on
xlim([0,round(uct_sx,-2)])
ylim([0,round(uct_sy,-2)])
zlim([0,round(uct_sz,-2)])
xlabel('x')
ylabel('y')
zlabel('z')
view(31.0897,25.5229)
f = gcf; f.Position = [151.6667 226.3333 566.6667 415.3333];

% % viewing 2d-2d registration quality
fixed = uint8(reghist);
fixed_c = uint8(histo(cyi:cyf,cxi:cxf,:));
fixed_H = Hgs(cyi:cyf,cxi:cxf,:);
fixed_E = Egs(cyi:cyf,cxi:cxf,:);

affine = uint8(reguct);
affineNonrigid = uint8(reguct_nr);
rigid = uint8(exslice_rigid(cyi:cyf,cxi:cxf));
affine_s2 = uint8(prepimage(reguct,vr_synch));
affineNonrigid_s2 = uint8(prepimage(reguct_nr,vr_synch));
rigid_s2 = uint8(prepimage(exslice_rigid(cyi:cyf,cxi:cxf),vr_synch));

synchcontours = [130,185];%[120,170];
figure, subplot(1,3,1)
imagesc(fixed_c), axis equal tight off
hold on
[C,h] = imcontour(rigid,synchcontours,'k-');
%h.LineWidth = 2;
title('rigid')
subplot(1,3,2)
imagesc(fixed_c), axis equal tight off
hold on
[C,h] = imcontour(affine,synchcontours,'k-');
%h.LineWidth = 2;
title('affine')
subplot(1,3,3)
imagesc(fixed_c), axis equal tight off
hold on
[C,h] = imcontour(affineNonrigid,synchcontours,'k-');
%h.LineWidth = 2;
title('affine+nonrigid')
f = gcf; f.Position = [117 151 1158 488.6667];

histocontours = [35,85];
figure, subplot(1,3,1)
imagesc(rigid_s2), axis equal tight off, colormap gray
hold on
[C,h] = imcontour(imgaussfilt(imcomplement(fixed_E),2),histocontours,'r-');
%h.LineWidth = 2;
title('rigid')
subplot(1,3,2)
imagesc(affine_s2), axis equal tight off, colormap gray
hold on
[C,h] = imcontour(imgaussfilt(imcomplement(fixed_E),2),histocontours,'r-');
%h.LineWidth = 2;
title('affine')
subplot(1,3,3)
imagesc(affineNonrigid_s2), axis equal tight off, colormap gray
hold on
[C,h] = imcontour(imgaussfilt(imcomplement(fixed_E),2),histocontours,'r-');
%h.LineWidth = 2;
title('affine+nonrigid')
f = gcf; f.Position = [117 151 1158 488.6667];


% % Check point locations for rigid, affine, and affine+nonrigid
figure, subplot(1,3,1)
imagesc(rigid,vr_synch), axis equal tight off, colormap gray
hold on, plot(uct_x2_rigid-cxi,uct_y2_rigid-cyi,'s','MarkerEdgeColor',bmc_red)
plot(his_x2-cxi,his_y2-cyi,'p','MarkerEdgeColor',bmc_green)
plot([his_x2-cxi,uct_x2_rigid-cxi]',[his_y2-cyi,uct_y2_rigid-cyi]','r-')
title('rigid')

subplot(1,3,2), imagesc(affine,vr_synch), axis equal tight off, colormap gray
hold on, plot(uct_x2_affine-cxi,uct_y2_affine-cyi,'s','MarkerEdgeColor',bmc_red)
plot(his_x2-cxi,his_y2-cyi,'p','MarkerEdgeColor',bmc_green)
plot([his_x2-cxi,uct_x2_affine-cxi]',[his_y2-cyi,uct_y2_affine-cyi]','r-')
title('affine')

subplot(1,3,3), imagesc(affineNonrigid,vr_synch), axis equal tight off, colormap gray
hold on, plot(uct_x2_nonrigid,uct_y2_nonrigid,'s','MarkerEdgeColor',bmc_red)
plot(his_x2-cxi,his_y2-cyi,'p','MarkerEdgeColor',bmc_green)
plot([his_x2-cxi,uct_x2_nonrigid]',[his_y2-cyi,uct_y2_nonrigid]','r-')
title('affine+nonrigid')
f = gcf; f.Position = [117 151 1158 488.6667];

% distance between matched points
dist_rigid = sqrt((uct_x2_rigid-his_x2).^2+(uct_y2_rigid-his_y2).^2);
dist_affine = sqrt((uct_x2_affine-his_x2).^2+(uct_y2_affine-his_y2).^2);
dist_nonrigid = sqrt((uct_x2_nonrigid+cxi-his_x2).^2+(uct_y2_nonrigid+cyi-his_y2).^2);

figure, boxplot([dist_rigid,dist_affine,dist_nonrigid])
ylim([0,40])
xticklabels({'rigid','affine','affine + non-rigid'})
ylabel('Distance Error [pixels]')
set(gca,'YGrid','on')
f = gcf; f.Position = [163 213.6667 321.3333 258.6667];

%% 10. List numerical results
disp('          & mean & std & 25th & median & 75th percentile \\')
dist_tmp = dist_rigid;
disp(['rigid     & ' num2str(mean(dist_tmp),'%.2f') ' & ' num2str(std(dist_tmp),'%.2f')  ...
     ' & ' num2str(prctile(dist_tmp,25),'%.2f') ' & ' num2str(median(dist_tmp),'%.2f') ...
     ' & ' num2str(prctile(dist_tmp,75),'%.2f') ' pxs \\'])
dist_tmp = dist_affine;
disp(['affine    & ' num2str(mean(dist_tmp),'%.2f') ' & ' num2str(std(dist_tmp),'%.2f')  ...
     ' & ' num2str(prctile(dist_tmp,25),'%.2f') ' & ' num2str(median(dist_tmp),'%.2f') ...
     ' & ' num2str(prctile(dist_tmp,75),'%.2f') ' pxs \\'])
dist_tmp = dist_nonrigid;
disp(['non-rigid & ' num2str(mean(dist_tmp),'%.2f') ' & ' num2str(std(dist_tmp),'%.2f')  ...
     ' & ' num2str(prctile(dist_tmp,25),'%.2f') ' & ' num2str(median(dist_tmp),'%.2f') ...
     ' & ' num2str(prctile(dist_tmp,75),'%.2f') ' pxs \\'])

disp('Program is finished')