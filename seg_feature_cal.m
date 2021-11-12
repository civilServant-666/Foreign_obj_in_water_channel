clear all; close all; clc

filePath = './water_vehicle/test/';
namelist = dir([filePath,'*.jpg']);
nl_length = length(namelist);
res = cell((nl_length-2)*200,80);
hog_mat = [];
sort_fn = cell(nl_length-2,1);
for i = 1:nl_length
    img_nm = namelist(i).name;
    img_nm_sp = strsplit(img_nm,'_');
    img_nm_sp = strsplit(char(img_nm_sp(2)),'.');
    img_seq = str2num(char(img_nm_sp(1)));
    sort_fn{img_seq,1} = img_nm;    
end   

%% Main loop
count = 1;
for id = 41:41 %(nl_length-2)
    id
    %% SLIC superpixel segmentation
    tic;
    path = ['./water_vehicle/test/',sort_fn{id,1}];
    img = imread(path);
    [labels, numlabels] = slicmex(img,300,20);   % input:img,lab_num,compactness;numlabels is the same as number of superpixels
    mask = slic_genMask(img);    % obtain background mask
    [remMaskLabel,remMaskLabelNum] = slic_remMaskLab(mask,labels,numlabels);    % re-sort the label by labeling all backgound pix as 0
    [height, width, channel] = size(img);
    t_slic = toc;
    % figure;imagesc(labels);
    % figure;imagesc(remMaskLabel);    

    % Show segmentation (and lable_num) on original image
    figure;
    imshow(img);
    title(sort_fn{id,1});
    hold on;
    for i = 1:remMaskLabelNum
        % Generate a binary img corresponding to current label i    
        bwslic = logical(zeros(height, width));
        bwslic(remMaskLabel == i) = 1;
        % Extract boundary of label region i, and draw it onto original image
        [B,L] = bwboundaries(bwslic,'noholes');
        boundary = B{1};
        plot(boundary(:,2), boundary(:,1), 'c', 'LineWidth', 2);  
        % Obtain the centroid of the label region, and write text onto it
        [L_reg,num_reg] = bwlabel(bwslic,8);
        L_reg_property = regionprops(L_reg,'all');
        x = L_reg_property(1).Centroid(1);
        y = L_reg_property(1).Centroid(2);
        text(x,y,int2str(i),'color','y');   % x is coordinate along horizontal axis, y along vertical
    end 

    %% Calculate LBP for each SLIC region
    adjMat = slic_adjSuPix(remMaskLabel,remMaskLabelNum);
    for i = 1:remMaskLabelNum
        tic;
        i
        fn_no_format_arr = strsplit(sort_fn{id,1},'.');
        fn_no_format = char(fn_no_format_arr(1));
        res{count,1} = [fn_no_format,'_',num2str(i)]; 
        % Determine the bounding box of label i
        bd_box = [];
        bw_index = remMaskLabel == i;
        if adjMat(1,i+1) == 1   % Region i is at the channel boundary
            res{count,2} = 1; 
            [bdbox_up, bdbox_down, bdbox_left, bdbox_right] = bd_box_boundary(bw_index);
        else                    % Region i is not adjcent to channel boundary
            res{count,2} = 0;
            [bdbox_up, bdbox_down, bdbox_left, bdbox_right] = bd_box_non(bw_index);
        end
        % Calculate LBP of current bounding box
        bd_box = img (bdbox_up:bdbox_down,bdbox_left:bdbox_right,:);
        %figure;subplot(1,2,1),imshow(bd_box);
        mapping = seg_getmapping(8,'ri');    
        hist_lbp_raw = seg_lbp(bd_box,1,8,mapping,'nh');    % histogram of lbp of the original
        bd_box_eq = histeq(rgb2gray(bd_box));
        %subplot(1,2,2),imshow(bd_box_eq);
        hist_lbp_eq = seg_lbp(bd_box_eq,1,8,mapping,'nh');  % histogram of lbp after equalize the image enhance contrast
        t_lbp = toc;
        res{count,3} = t_slic;
        res{count,4} = t_lbp;
        for j = 5:76
            if j <= 40
                res{count,j} = hist_lbp_raw(j-4);
            else
                res{count,j} = hist_lbp_eq(j-40);
            end
        end
        count = count + 1;
    end 

  

    %% Wait until current figure gui is closed...
    uiresume;
    uiremind = errordlg('Please close the figure window to move forward.');
    set(uiremind, 'WindowStyle', 'modal');   
    uiwait;     
end

%% Function -- determine bounding box of a boundary SLIC region
function [up, down, left, right] = bd_box_boundary(bw_index)
    [int_up, int_down, int_left, int_right] = bd_box_non(bw_index);
    wth = int_right-int_left;
    hgt = int_down-int_up;
    cycles = min([wth hgt]);
    cycles = floor(cycles/3);
    for i = 1:cycles
        up = int_up + i;
        down = int_down - i;
        left = int_left + i;
        right = int_right - i;
        isbelong_up_left = bw_index(up,left) == 1;
        isbelong_up_right = bw_index(up,right) == 1;
        isbelong_down_left = bw_index(down,left) == 1;
        isbelong_down_right = bw_index(down,right) == 1;
        sum_isbelong = isbelong_up_left + isbelong_up_right +isbelong_down_left + isbelong_down_right;
        if sum_isbelong >= 4
            break;
        end
    end     
end    

%% Function -- determine bounding box of a non-boundary SLIC region
function [up, down, left, right] = bd_box_non(bw_index)
    bw_index_bd_box = regionprops(bw_index,'boundingbox');
    up = bw_index_bd_box.BoundingBox(2);    
    left = bw_index_bd_box.BoundingBox(1);
    down = up + bw_index_bd_box.BoundingBox(4);
    right = left + bw_index_bd_box.BoundingBox(3);    
    up = floor(up)+1;
    left = floor(left)+1;
    down = floor(down)-1;
    right = floor(right)-1;
end