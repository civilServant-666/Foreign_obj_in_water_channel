clear all; close all; clc
filePath = './water_vehicle/test/';
namelist = dir([filePath,'*.jpg']);
nl_length = length(namelist);
res = cell(nl_length,10);
for i = 1:nl_length  
    img_nm = namelist(i).name;
    img_nm_sp = strsplit(img_nm,'.');
    img_nm = char(img_nm_sp(1));    
    res{i,1} = img_nm;   
end

%%
for i = 4:nl_length
    %% Extract ROI and calculate IoU
    img_path = [filePath,res{i,1},'.jpg'];
    slic_label_path = [filePath,res{i,1},'_slic_label.mat'];
    load(slic_label_path);    
    % SLIC segmentation
    img = imread(img_path);
    [labels, numlabels] = slicmex(img,300,20);   % input:img,lab_num,compactness;numlabels is the same as number of superpixels
    mask = slic_genMask(img);    % obtain background mask
    [remMaskLabel,remMaskLabelNum] = slic_remMaskLab(mask,labels,numlabels);    % re-sort the label by labeling all backgound pix as 0
    save([filePath,res{i,1},'_slic_seg.mat'],'remMaskLabel');
    [img_h, img_w, img_c] = size(img); 
    
%     % Show segmentation (and lable_num) on original image
%     figure;
%     imshow(img);    
%     hold on;
%     for i = 1:remMaskLabelNum
%         % Generate a binary img corresponding to current label i    
%         bwslic = logical(zeros(img_h, img_w));
%         bwslic(remMaskLabel == i) = 1;
%         % Extract boundary of label region i, and draw it onto original image
%         [B,L] = bwboundaries(bwslic,'noholes');
%         boundary = B{1};
%         plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 2);  
%         % Obtain the centroid of the label region, and write text onto it
%         [L_reg,num_reg] = bwlabel(bwslic,8);
%         L_reg_property = regionprops(L_reg,'all');
%         x = L_reg_property(1).Centroid(1);
%         y = L_reg_property(1).Centroid(2);
%         text(x,y,int2str(i),'color','r');   % x is coordinate along horizontal axis, y along vertical
%     end
    
    % ROI (regions contains abnormal objects) extraction
    %ROI_predict = logical(zeros(img_h,img_w));
    ROI_gt = logical(zeros(img_h,img_w));
    for j = 1:remMaskLabelNum
%         if slic_label(j,1) == 0
%             ROI_predict(remMaskLabel == j) = 1;
%         end
        if slic_label(j,1) == 0
            ROI_gt(remMaskLabel == j) = 1;
        end        
    end 
    % Save predictive ROI
%     ROI_predict_gray = ROI_predict.*255;
%     alphachannel = double(mask);
%     savepath = [filePath,res{i,1},'_ROI_predict.png'];
%     imwrite(ROI_predict_gray, savepath, 'Alpha', alphachannel);
    % Save groundtruth ROI
    ROI_gt_gray = ROI_gt.*255;
    alphachannel = double(mask);
    savepath = [filePath,res{i,1},'_ROI_gt.png'];
    imwrite(ROI_gt_gray, savepath, 'Alpha', alphachannel);
    % IoU calculation
%     if ROI_gt == 0        
%     else
%         [intesec, union, IoU_value] = IoU(ROI_gt, ROI_predict);
%         res{i,2} = intesec;
%         res{i,3} = union;
%         res{i,4} = IoU_value;
%     end    

    %% Seperate ROI instances
    slic_seg_path = [filePath,res{i,1},'_slic_seg.mat'];
    load(slic_seg_path);
    ROI_gt_gray = imread([filePath,res{i,1},'_ROI_gt.png']);
    ROI_gt = im2bw(ROI_gt_gray,0.5);
    [rows cols] = size(ROI_gt);
    % Label connected ROI
    [L_reg,num_reg] = bwlabel(ROI_gt,8);
    L_reg_property = regionprops(L_reg,'all');
    figure;
    imshow(ROI_gt),hold on;
    for j = 1:num_reg
        x = L_reg_property(j).Centroid(1);
        y = L_reg_property(j).Centroid(2);
        text(x,y,int2str(j),'color','b');
    end
    set(gcf,'Position',[0,0,cols,rows]);
    savepath = [filePath,res{i,1},'_ROI_gt_labeled.png'];
    saveas(gca, savepath);    
    close;
    % Generate connected ROI construction matrix
    L_mat = [];
    for j = 1:num_reg
        for k  = 1:max(max(remMaskLabel))
            L_reg_j = L_reg == j;
            remMaskLabel_k = remMaskLabel == k;
            intersec = L_reg_j.*remMaskLabel_k;
            if intersec == remMaskLabel_k
                L_mat(j,k) = 1;
            else
                L_mat(j,k) = 0;
            end
        end
    end
    % Generate ROI instances construction matrix
    I_mat = zeros(1,num_reg);    % Instances construction matrix
    adj_mat = zeros(num_reg,num_reg);   % Adjacent matrix between all connected ROI    
    for j = 1:num_reg
        for k = (j):num_reg
            center_j = L_reg_property(j).Centroid;
            center_k = L_reg_property(k).Centroid;
            distance_j_k = pdist2(center_j,center_k);
            dist_th = 2*(L_reg_property(j).EquivDiameter + L_reg_property(k).EquivDiameter)/2;
            if distance_j_k <= dist_th
                adj_mat(j,k) = 1;
                adj_mat(k,j) = 1;
            end            
        end
    end
    I_mat(1,1) = 1;
    for j = 2:num_reg
        preInsLabel = [];
        for k = 1:j-1            
            if find(adj_mat(k,:) == 1 & adj_mat(j,:) == 1)
                InsLabel_k = I_mat(1,k);
                if find(preInsLabel == InsLabel_k)
                else
                    preInsLabel(end+1) = InsLabel_k;
                end 
            end
        end
        if isempty(preInsLabel)
            I_mat(1,j) = max(I_mat)+1;
        else
            I_mat(1,j) = min(preInsLabel);
            for m = 1:length(preInsLabel)
                I_mat(I_mat == preInsLabel(m)) = min(preInsLabel);
            end
        end
    end                
    % Save connected ROI, ROI instances construction matrix, and adjacent matrix 
    save([filePath,res{i,1},'_instances_constr.mat'],'L_mat','adj_mat','I_mat');
    % Generate figure of ROI instances with bounding box around them
    ROI_instances = L_reg;
    for j = 1:max(I_mat)
        index_j = find(I_mat == j);
        for k = 1:length(index_j)
            ROI_instances(L_reg == index_j(k)) = j;
        end
    end    
    ROI_instances_property = regionprops(ROI_instances,'all');
    figure;
    imshow(ROI_instances);hold on;
    for j = 1:length(ROI_instances_property)
        ROI_instances_j_bd = ROI_instances_property(j).BoundingBox;
        ROI_instances_j_center = ROI_instances_property(j).Centroid;
        text(ROI_instances_j_center(1),ROI_instances_j_center(2),int2str(j),'color','g');
        rectangle('Position',ROI_instances_j_bd,'LineWidth',2,'EdgeColor','g');
    end
    set(gcf,'Position',[0,0,cols,rows]);
    savepath = [filePath,res{i,1},'_ROI_instances.png'];
    saveas(gca, savepath);    
    close;
    
end
    
    