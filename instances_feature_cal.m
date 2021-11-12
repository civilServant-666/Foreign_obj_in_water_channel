clear all; close all; clc

filePath = './water_vehicle/test/';
namelist = dir([filePath,'*.jpg']);
nl_length = length(namelist);
res = cell((nl_length)*50,80);
sort_fn = cell(nl_length,1);
for i = 1:nl_length
    img_nm = namelist(i).name;
    img_nm_sp = strsplit(img_nm,'.');    
    img_nm = char(img_nm_sp(1));
    sort_fn{i,1} = img_nm;    
end   

%% Main loop
count = 1;
for i = 4:length(sort_fn)
    % Load data
    img = imread([filePath,sort_fn{i,1},'.jpg']);
    slic_seg_path = [filePath,sort_fn{i,1},'_slic_seg.mat'];
    load(slic_seg_path);
    [rows cols] = size(remMaskLabel);
    instance_constr_path = [filePath,sort_fn{i,1},'_instances_constr.mat'];
    load(instance_constr_path);
    
    for j = 1:max(I_mat)
        slic_of_instance_j = [];
        roi_of_instance_j = find(I_mat == j);
        for k = 1:length(roi_of_instance_j)
            if ~(isempty(L_mat))            
                slic_of_roi_k = find(L_mat(roi_of_instance_j(k),:) == 1);
                slic_of_instance_j = [slic_of_instance_j,slic_of_roi_k];
            end
        end         
        if length(slic_of_instance_j) <= 8
            combo_up_limit = min([3,length(slic_of_instance_j)]);
            for k = 1:combo_up_limit
                combo_slic_of_instance_j = combntns(slic_of_instance_j,k);
                [combo_rows,combo_cols] = size(combo_slic_of_instance_j);
                for m = 1:combo_rows
                    lbp_area_id = [sort_fn{i,1},'_',num2str(j)];
                    bw_index = logical(zeros(rows,cols));
                    for n = 1:k
                        bw_index(remMaskLabel == combo_slic_of_instance_j(m,n)) = 1;
                        lbp_area_id = [lbp_area_id,'_',num2str(combo_slic_of_instance_j(m,n))];
                    end
                    res{count,1} = lbp_area_id;
                    [bdbox_up, bdbox_down, bdbox_left, bdbox_right] = bd_box_non(bw_index);
                    bd_box = img (bdbox_up:bdbox_down,bdbox_left:bdbox_right,:);
                    mapping = seg_getmapping(8,'ri');    
                    hist_lbp_raw = seg_lbp(bd_box,1,8,mapping,'nh');
                    for n = 1:36
                        res{count,n+1} = hist_lbp_raw(n);
                    end
                    count = count+1;
                end
            end
        else            
            for k = 1:length(slic_of_instance_j)
                lbp_area_id = [sort_fn{i,1},'_',num2str(j)];
                lbp_area_id = [lbp_area_id,'_',num2str(slic_of_instance_j(k))];
                res{count,1} = lbp_area_id;
                bw_index = logical(zeros(rows,cols));
                bw_index(remMaskLabel == slic_of_instance_j(k)) = 1;
                [bdbox_up, bdbox_down, bdbox_left, bdbox_right] = bd_box_non(bw_index);
                bd_box = img (bdbox_up:bdbox_down,bdbox_left:bdbox_right,:);
                mapping = seg_getmapping(8,'ri');    
                hist_lbp_raw = seg_lbp(bd_box,1,8,mapping,'nh');
                for n = 1:36
                    res{count,n+1} = hist_lbp_raw(n);
                end
                count = count+1;
            end
        end
    end 
        
end

%% Function -- determine bounding box of a non-boundary SLIC region
function [up, down, left, right] = bd_box_non(bw_index)
    bw_index_bd_box = regionprops(double(bw_index),'boundingbox');
    up = bw_index_bd_box.BoundingBox(2);    
    left = bw_index_bd_box.BoundingBox(1);
    down = up + bw_index_bd_box.BoundingBox(4);
    right = left + bw_index_bd_box.BoundingBox(3);    
    up = floor(up)+1;
    left = floor(left)+1;
    down = floor(down)-1;
    right = floor(right)-1;
end