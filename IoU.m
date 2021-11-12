function [intersection union ratio] = IoU(mask1, mask2)
    % Input Arguments:
    % (1)two binary images that respectively show the groundtruth ROI and
    % predicted ROI;
    % (2)Note that the input images should be of the same size
    % Output Results:
    % (1)Pixel quantity of the intersection;
    % (2)Pixel quantity of the union;
    % (3)IoU ratio.
    
    [row col] = size(mask1);
    intersection = 0;
    union = 0;
    
    for i=1:row
        for j=1:col
            val_mask1 = mask1(i,j);
            val_mask2 = mask2(i,j);
            % count intersection
            if val_mask1 == 1 && val_mask2 == 1
                intersection = intersection + 1;
            end
            % count union
            if val_mask1 == 1 || val_mask2 == 1
                union = union + 1;
            end
        end
    end    
    ratio = intersection/union;   
end