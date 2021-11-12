function adjMat = slic_adjSuPix(labelMat,labelNum)

laM = labelMat;
laN = labelNum;

adjMat = int32(zeros(laN,laN));    % Initialize adjacent matrix
[h w] = size(labelMat);    % Obtain the dimension of labelMat, h--num of rows, w--num of column

for i = 1:(h-1)
    for j = 1:(w-1)
        % Acquire labels of the current pixel and its adjacent pixels 
        pixlab = laM(i,j);
        pixlab_right = laM(i,j+1);
        pixlab_down = laM(i+1,j);
        % compare to the right, if not equal, we got two adjacent
        % superpixels
        if pixlab ~= pixlab_right
            adjMat(pixlab+1, pixlab_right+1) = 1;
            adjMat(pixlab_right+1, pixlab+1) = 1;
        end
        % compare to the lower, if not equal, we got two adjacent
        % superpixels
        if pixlab ~= pixlab_down
            adjMat(pixlab+1, pixlab_down+1) = 1;
            adjMat(pixlab_down+1, pixlab+1) = 1;
        end       
    end
end    

end