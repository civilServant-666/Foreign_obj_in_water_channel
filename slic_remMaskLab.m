function [remMaskLabel,remMaskLabelNum] = slic_remMaskLab(mask,labels,numlabels)
    labels = labels + 1;
    count = 0;
    for i = 1:numlabels
        isBack = mean(mask(labels == i));   % use a number to measure the likelihood of background, the closer to zero, the more likely being background
        if isBack < 0.25
            labels (labels == i) = 0;
            count = count + 1;
        else
            labels (labels == i) = i-count;
        end     
    end
    remMaskLabel = labels;
    remMaskLabelNum = numlabels-count;
end