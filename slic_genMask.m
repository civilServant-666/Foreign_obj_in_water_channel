function mask = slic_genMask(img)
    gray = rgb2gray(img);
    level = 0.99;
    bw = im2bw(gray,level);
    bw = imfill(bw,'holes');
    bwc = imcomplement(bw);
    mask = imfill(bwc,'holes');
       
end