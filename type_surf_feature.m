img_rgb = imread('./test_data/ship2.png');
img_gray = rgb2gray(img_rgb);

points = detectSURFFeatures(img_gray);
[features, vpts] = extractFeatures(img_gray, points);

figure;
imshow(img_gray);hold on;
plot(vpts.selectStrongest(10));