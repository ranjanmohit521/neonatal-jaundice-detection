function predict_image(imagePath, modelFile)

if nargin < 2
    modelFile = 'jaundice_model.mat';
end

if ~isfile(modelFile)
    error('Model not found. Run train_model.m first.');
end

loaded = load(modelFile);
fields = fieldnames(loaded);
net = loaded.(fields{1});

img = imread(imagePath);
img = imresize(img, [224 224]);

[label, scores] = classify(net, img);
confidence = max(scores) * 100;

fprintf('\nImage: %s\n', imagePath);
fprintf('Prediction: %s\n', string(label));
fprintf('Confidence: %.2f%%\n', confidence);

classNames = net.Layers(end).Classes;
fprintf('\nClass Probabilities:\n');
for i = 1:numel(classNames)
    fprintf('  %-15s %.2f%%\n', string(classNames(i)), scores(i) * 100);
end

figure('Name', 'Prediction Result');
imshow(img);
title(sprintf('%s (%.1f%%)', string(label), confidence), 'FontSize', 14);

end
