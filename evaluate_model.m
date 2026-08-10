function evaluate_model(net, augVal, trueLabels)

predLabels = classify(net, augVal);
scores = predict(net, augVal);

accuracy = sum(predLabels == trueLabels) / numel(trueLabels);
fprintf('\nAccuracy: %.2f%%\n', accuracy * 100);

classNames = categories(trueLabels);
cm = confusionmat(trueLabels, predLabels, 'Order', classNames);

figure('Name', 'Confusion Matrix');
confusionchart(trueLabels, predLabels);
title('Confusion Matrix');

numClassesEval = numel(classNames);
precision = zeros(numClassesEval, 1);
recall = zeros(numClassesEval, 1);
f1 = zeros(numClassesEval, 1);

for i = 1:numClassesEval
    tp = cm(i, i);
    fp = sum(cm(:, i)) - tp;
    fn = sum(cm(i, :)) - tp;

    if (tp + fp) > 0
        precision(i) = tp / (tp + fp);
    end
    if (tp + fn) > 0
        recall(i) = tp / (tp + fn);
    end
    if (precision(i) + recall(i)) > 0
        f1(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i));
    end
end

fprintf('\n%-15s %-10s %-10s %-10s\n', 'Class', 'Precision', 'Recall', 'F1-Score');
fprintf('%-15s %-10s %-10s %-10s\n', '-----', '---------', '------', '--------');
for i = 1:numClassesEval
    fprintf('%-15s %-10.4f %-10.4f %-10.4f\n', string(classNames(i)), precision(i), recall(i), f1(i));
end

jaundiceIdx = find(strcmp(classNames, 'Jaundice'));
if ~isempty(jaundiceIdx)
    jaundiceScores = scores(:, jaundiceIdx);
    jaundiceTrue = double(trueLabels == 'Jaundice');

    [X, Y, ~, auc] = perfcurve(jaundiceTrue, jaundiceScores, 1);

    figure('Name', 'ROC Curve');
    plot(X, Y, 'b-', 'LineWidth', 2);
    hold on;
    plot([0 1], [0 1], 'r--');
    hold off;
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(sprintf('ROC Curve (AUC = %.4f)', auc));
    legend('Model', 'Random', 'Location', 'southeast');
    grid on;

    fprintf('\nROC-AUC: %.4f\n', auc);
end

end
