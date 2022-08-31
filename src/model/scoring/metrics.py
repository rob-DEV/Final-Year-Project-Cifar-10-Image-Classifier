import numpy as np

from maths.maths import EPSILON


class Metrics:
    DECIMAL_PLACES = 3
    @staticmethod
    def confusion_matrix(y_true, y_pred, labels, normalized=False):
        # Generate class matrix of 10x10
        class_count = labels.shape[0]
        matrix = np.zeros((class_count, class_count), dtype=np.float32)

        # For each pair of actual/pred add 1 to the appropriate matrix position
        for act, pred in zip(y_true, y_pred):
            matrix[act, pred] = matrix[act, pred] + 1

        if normalized:
            matrix = matrix / (np.max(matrix) - np.min(matrix))

        return matrix

    @staticmethod
    def plot_confusion_matrix(confusion_matrix, axes=None, dp=0):
        from matplotlib import pyplot as plt
        if axes is not None:
            axes.matshow(confusion_matrix, cmap='Blues')
            for (x, y), value in np.ndenumerate(confusion_matrix):
                axes.text(y, x, f"{value:.{dp}f}", va="center", ha="center")
        else:
            plt.matshow(confusion_matrix, cmap='Blues')
            for (x, y), value in np.ndenumerate(confusion_matrix):
                plt.text(y, x, f"{value:.{dp}f}", va="center", ha="center")

    @staticmethod
    def accuracy(y_true, y_pred):
        num_correct = np.sum(y_true == y_pred)
        return round(num_correct / y_true.shape[0], Metrics.DECIMAL_PLACES)

    @staticmethod
    def precision(y_true, y_pred, pos_label=1):
        # TP / TP + FP
        classes = np.unique(y_true)

        if classes.shape[0] == 2:
            # Binary
            tp = np.sum((y_true == pos_label) & (y_true == y_pred))
            fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
            precision = tp / (float(tp + fp) + EPSILON)
            precision = round(precision, Metrics.DECIMAL_PLACES)
            return precision
        elif classes.shape[0] > 2:
            # Multiclass
            # Score for each class
            precision = []
            for i in range(classes.shape[0]):
                pos_label = classes[i]
            
                tp = np.sum((y_true == pos_label) & (y_true == y_pred))
                fp = np.sum(y_pred == pos_label)

                class_precision = tp / (float(fp) + EPSILON)
                class_precision = round(class_precision, Metrics.DECIMAL_PLACES)
                precision.append((pos_label, class_precision))

            return precision

    @staticmethod
    def recall(y_true, y_pred, pos_label=1):
        # TP / TP + FN
        classes = np.unique(y_true)

        if classes.shape[0] == 2:
            # Binary
            tp = np.sum((y_true == pos_label) & (y_true == y_pred))
            fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
            recall = tp / float(tp + fn)
            recall = round(recall, Metrics.DECIMAL_PLACES)
            return recall
        elif classes.shape[0] > 2:
            # Multiclass
            # Score for each class
            precision = []
            for i in range(classes.shape[0]):
                pos_label = classes[i]
                tp = np.sum((y_true == pos_label) & (y_true == y_pred))
                fn = np.sum(y_true == pos_label)

                class_precision = tp / (float(fn) + EPSILON)
                class_precision = round(class_precision, Metrics.DECIMAL_PLACES)
                precision.append((pos_label, class_precision))

            return precision

    @staticmethod
    def f1(y_true, y_pred):
        precision = Metrics.precision(y_true, y_pred)
        recall = Metrics.recall(y_true, y_pred)

        classes = np.unique(y_true)

        if classes.shape[0] == 2:
            f1 = (2.0 * precision * recall) / (precision + recall + EPSILON)
            f1 = round(f1, Metrics.DECIMAL_PLACES)
            return f1
        elif classes.shape[0] > 2:
            f1 = []
            for i in range(classes.shape[0]):
                pos_label = classes[i]

                class_precision = precision[i]
                class_recall = recall[i]

                class_f1 = (2.0 * class_precision[1] * class_recall[1]) / (class_precision[1] + class_recall[1] + EPSILON)
                class_f1 = round(class_f1, Metrics.DECIMAL_PLACES)

                f1.append((pos_label, class_f1))

            return f1

    @staticmethod
    def roc_curve(y_true, probabilties, target_label=1, thresholds=None):
        if thresholds == None:
            thresholds = np.arange(0, 1.0, 0.005, dtype=np.float32)

        tpr = np.zeros(thresholds.shape)
        fpr = np.zeros(thresholds.shape)

        for i in range(thresholds.shape[0]):
            # Where class < thres = TP
            # Where other classes < thres = FP
            meets_threshold = np.greater_equal(
                probabilties, thresholds[i]).astype(int)

            true_positive = np.equal(
                meets_threshold, 1) & np.equal(y_true, target_label)
            true_negative = np.equal(
                meets_threshold, 0) & np.not_equal(y_true, target_label)
            false_positive = np.equal(
                meets_threshold, 1) & np.not_equal(y_true, target_label)
            false_negative = np.equal(
                meets_threshold, 0) & np.equal(y_true, target_label)

            _tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum() + EPSILON)
            _fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum() + EPSILON)

            tpr[i] = _tpr
            fpr[i] = _fpr

        return tpr, fpr, thresholds

    @staticmethod
    def precision_recall_curve(y_true, probabilties, target_label=1, thresholds=None):
        if thresholds == None:
            thresholds = np.arange(0, 1.0, 0.001, dtype=np.float32)

        precision = np.zeros(thresholds.shape)
        recall = np.zeros(thresholds.shape)

        for i in range(thresholds.shape[0]):
            # Where class < thres = TP
            # Where other classes < thres = FP
            meets_threshold = np.greater_equal(
                probabilties, thresholds[i]).astype(int)

            true_positive = np.equal(
                meets_threshold, 1) & np.equal(y_true, target_label)
            false_positive = np.equal(
                meets_threshold, 1) & np.not_equal(y_true, target_label)
            false_negative = np.equal(
                meets_threshold, 0) & np.equal(y_true, target_label)

            _precision = true_positive.sum() / (true_positive.sum() + false_positive.sum() + EPSILON)
            _recall = true_positive.sum() / (true_positive.sum() + false_negative.sum() + EPSILON)

            precision[i] = _precision
            recall[i] = _recall

        return precision, recall, thresholds

    @staticmethod
    def area_under_curve(x,y):
        points = len(x)
        area = 0
        for i in range(points - 1):
            area += (x[i] - x[i+1]) * y[i]

        return area
