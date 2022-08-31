from dataset.cifar_dataset_utils import CifarDatasetUtils


class UiUtils:
    def remove_all_widgets_from_layout(layout):
        for cnt in reversed(range(layout.count())):
            widget = layout.takeAt(cnt).widget()
            layout.removeWidget(widget)
            widget.setParent(None)

    def plot_history_metric(history, metric, figure, title="", x_label="", y_label="", show_class_legend=True):
        axes = figure.add_subplot()

        losses = history[metric]
        axes.plot(losses)
        axes.set_title(title)

    def plot_history_metric_multi_classifier(history, metric, figure, title="", x_label="", y_label="", show_class_legend=True):
        axes = figure.add_subplot()
        for i in range(len(history)):
            hist = history[i]

            losses = hist[metric]
            axes.plot(losses, label="{}".format(CifarDatasetUtils.classifications(i)))
        axes.set_title(title)

        if show_class_legend:
            axes.legend()
