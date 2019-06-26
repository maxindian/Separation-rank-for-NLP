import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_graph(x_axis, y_axis1, y_axis2, x_label, y_label, title, axis, file_name):

    plt.plot(x_axis, y_axis1, 'r-o', label='baseline')
    plt.plot(x_axis, y_axis2, 'b-o', label='input')

    red_patch = mpatches.Patch(color='red', label='Test accuracy', linestyle='solid', linewidth=0.1)
    blue_patch = mpatches.Patch(color='blue', label='Test OOV accuracy', linestyle='solid', linewidth=0.1)

    # lgd = plt.legend(handles=[red_patch, blue_patch], loc='upper left', bbox_to_anchor=(1.5, 1.0))
    lgd = plt.legend(loc='upper right')
    plt.grid(True)
    # plt.axis(axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    fig = plt.figure(1)
    fig.savefig(file_name, dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()

    plt.close()


def main():
    epochs = [1, 2, 3, 4, 5, 6, 8, 10, 13, 15]
    tests_accuracies = [49.6, 94.8, 95.7, 96.3, 96.3, 96.4, 96.5, 96.4, 96.1, 95.6]
    test_oov_accuracies = [43.6, 73.7, 73.1, 75.1, 73.8, 74.7, 75.3, 73.8, 71.6, 66.0]


    x_label = 'Number of epochs training is done'
    y_label = 'Accuracy (%)'
    title = 'Accuracy vs number of epochs'
    axis = [0, 1, 50, 250]
    file_name = 'underfitting_overfitting_graph.png'

    plot_graph(epochs, tests_accuracies, test_oov_accuracies, x_label, y_label, title, axis, file_name)


main()
