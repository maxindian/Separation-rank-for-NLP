import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_graph(x_axis, y_axis1, y_axis2, y_axis3, x_label, y_label, title, axis, file_name):

    plt.plot(x_axis, y_axis1, 'r-o', label='baseline')
    plt.plot(x_axis, y_axis2, 'b-o', label='input')
    plt.plot(x_axis, y_axis3, 'g-o', label='output')

    red_patch = mpatches.Patch(color='red', label='Baseline', linestyle='solid', linewidth=0.1)
    blue_patch = mpatches.Patch(color='blue', label='Input', linestyle='solid', linewidth=0.1)
    green_patch = mpatches.Patch(color='green', label='Output', linestyle='solid', linewidth=0.1)

    lgd = plt.legend(handles=[red_patch, blue_patch, green_patch], loc='upper right', bbox_to_anchor=(1.4, 1.0))

    plt.grid(True)
    # plt.axis(axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    fig = plt.figure(1)
    fig.savefig(file_name, dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.close()


def main():
    batch_sizes = [50, 100, 150, 200, 250, 300]
    input_times = [218, 491, 765, 1038, 1312, 1585, 1859, 2132, 2405, 2679, 3226, 3500, 3773, 4046, 4320, 4594, 4868, 5141, 5415, 5689, 5962, 6236, 6510, 6783, 7057, 7330, 7604, 7878, 8151, 8424, 8698, 8972, 9245, 9519, 9793, 10066]
    output_times = [154, 348, 541, 737, 1124, 1124, 1318, 1511, 1704, 1898, 2092, 2286, 2479, 2673, 2867, 3060, 3254, 3447, 3640, 3834, 4027, 4221, 4414, 4608, 4802, 5188, 5382, 5575, 5769, 5963, 6156, 6350, 6543, 6737, 6930, 7124]
    baseline_times = [151, 342, 533, 724, 915, 1106, 1296, 1487, 1678, 1868, 2059, 2250, 2441, 2631, 2822, 3013, 3204, 3395, 3585, 3776, 3966, 4157, 4348, 4539, 4729, 4920, 5111, 5301, 5492, 5683, 5874, 6065, 6255, 6446, 6637, 6827]


    x_label = 'Number of batches trained'
    y_label = 'Time taken (sec)'
    title = 'Time taken for training in different models'
    axis = [0, 1, 50, 250]
    file_name = 'training_time_plog.png'

    plot_graph(batch_sizes, baseline_times, input_times, output_times, x_label, y_label, title, axis, file_name)


main()
