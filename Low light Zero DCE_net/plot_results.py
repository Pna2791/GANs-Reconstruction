import matplotlib.pyplot as plt


def plot_loss(history, items):
    for i, item in enumerate(items):
        plt.subplot(2, 3, i + 1)
        plt.axis("off")
        plt.title("Train and Validation {} Over Epochs".format(item))
        plt.plot(history.history[item], label=item)
        plt.plot(history.history["val_" + item], label="val_" + item)
        plt.xlabel("Epochs")
        plt.ylabel(item)
        plt.legend()
        plt.grid()
    plt.show()


def plot_results(images, titles, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()

