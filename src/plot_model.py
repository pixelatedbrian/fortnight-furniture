import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def clean_history(hist):
    '''
    For whatever reason saving and loading weights prepends a 0 to the history
    of the model. (weird)

    This attempts to strip that padding so that the charts appear as they Should

    INPUTS:
    hist: dictionary with the history of a model (acc, val, etc)

    RETURNS:
    dictionary of lists with the padded zeros removed
    '''
    temp_hist = hist.copy()
    chop = sum([1 for item in hist["acc"] if item == 0])

    for key in temp_hist.keys():
        temp_hist[key] = temp_hist[key][chop:]

    return temp_hist


def plot_hist(history, info_str, epochs=2, augmentation=1, sprint=False):
    '''
    Make a plot of the rate of error as well as the accuracy of the model
    during training.  Also include a line at error 0.20 which was the original
    minimum acceptable error (self imposed) to submit results to the test
    set when doing 3-way split.
    Even after performance regularly exceeded the minimum requirement the line
    was unchanged so that all of the graphs would be relative to each other.
    Also it was still useful to see how a model's error was performing relative
    to this baseline.
    Also, the 2 charts written as a png had the filename coded to include
    hyperparameters that were used in the model when the chart was created.
    This allowed a simple visual evaluation of a model's performance when
    doing randomized hyperparameter search. If a model appeared to be high
    performing then the values could be reused in order to attempt to
    replicate the result.
    '''

    # clean the history first in case it was zero padded from keras
    history = clean_history(history)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    fig.suptitle("", fontsize=12, fontweight='normal')

    # stuff for marking the major and minor ticks dynamically relative
    # to the numper of epochs used to train
    major_ticks = int(epochs / 10.0)
    minor_ticks = int(epochs / 20.0)

    title_text = "Homewares and Furniture Image Identification\n Train Set and Dev Set"
    ACC = 0.829   # record accuracy
    if sprint is True:
        ACC = 0.750
        title_text = "SPRINT: Homewares and Furniture Image Identification\n Train Set and Dev Set"

    if major_ticks < 2:
        major_ticks = 2

    if minor_ticks < 1:
        minor_ticks = 1

    majorLocator = MultipleLocator(major_ticks)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator = MultipleLocator(minor_ticks)

    # # determine how many zero layers there are
    # drop = np.sum([1 if loss == 0 else 0 for loss in history['loss']])
    #
    # if drop > 1:
    #     drop -= 1
    #
    # history['loss'] = history['loss'][drop:]
    # history['val_loss'] = history['val_loss'][drop:]
    # history['acc'] = history['acc'][drop:]
    # history['val_acc'] = history['val_acc'][drop:]

    # correct x axis
    history['loss'] = [0.0] + history['loss']
    history['val_loss'] = [0.0] + history['val_loss']
    history['acc'] = [0.0] + history['acc']
    history['val_acc'] = [0.0] + history['val_acc']

    x_line = [ACC] * (epochs + 1)  # this line is now for accuracy of test set

    # stuff for the loss chart
    axs[0].set_title(title_text)

    if augmentation > 1:
        axs[0].set_xlabel('Epochs\nAugmentation of {:3d}'.format(augmentation))
    else:
        axs[0].set_xlabel('Epochs')

    axs[0].set_xlim(1, epochs)
    axs[0].set_ylabel('Loss')
    axs[0].set_ylim(0, 5.0)

    axs[0].plot(history['loss'], color="blue", linestyle="--", alpha=0.8, lw=1.0)
    axs[0].plot(history['val_loss'], color="blue", alpha=0.8, lw=1.0)
    axs[0].legend(['Training', 'Validation'])
    axs[0].xaxis.set_major_locator(majorLocator)
    axs[0].xaxis.set_major_formatter(majorFormatter)

    # for the minor ticks, use no labels; default NullFormatter
    axs[0].xaxis.set_minor_locator(minorLocator)

    # stuff for the accuracy chart
    axs[1].set_title(title_text)

    if augmentation > 1:
        axs[0].set_xlabel('Epochs\nAugmentation of {:3d}'.format(augmentation))
    else:
        axs[0].set_xlabel('Epochs')

    axs[1].set_xlim(1, epochs)
    axs[1].set_ylabel('Accuracy')
    axs[1].set_ylim(0.0, 1.0)
    axs[1].plot(x_line, color="red", alpha=0.3, lw=4.0)
    axs[1].plot(history['acc'], color="blue", linestyle="--", alpha=0.5, lw=1.0)
    axs[1].plot(history['val_acc'], color="blue", alpha=0.8, lw=1.0)
    axs[1].plot(x_line, color="red", linestyle="--", alpha=0.8, lw=1.0)
    axs[1].legend(['Record Accuracy ({:1.2f})'.format(ACC), 'Training', 'Validation'], loc='lower right')
    axs[1].xaxis.set_major_locator(majorLocator)
    axs[1].xaxis.set_major_formatter(majorFormatter)

    # for the minor ticks, use no labels; default NullFormatter
    axs[1].xaxis.set_minor_locator(minorLocator)

    plt.savefig("../imgs/" + info_str, facecolor='w', edgecolor='w', transparent=False)
    # plt.show()
    plt.close('all')
