import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(train_loss,valid_loss):
    A = []
    # file = 'TrainingLoss.txt'
    # with open(file, 'r') as f:
    #     lines = f.read()
    # lines = [line for line in lines.split("\n")]
    # for i in range(len(lines) - 3):
    #     A.append(float(lines[i + 2]))

    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Trainning loss")
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label="validation loss")

    # find position of lowest validation loss
    minposs = train_loss.index(min(train_loss)) + 1
    print("stop checkpoint: ", minposs)
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint1')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 1)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')


