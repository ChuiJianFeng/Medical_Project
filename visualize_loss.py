import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    A = []
    file = 'TrainingLoss.txt'
    with open(file, 'r') as f:
        lines = f.read()
    lines = [line for line in lines.split("\n")]
    for i in range(len(lines)-3):
        A.append(float(lines[i + 2]))


    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(A)+1),A,label="Trainning loss")

   # find position of lowest validation loss
    minposs = A.index(min(A))+1
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5) # consistent scale
    plt.xlim(0, len(A)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()