import torch
import torch.nn as nn
from torchinfo import summary
from Model_VGG16 import VGG16
from dataset import IMAGE_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
import time
import os

##REPRODUCIBILITY
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#args = parse_args()
#CUDA_DEVICES = args.cuda_devices
#DATASET_ROOT = args.path
CUDA_DEVICES = 7
DATASET_ROOT = './train'

# Initial learning rate
init_lr = 0.01

# Save model every 5 epochs
checkpoint_interval = 5
if not os.path.isdir('Checkpoint/'):
    os.mkdir('Checkpoint/')

# Setting learning rate operation
def adjust_lr(optimizer, epoch):

    # 1/10 learning rate every 5 epochs
    lr = init_lr * (0.1 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    #print(DATASET_ROOT)
    train_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)

    # If out of memory , adjusting the batch size smaller
    data_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True, num_workers=1)

    #print(train_set.num_classes)
    model = VGG16(num_classes=train_set.num_classes)
    model = model.cuda(CUDA_DEVICES)
    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Training epochs
    num_epochs = 20
    criterion = nn.CrossEntropyLoss()

    # Optimizer setting
    optimizer = torch.optim.SGD(params=model.parameters(), lr=init_lr, momentum=0.9)


    # Log
    with open('TrainingAccuracy.txt','w') as fAcc:
        print('Accuracy\n', file = fAcc)
    with open('TrainingLoss.txt','w') as fLoss:
        print('Loss\n', file = fLoss)

    for epoch in range(num_epochs):
        localtime = time.asctime( time.localtime(time.time()) )
        print('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,num_epochs,localtime))
        print('-' * len('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,num_epochs,localtime)))

        training_loss = 0.0
        training_corrects = 0
        adjust_lr(optimizer, epoch)

        for i, (inputs, labels) in enumerate(data_loader):
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            # inputs = Variable(inputs)
            # labels = Variable(labels)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += float(loss.item() * inputs.size(0))
            training_corrects += torch.sum(preds == labels.data)

        training_loss = training_loss / len(train_set)
        training_acc = training_corrects.double() /len(train_set)
        print('Training loss: {:.4f}\taccuracy: {:.4f}\n'.format(training_loss,training_acc))


        # Check best accuracy model ( but not the best on test )
        if training_acc > best_acc:
            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())


        with open('TrainingAccuracy.txt','a') as fAcc:
            print('{:.4f} '.format(training_acc), file = fAcc)
        with open('TrainingLoss.txt','a') as fLoss:
            print('{:.4f} '.format(training_loss), file = fLoss)

        # Checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(model, 'Checkpoint/model-epoch-{:d}-train.pth'.format(epoch + 1))

    # Save best training/valid accuracy model ( not the best on test )
    model.load_state_dict(best_model_params)
    best_model_name = 'model-{:.2f}-best_train_acc.pth'.format(best_acc)
    torch.save(model, best_model_name)
    # with open("model_name.txt", "a") as txtfile:
    #     print("{}".format(best_model_name), file=txtfile)
    with open("info.txt", "a") as txtfile2:
        print("{}".format(summary(model, input_size=(16, 3, 224, 224))), file=txtfile2)

if __name__ == '__main__':
    train()
