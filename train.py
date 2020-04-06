import time
import torch
from dataloader import Image_Loader
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
from Res_net import resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


# Define hyper-parameter
learning_rate = 0.00001
num_epochs = 40
batch_size = 64
valid_step = 5 # after every 20 iterations, evaluate model once time
plot_step = 10 # after every 20 iterations, save and plot loss of training and testing 

img_size = [192, 192]


# Load data for training and testing
train_data = Image_Loader(root_path='./data_train.csv', image_size=img_size, transforms_data=True)
test_data = Image_Loader(root_path='./data_test.csv', image_size=img_size, transforms_data=True)
total_train_data = len(train_data)
total_test_data = len(test_data)
print('Number of training images: ', total_train_data)
print('Number of testing images: ', total_test_data)

# Generate the batch in each iteration for training and testing
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

#define model and port to model to gpu if you have gpu
model = resnet18(pretrained=True)
model = model.to(device)
model.train()

#define the loss function
criterion = nn.MSELoss()

#using SGD optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#TRAINING
iters = 0
everage_training_loss=[]
everage_testing_loss=[]

print('=======> Start Training:')
for epoch in range(num_epochs):
    training_loss = 0.0
    for index, data in enumerate(train_loader):
        iters = iters + 1
        image, label = data
        image = image.to(device)
        label = label.to(device)
        label = label.unsqueeze(1)

        y_pred = model(image)
        # print(type(y_pred), type(label))
        # print(y_pred.size(), label.size())
        loss = criterion(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss = training_loss + float(loss.item())

        # evaluate model after every 20 iteration
        if (iters % valid_step) ==0:
            test_loss = 0.0
            correct_pred = 0
            model.eval()
            test_iter = 0
            with torch.no_grad():
                for _, data in enumerate(test_loader):
                    test_iter = test_iter + 1
                    image, label = data
                    image = image.to(device)
                    label = label.to(device)
                    label = label.unsqueeze(1)
                    y_pred = model(image)
                    
                    # _, pred = torch.max(y_pred, 1)
                    # correct_pred += (pred == label).sum()
                    # correct += pred.eq(target.view_as(pred)).sum().item()

                    
                    loss = criterion(y_pred, label)
                    test_loss += float(loss.item())

            print('Iteration: {}, Training loss: {:.4f}, Test loss: {:.4f}'.format(iters, training_loss/iters, test_loss / test_iter))
            model.train()

        if (iters % plot_step) == 0:
            everage_training_loss.append(training_loss/iters)
            everage_testing_loss.append(test_loss / test_iter)

            plt.figure(1)
            plt.plot(everage_training_loss, color = 'r') # plotting training loss
            plt.plot(everage_testing_loss, color = 'b') # plotting evaluation loss

            plt.legend(['training loss', 'testing loss'], loc='upper left')
            plt.savefig('plot_loss.png')
            # training_/loss = 0

    torch.save(model.state_dict(), './saved_models/saved_model_epoch_{}.pth'.format(epoch))

print('Finished training!')
