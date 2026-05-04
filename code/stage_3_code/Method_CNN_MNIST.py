'''
CNN Architecture for ORL Dataset
'''

import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Method_CNN_MNIST(nn.Module):
    # it defines the max rounds to train the model
    max_epoch = 2
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    trainloader = None
    testloader = None

    method_name = None
    method_description = None

    def __init__(self, nName, mDescription, loaded_data):
        nn.Module.__init__(self)

        self.method_name = nName
        self.method_description = mDescription
        self.trainloader, self.testloader = loaded_data

        self.conv1 = nn.Conv2d(1, 6, 5, padding='valid')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def fit(self, trainloader):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose

        loss_history = []

        for epoch in range(self.max_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            total_samples = 0  # track samples
            total_batches = 0  # track batches

            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                total_samples += len(labels)  # count images this batch
                total_batches += 1  # count batches

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item()) # Gets current loss value from gradient descent

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

                # Print summary after every epoch
                print(f'[Epoch {epoch + 1}] '
                      f'batches: {total_batches}, '
                      f'samples seen: {total_samples}, '
                      f'loss: {running_loss / total_batches:.3f}')

        # Training Loss Plot
        plt.figure()
        plt.plot(loss_history)
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("../../result/stage_3_result/mnist_training_loss_curve.png")

    def test(self, testloader):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    def run(self):
        print('method running...')
        print('--start training...')
        self.fit(self.trainloader)
        print('--start testing...')
        self.test(self.testloader)
        return {'Training / Testing Finished'}
