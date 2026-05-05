'''
CNN Architecture for MNIST Dataset
'''

import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Method_CNN_MNIST(nn.Module):
    # it defines the max rounds to train the model
    max_epoch = 10
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

        # MNIST images: 1 x 28 x 28
        # After conv1(1->6, kernel=5, valid): 6 x 24 x 24
        # After pool(2,2): 6 x 12 x 12
        # After conv2(6->16, kernel=5, valid): 16 x 8 x 8
        # After pool(2,2): 16 x 4 x 4 = 256
        self.conv1 = nn.Conv2d(1, 6, 5, padding='valid')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
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

        loss_history = []

        for epoch in range(self.max_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            total_samples = 0
            total_batches = 0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                total_samples += len(labels)
                total_batches += 1

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item())

                running_loss += loss.item()
                if i % 200 == 199:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                    running_loss = 0.0

            # Print summary after every epoch
            avg_loss = sum(loss_history[-total_batches:]) / total_batches
            print(f'[Epoch {epoch + 1}] '
                  f'batches: {total_batches}, '
                  f'samples seen: {total_samples}, '
                  f'avg loss: {avg_loss:.3f}')

        # Training Loss Plot
        plt.figure()
        plt.plot(loss_history)
        plt.title("MNIST Training Loss Curve")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.savefig("result/stage_3_result/mnist_training_loss_curve.png")
        print("Saved learning curve to result/stage_3_result/mnist_training_loss_curve.png")

    def test(self, testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the {total} test images: {100 * correct / total:.2f} %')

    def run(self):
        print('method running...')
        print('--start training...')
        self.fit(self.trainloader)
        print('--start testing...')
        self.test(self.testloader)
        return {'Training / Testing Finished'}
