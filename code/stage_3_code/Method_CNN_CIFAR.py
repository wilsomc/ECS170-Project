'''
CNN Architecture for CIFAR-10 Dataset
CIFAR images: 3 x 32 x 32, 10 classes
'''

import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Method_CNN_CIFAR(nn.Module):
    # it defines the max rounds to train the model
    max_epoch = 20
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

        # CIFAR images: 3 x 32 x 32
        # Block 1: conv(3->32, 3, pad=1) -> BN -> ReLU -> MaxPool(2) => 32 x 16 x 16
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Block 2: conv(32->64, 3, pad=1) -> BN -> ReLU -> MaxPool(2) => 64 x 8 x 8
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Block 3: conv(64->128, 3, pad=1) -> BN -> ReLU -> MaxPool(2) => 128 x 4 x 4
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout_fc = nn.Dropout(0.5)

        # 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def fit(self, trainloader):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        loss_history = []

        for epoch in range(self.max_epoch):
            running_loss = 0.0
            total_samples = 0
            total_batches = 0

            self.train()  # set to training mode (enables dropout/BN)

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
        plt.title("CIFAR-10 Training Loss Curve")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.savefig("result/stage_3_result/cifar_training_loss_curve.png")
        print("Saved learning curve to result/stage_3_result/cifar_training_loss_curve.png")

    def test(self, testloader):
        correct = 0
        total = 0
        self.eval()  # set to eval mode (disables dropout/BN)
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
