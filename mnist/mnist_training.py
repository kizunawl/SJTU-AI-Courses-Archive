import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(in_features=7*7*64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
            nn.Softmax(dim=1)
        )
    
    def forward(self, input):
        output = self.model(input)
        return output


def training():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

    trainData = datasets.MNIST(root='./data/', train=True, transform=transform)
    testData = datasets.MNIST(root='./data/', train=False, transform=transform)

    BATCHSIZE = 256
    EPOCHS = 10

    trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCHSIZE, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCHSIZE)
    net = Net()
    net.load_state_dict(torch.load('./MNIST_model.pth'))

    getLoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    AccuracyLog = []

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainDataLoader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = getLoss(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        if (epoch%2==1):
            total = 0
            correct = 0
            with torch.no_grad():
                for images, labels in testDataLoader:
                    outputs = net(images)
                    _, predictions = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predictions==labels).sum().item()
            AccuracyLog.append(correct/total)
            print('[Epoch{}]:{}%'.format(epoch, 100*correct/total))

    print('Training Done')

    torch.save(net.state_dict(), './MNIST_model.pth')

    plt.plot(AccuracyLog, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('training_accuracy.jpg')
    plt.show()


def test():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

    testData = datasets.MNIST(root='./data/', train=False, transform=transform)

    BATCHSIZE = 256

    testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCHSIZE)

    net = Net()
    net.load_state_dict(torch.load('./MNIST_model.pth'))

    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in testDataLoader:
            outputs = net(images)
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions==labels).sum().item()
    print('Accuracy: {}%'.format(100*correct/total))


if __name__=='__main__':
    print('[train]:1, [test]:2')
    a = int(input())
    if (a==1):
        training()
    else:
        test()