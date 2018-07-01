import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils as utils
import torchvision
import torch
import torch.optim as optim
import torchvision.models as models
import time
import argparse
from pruning_package.model_compressor import PruneSimilarFilter

data_root = "data"


class VGG4CIFAR10(nn.Module):
    def __init__(self, model):
        super(VGG4CIFAR10, self).__init__()
        self.features = model.features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



transforms = transforms.Compose([transforms.Resize(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),])

dataset = torchvision.datasets.CIFAR10(root=data_root, transform=transforms, download=True, train=True)
train_data = utils.data.DataLoader(dataset, shuffle=True, batch_size=32, num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(root=data_root, transform=transforms, download=True, train=False)
test_data = utils.data.DataLoader(test_dataset, shuffle=False, batch_size=32, num_workers=2)


def test(model, test_data):
    correct = 0
    total = 0
    for (images, labels) in test_data:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (accuracy))
    print('Testing is Done!')
    return accuracy


def train(net):
    start = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(30):
        print("Epoch: {}".format(epoch + 1))
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_data, 0):
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    print('Training is Done!')
    end = time.time()
    print("Training time is: {}".format(end - start))
    test_start_time = time.time()
    test(net, test_data)
    test_end_time = time.time()
    print("Testing time is: {}".format(test_end_time - test_start_time))

    torch.save(net.state_dict(), './model.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train and prune a vgg 16 model for cifar10 dataset")
    parser.add_argument("--train", help="train model and save", default=True, type=bool, required=True)
    parser.add_argument("--prune", help="prune model and save", default=False, type=bool, required=False)

    args = parser.parse_args()
    model = VGG4CIFAR10(models.vgg16(pretrained=True))

    if args.train:
        print("################# Training is started #################")
        train(model)

    if args.prune:
        print("################# Pruning is started #################")
        model.load_state_dict(torch.load('./model.pth'))
        test(model, test_data)
        pruneSimilarFilter = PruneSimilarFilter(model=model, test_data=test_data, train_data=train_data)
        pruneSimilarFilter.compressor(similarity_threshold=0.98, final_epochs=30, pruned_model_path="pruned_model.pth")
