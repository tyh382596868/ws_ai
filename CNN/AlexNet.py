import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256,384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Flatten(),

            nn.Linear(43264, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )



        

    def forward(self, x):
        x = self.features(x)
        return x

def visualize_relu():
    import torch
    import matplotlib.pyplot as plt

    # Define the ReLU function
    relu = torch.nn.ReLU()

    # Create a range of input values
    x = torch.linspace(-10, 10, 100)

    # Apply the ReLU function
    y = relu(x)

    # Plot the input vs output curve
    plt.figure(figsize=(8, 6))
    plt.plot(x.numpy(), y.numpy(), label='ReLU', linewidth=2)
    plt.title('ReLU Activation Function')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.grid(alpha=0.3)
    plt.legend()
    # Save the plot
    plt.savefig("./relu_activation_function.png", dpi=300, bbox_inches="tight")  # Save as PNG with high resolution

def visualize_softmax():
    import torch
    import matplotlib.pyplot as plt

    # Define the ReLU function
    softmax = torch.nn.Softmax()

    # Create a range of input values
    x = torch.linspace(-10, 10, 100)

    # Apply the ReLU function
    y = softmax(x)

    # Plot the input vs output curve
    plt.figure(figsize=(8, 6))
    plt.plot(x.numpy(), y.numpy(), label='softmax', linewidth=2)
    plt.title('softmax Activation Function')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.grid(alpha=0.3)
    plt.legend()
    # Save the plot
    plt.savefig("./softmax_activation_function.png", dpi=300, bbox_inches="tight")  # Save as PNG with high resolution



from torch.utils.data import DataLoader



if __name__ == '__main__':

    train_dataset = torchvision.datasets.CIFAR100(root="/cpfs01/shared/optimal/tyh/dataset",train=True,transform=transforms.ToTensor(),download=True)
    test_dataset = torchvision.datasets.CIFAR100(root="/cpfs01/shared/optimal/tyh/dataset",train=False,transform=transforms.ToTensor(),download=True)
    # 定义数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,  # 训练数据集
        batch_size=64,  # 每批加载 64 个样本
        shuffle=True,  # 打乱数据顺序
        num_workers=4  # 使用 4 个子线程加载数据
    )


    net = AlexNet().to(device)


    for epoch in range(10):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            net.zero_grad()
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            
            optimizer.step()

            if i % 100 == 99:
                print('Epoch: %d, loss: %.3f' % (epoch + 1, loss.item()))

            
            if i % 1000 == 999:
                torch.save(net.state_dict(), 'AlexNet.pkl')
                print('Model saved')
                print('Testing...')
                net.eval()
                correct = 0
                total = 0
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    if (i + 1) % 100 == 0:
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 100))
                        running_loss = 0.0
                        print('Accuracy of the network on the 10000 test images: %d %%' % (
                            100 * correct / total))

    






    # input = torch.randn(1, 3, 224, 224)
    # output = net(input)
    # print(output.size())
    # visualize_relu()
    # visualize_softmax()