import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve
import numpy as np

# Enable anomaly detection to find the operations that failed to compute their gradient
torch.autograd.set_detect_anomaly(True)
# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# Convolutional neural network (AlexNet)
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):  # CIFAR-10 has 10 classes
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
class Fire(nn.Module):
    def __init__(self, in_channel, s1x1, e1x1, e3x3):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channel, s1x1, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(s1x1, e1x1, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(s1x1, e3x3, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)
    

    
def trainAlexNet(num_epochs, batch_size, learning_rate, train_loader):

    model = AlexNet(num_classes=10).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)

    # Initialize a list to store the loss values
    loss_values = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Record the loss value
            loss_values.append(loss.item())
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))     

    # Save the trained model after the training loop
    torch.save(model.state_dict(), 'alexnet_cifar10_model.pth')

    return loss_values



def trainSqueezeNet(num_epochs, batch_size, learning_rate, train_loader):
    model = SqueezeNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)

    # Initialize a list to store the loss values
    loss_values = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Record the loss value
            loss_values.append(loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Save the trained model after the training loop
    torch.save(model.state_dict(), 'vgg16_cifar10_model.pth')

    return loss_values



# Hyper-parameters
num_epochs = 10
batch_size = 100
learning_rate = 0.0001

# Data Preprocessing:
# The transforms are applied to the dataset.
# Resize: Adjusts the size of the image to 227x227 pixels for AlexNet input.
# ToTensor: Converts the image to a PyTorch tensor.
# Normalize: Normalizes the image data to have mean and std deviation of 0.5 for each channel.
transform = transforms.Compose([
    transforms.Resize(227),  # Resizing to 227x227 pixels
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the pixel values
])

# Download and load the CIFAR-10 training dataset with the defined transformations
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
# Download and load the CIFAR-10 test dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#loss_values = trainAlexNet(num_epochs, batch_size, learning_rate, train_loader)
loss_values = trainSqueezeNet(num_epochs, batch_size, learning_rate, train_loader)

# After training, plot the loss values
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Training Loss')
plt.title('Loss as a Function of Training Steps')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()

# # Load the model for evaluation
# model = AlexNet(num_classes=10)  # Re-create the model structure
# model.load_state_dict(torch.load('alexnet_cifar10_model.pth'))
# model.to(device)

# # After testing the model, create confusion matrix
# all_labels = []
# all_predictions = []

# # Test the model
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         all_labels.extend(labels.cpu().numpy())
#         all_predictions.extend(predicted.cpu().numpy())

# print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# cm = confusion_matrix(all_labels, all_predictions)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d')
# plt.title('Confusion Matrix')
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()


# # For Precision-Recall Curve collect scores and actual labels
# all_labels = []
# all_scores = []

# with torch.no_grad():
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         all_scores.append(outputs.cpu().numpy())  # Collect scores
#         all_labels.append(labels.cpu().numpy())   # Collect true labels

# # Convert the lists to NumPy arrays
# all_scores_array = np.vstack(all_scores)  # This stacks the arrays along the vertical axis (row-wise)
# all_labels_array = np.concatenate(all_labels)  # This concatenates the label arrays

# # Convert labels to one-hot encoding for multiclass precision-recall calculation
# all_labels_one_hot = np.eye(10)[all_labels_array]  # CIFAR-10 has 10 classes

# # Calculate precision and recall for each class
# precisions, recalls = {}, {}
# for i in range(10):
#     precisions[i], recalls[i], _ = precision_recall_curve(all_labels_one_hot[:, i], all_scores_array[:, i])

# for i in range(10):
#     print(f"Class {i}:")
#     print(f"Precision: {precisions[i]}")
#     print(f"Recall: {recalls[i]}\n")


# # Plotting the precision-recall curve
# plt.figure(figsize=(12, 8))
# for i in range(10):
#     plt.plot(recalls[i], precisions[i], lw=2, label=f'Class {i}')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve for Each Class')
# plt.legend()
# plt.show()
