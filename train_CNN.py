import os
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Define the Convolutional Neural Network (CNN) architecture
class AniCNN(nn.Module):
    def __init__(self):
        super(AniCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) 
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout layers
        self.dropout1 = nn.Dropout(0.25) 
        self.dropout2 = nn.Dropout(0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 64 * 64, 512)  # Assuming input image size is 64x64
        self.fc2 = nn.Linear(512, len(set(dataset.labels)))  # Output layer

    def forward(self, x):
        # Forward pass through the network
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 64 * 64)  # Flatten the output for the fully connected layers
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
# Define a custom dataset class for loading and preprocessing images
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, allowed_extensions=('jpg', 'jpeg', 'png', 'bmp', 'gif')):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        # Traverse through the root directory to collect images and their labels
        for dir_name, _, file_names in os.walk(root_dir):
            if file_names:
                parts = dir_name.split(os.sep)[-1].split('_')
                # Specify the training scope by checking folder name parts
                if len(parts) >= 4 and parts[3] == 'Aves':
                    label = parts[4]  # Extract label from folder name
                    if label not in self.class_to_idx:
                        self.class_to_idx[label] = len(self.class_to_idx)
                    label_index = self.class_to_idx[label]
                    for file_name in file_names:
                        if file_name.split('.')[-1].lower() in allowed_extensions:
                            # Add image path and its corresponding label to the lists
                            self.images.append(os.path.join(dir_name, file_name))
                            self.labels.append(label_index)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and its label
        img_name = self.images[idx]
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            # Apply transformations if specified
            image = self.transform(image)

        return image, label

# Define the image transformations to be applied
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to 256x256
    transforms.ToTensor(),           # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image data
])

if __name__ == '__main__':
    dataset_path = 'train_mini'  # Path to the dataset
    # Create an instance of the custom dataset
    dataset = CustomDataset(root_dir=dataset_path, transform=transform)
    # Create a data loader for batching and shuffling the data
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the CNN model
    model = AniCNN()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(2):  # Train for 2 epochs
        running_loss = 0.0
        # Iterate over batches of data
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)):
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        print(f'Completed Epoch {epoch+1}')

    # Save the trained model weights
    model_path = 'weights/bird_weights.pth'
    torch.save(model.state_dict(), model_path)
