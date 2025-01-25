
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Paths for images and CSV files
train_csv_path = r"path"
train_img_dir = r"path"

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to fixed size for consistency
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load Data function
def load_data(csv_path, img_dir, transform):
    data = pd.read_csv(csv_path)
    image_data, text_data = [], []
    for idx, row in data.iterrows():
        image_name = row['imagename']
        text = row['captions']
        image_path = os.path.join(img_dir, image_name)
        try:
            image = Image.open(image_path).convert('RGB')
            if transform:
                image = transform(image)
            image_data.append(image)
            text_data.append(text)
        except FileNotFoundError:
            continue
    return image_data, text_data

# Load training data
train_image_data, train_text_data = load_data(train_csv_path, train_img_dir, transform)
if not train_image_data:
    print("No training data loaded.")
    exit()

# Residual Block for improved feature learning
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x += residual  # Skip connection
        x = self.relu(x)
        return x

class Generator(nn.Module):
    def __init__(self, z_dim, text_embedding_dim, img_channels):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + text_embedding_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.LeakyReLU(0.2)
        )
        
        # Progressive upsampling layers for higher resolution output
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ResidualBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            ResidualBlock(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.Tanh()
        )

    def forward(self, z, text_embedding):
        x = torch.cat([z, text_embedding], dim=1)
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        x = self.upsample(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, img_channels, text_embedding_dim):
        super(Discriminator, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64, 64), 
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128, 128), 
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8 + text_embedding_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, text_embedding):
        x = self.downsample(img)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, text_embedding], dim=1)
        x = self.fc(x)
        return x

# Hyperparameters
z_dim = 100
text_embedding_dim = 768
img_channels = 3
num_epochs = 100
batch_size = 32
lr = 0.0001

# Initialize models and optimizers
generator = Generator(z_dim, text_embedding_dim, img_channels).to(device)
discriminator = Discriminator(img_channels, text_embedding_dim).to(device)
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# BERT setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Dataset class
class MemeDataset(Dataset):
    def __init__(self, image_data, text_data, tokenizer):
        self.image_data = image_data
        self.text_data = text_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = self.image_data[idx]
        text = self.text_data[idx]
        tokens = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
        return image, tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

# Create DataLoader for training
train_dataset = MemeDataset(train_image_data, train_text_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for images, captions, attention_masks in train_loader:
        images = images.to(device)
        captions, attention_masks = captions.to(device), attention_masks.to(device)
        
        # Get text embeddings from BERT
        with torch.no_grad():
            text_embeddings = bert_model(captions, attention_mask=attention_masks).pooler_output

        # Discriminator training
        optimizer_d.zero_grad()
        real_labels = torch.ones(images.size(0), 1).to(device)
        fake_labels = torch.zeros(images.size(0), 1).to(device)
        
        outputs = discriminator(images, text_embeddings)
        d_loss_real = criterion(outputs, real_labels)
        
        z = torch.randn(images.size(0), z_dim).to(device)
        fake_images = generator(z, text_embeddings)
        outputs = discriminator(fake_images.detach(), text_embeddings)
        d_loss_fake = criterion(outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Generator training
        optimizer_g.zero_grad()
        outputs = discriminator(fake_images, text_embeddings)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Generate a meme based on a caption
def generate_meme(caption):
    # Set the generator to evaluation mode
    generator.eval()
    caption_tokens = tokenizer(caption, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
    caption_tokens = caption_tokens['input_ids'].to(device)
    attention_mask = caption_tokens.ne(0).type(torch.int).to(device)  # Create attention mask

    with torch.no_grad():
        text_embedding = bert_model(caption_tokens, attention_mask=attention_mask).pooler_output
        z = torch.randn(1, z_dim).to(device)
        fake_image = generator(z, text_embedding).detach().cpu().squeeze()

    # Convert the tensor to a PIL Image
    fake_image = (fake_image * 0.5 + 0.5).clamp(0, 1)  # Rescale to [0, 1]
    fake_image = transforms.ToPILImage()(fake_image)
    return fake_image

# Example usage
caption = input("Your meme caption here:")
meme_image = generate_meme(caption)
meme_image.show()
