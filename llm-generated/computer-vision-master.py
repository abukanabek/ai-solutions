"""
4_computer_vision_master.py
Comprehensive Computer Vision Pipelines
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==================== CNN ARCHITECTURES ====================

class AdvancedCNN(nn.Module):
    """Advanced CNN with multiple architecture options"""
    
    def __init__(self, num_classes=10, architecture='resnet-like', dropout=0.5):
        super(AdvancedCNN, self).__init__()
        
        if architecture == 'resnet-like':
            self.features = nn.Sequential(
                # Initial conv block
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                # Residual block 1
                self._residual_block(64, 64),
                self._residual_block(64, 64),
                
                # Residual block 2
                self._residual_block(64, 128, stride=2),
                self._residual_block(128, 128),
                
                # Residual block 3
                self._residual_block(128, 256, stride=2),
                self._residual_block(256, 256),
            )
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes)
            )
            
        elif architecture == 'simple':
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )
    
    def _residual_block(self, in_channels, out_channels, stride=1):
        """Create residual block"""
        shortcut = None
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        return ResidualBlock(layers, shortcut)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, layers, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.layers = layers
        self.shortcut = shortcut
        
    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(x)
        
        out = self.layers(x)
        out += identity
        out = F.relu(out)
        return out

class UNet(nn.Module):
    """U-Net for segmentation"""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(UNet._block(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = UNet._block(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(UNet._block(feature * 2, feature))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)
        
        return torch.sigmoid(self.final_conv(x))

# ==================== DATA AUGMENTATION ====================

class AdvancedAugmentation:
    """Comprehensive data augmentation for images"""
    
    @staticmethod
    def get_train_transforms(img_size=224):
        """Training transforms with heavy augmentation"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_val_transforms(img_size=224):
        """Validation transforms (minimal augmentation)"""
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

# ==================== COMPUTER VISION PIPELINES ====================

class ImageClassificationPipeline:
    """Complete image classification pipeline"""
    
    def __init__(self, num_classes, model_name='resnet18', pretrained=True, device='auto'):
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = self._create_model(model_name, num_classes, pretrained)
        self.model.to(self.device)
        
    def _get_device(self, device):
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _create_model(self, model_name, num_classes, pretrained):
        """Create model architecture"""
        if model_name == 'resnet18':
            model = torchvision.models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'efficientnet':
            model = torchvision.models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'custom':
            model = AdvancedCNN(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return model
    
    def train(self, train_loader, val_loader, epochs=25, lr=0.001):
        """Train the model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # Validation phase
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            scheduler.step()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs
        }
        
        return history
    
    def evaluate(self, data_loader, criterion):
        """Evaluate model performance"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = running_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def predict(self, data_loader):
        """Make predictions"""
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

class ObjectDetectionPipeline:
    """Object detection pipeline with YOLO/SSD"""
    
    def __init__(self, model_type='yolov5', num_classes=80, device='auto'):
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = self._get_device(device)
        self.model = self._create_model(model_type, num_classes)
        
    def _create_model(self, model_type, num_classes):
        """Create object detection model"""
        if model_type == 'yolov5':
            # Using torch hub for YOLOv5
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        elif model_type == 'fasterrcnn':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            # Modify for custom number of classes
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model.to(self.device)
    
    def detect_objects(self, image_path, confidence_threshold=0.5):
        """Detect objects in an image"""
        if self.model_type == 'yolov5':
            results = self.model(image_path)
            detections = results.pandas().xyxy[0]
            filtered_detections = detections[detections['confidence'] > confidence_threshold]
            return filtered_detections
        else:
            # For Faster R-CNN
            image = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            return predictions[0]

class ImageSegmentationPipeline:
    """Image segmentation pipeline"""
    
    def __init__(self, model_type='unet', num_classes=2, device='auto'):
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = self._get_device(device)
        self.model = self._create_model(model_type, num_classes)
        
    def _create_model(self, model_type, num_classes):
        if model_type == 'unet':
            return UNet(in_channels=3, out_channels=num_classes).to(self.device)
        elif model_type == 'deeplabv3':
            model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
            model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
            return model.to(self.device)
        else:
            raise ValueError(f"Unsupported segmentation model: {model_type}")
    
    def train_segmentation(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Train segmentation model"""
        criterion = nn.BCEWithLogitsLoss() if self.num_classes == 2 else nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            
            for images, masks in train_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    def segment_image(self, image_path):
        """Segment a single image"""
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            if self.num_classes == 2:
                prediction = torch.sigmoid(output) > 0.5
            else:
                prediction = torch.argmax(output, dim=1)
        
        return prediction.squeeze().cpu().numpy()

# ==================== UTILITY FUNCTIONS ====================

def visualize_predictions(images, predictions, true_labels=None, class_names=None, num_images=8):
    """Visualize model predictions"""
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        axes[i].imshow(images[i].permute(1, 2, 0))
        
        pred_class = predictions[i]
        pred_name = class_names[pred_class] if class_names else str(pred_class)
        
        if true_labels is not None:
            true_class = true_labels[i]
            true_name = class_names[true_class] if class_names else str(true_class)
            color = 'green' if pred_class == true_class else 'red'
            axes[i].set_title(f'Pred: {pred_name}\nTrue: {true_name}', color=color)
        else:
            axes[i].set_title(f'Pred: {pred_name}')
        
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Example usage
if __name__ == "__main__":
    # This would require actual image data to run
    print("Computer Vision Pipelines Ready!")
    print("Available pipelines:")
    print("1. Image Classification")
    print("2.
