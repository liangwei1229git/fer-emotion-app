import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


class CustomSwinTransformer(torch.nn.Module):
    def __init__(self, pretrained=False, num_classes=7):
        super(CustomSwinTransformer, self).__init__()
        self.backbone = None
        try:
            import timm
            self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features
        except:
            in_features = 1024
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.6),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        if self.backbone is not None:
            x = self.backbone(x)
        return self.classifier(x)


def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        model = torch.load(model_path, map_location=device)
        print("Model loaded successfully!")
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, device


def main():
    model_path = "Models/Swin_Transformer/Swin_Transformer_best_model.pth"
    
    if os.path.exists(model_path):
        print(f"Model file found: {model_path}")
        print(f"File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        model, device = load_model(model_path)
        
        if model is not None:
            print("\nModel information:")
            print(f"Model type: {type(model)}")
            
            if hasattr(model, 'state_dict'):
                print("Model has state_dict")
                keys = list(model.state_dict().keys())
                print(f"Number of parameters: {len(keys)}")
                print(f"First few keys: {keys[:5]}")
    else:
        print(f"Model file not found: {model_path}")


if __name__ == "__main__":
    main()

