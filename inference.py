import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
import os
import numpy as np


class FERInference:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=7)
            model.load_state_dict(checkpoint)
            model.to(self.device)
            model.eval()
            
            print("Model loaded successfully!")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict(self, image_path):
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None, None
            
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
            emotion = self.emotions[predicted.item()]
            confidence = probabilities[0][predicted.item()].item()
            
            return emotion, confidence
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def main():
    model_path = "Models/Swin_Transformer/Swin_Transformer_best_model.pth"
    
    fer = FERInference(model_path)
    
    if fer.model is not None:
        print("\nFER System is ready!")
        print("\nEmotions recognized:", fer.emotions)
        print("\nTo use this model, call fer.predict(image_path)")


if __name__ == "__main__":
    main()

