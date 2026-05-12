import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
import os
import numpy as np


class FERInferenceEnhanced:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        self.emotion_intensity = {
            'Angry': ['Slightly Angry', 'Angry', 'Very Angry', 'Extremely Angry'],
            'Disgust': ['Slightly Disgusted', 'Disgusted', 'Very Disgusted', 'Extremely Disgusted'],
            'Fear': ['Slightly Fearful', 'Fearful', 'Very Fearful', 'Extremely Fearful'],
            'Happy': ['Slightly Happy', 'Happy', 'Very Happy', 'Extremely Happy'],
            'Sad': ['Slightly Sad', 'Sad', 'Very Sad', 'Extremely Sad'],
            'Surprise': ['Slightly Surprised', 'Surprised', 'Very Surprised', 'Extremely Surprised'],
            'Neutral': ['Calm', 'Neutral', 'Very Neutral', 'Completely Neutral']
        }
        
        self.mental_health_notes = {
            'Sad': '长时间的悲伤可能与抑郁有关，建议关注情绪变化',
            'Angry': '频繁的愤怒可能与压力或焦虑有关',
            'Fear': '持续的恐惧可能是焦虑障碍的征兆',
            'Happy': '积极的情绪对心理健康有益！',
            'Neutral': '中性情绪是正常的情绪状态'
        }
        
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
            return None

    def get_intensity_level(self, confidence):
        if confidence < 0.4:
            return 0
        elif confidence < 0.6:
            return 1
        elif confidence < 0.85:
            return 2
        else:
            return 3

    def predict(self, image_path):
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
            
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
            emotion = self.emotions[predicted.item()]
            confidence = probabilities[0][predicted.item()].item()
            
            intensity_level = self.get_intensity_level(confidence)
            intensity_desc = self.emotion_intensity[emotion][intensity_level]
            
            all_probabilities = {self.emotions[i]: float(probabilities[0][i]) 
                                for i in range(len(self.emotions))}
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'intensity_level': intensity_level,
                'intensity_description': intensity_desc,
                'all_probabilities': all_probabilities,
                'mental_health_note': self.mental_health_notes.get(emotion, '')
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    def print_result(self, result):
        if result is None:
            return
            
        print("=" * 60)
        print("📊 情绪识别结果")
        print("=" * 60)
        print(f"主要情绪:      {result['emotion']}")
        print(f"置信度:        {result['confidence']:.2%}")
        print(f"情绪强度:      {result['intensity_description']}")
        print(f"强度等级:      {result['intensity_level'] + 1}/4")
        print("\n" + "-" * 60)
        print("📈 所有情绪概率:")
        for emo, prob in sorted(result['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 30)
            print(f"  {emo:12} {prob:6.2%} |{bar:<30}|")
        print("-" * 60)
        if result['mental_health_note']:
            print(f"\n💡 心理健康提示: {result['mental_health_note']}")
        print("=" * 60)


def main():
    print("🎭 增强版面部表情识别系统")
    print("支持情绪强度判断和心理健康提示\n")
    
    model_path = "Models/Swin_Transformer/Swin_Transformer_best_model.pth"
    
    fer = FERInferenceEnhanced(model_path)
    
    if fer.model is None:
        print("模型加载失败！")
        return
    
    print("\n系统就绪！可以使用 fer.predict(image_path) 进行预测")


if __name__ == "__main__":
    main()

