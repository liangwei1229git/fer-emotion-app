from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import torch
import timm
from torchvision import transforms
import os
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

class FERModelOptimized:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        
        self.mental_health_score = {
            'Happy': 85,
            'Neutral': 70,
            'Surprise': 60,
            'Angry': 40,
            'Fear': 35,
            'Disgust': 30,
            'Sad': 25
        }
        
        self.mental_health_notes = {
            'Sad': '长时间的悲伤可能与抑郁有关，建议关注情绪变化，必要时寻求帮助',
            'Angry': '频繁的愤怒可能与压力或焦虑有关，尝试放松技巧',
            'Fear': '持续的恐惧可能是焦虑障碍的征兆，建议咨询专业人士',
            'Happy': '积极的情绪对心理健康有益！继续保持',
            'Neutral': '中性情绪是正常的情绪状态，注意观察情绪变化',
            'Surprise': '惊讶是正常的情绪反应',
            'Disgust': '厌恶情绪可能与压力有关'
        }
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.face_cascade = self._load_face_cascade()
        self.model = self._load_model(model_path)
    
    def _load_face_cascade(self):
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                return cv2.CascadeClassifier(cascade_path)
            else:
                print("Warning: Haar cascade not found, face detection disabled")
                return None
        except Exception as e:
            print(f"Error loading face cascade: {e}")
            return None
    
    def detect_and_crop_face(self, image):
        if self.face_cascade is None:
            return image, None
        
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                margin = int(0.2 * w)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(img_array.shape[1] - x, w + 2 * margin)
                h = min(img_array.shape[0] - y, h + 2 * margin)
                
                face_img = img_array[y:y+h, x:x+w]
                face_pil = Image.fromarray(face_img)
                
                return face_pil, (x, y, w, h)
            else:
                return image, None
        except Exception as e:
            print(f"Error in face detection: {e}")
            return image, None
    
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
    
    def predict(self, image):
        if self.model is None:
            return None
        
        try:
            face_image, face_rect = self.detect_and_crop_face(image)
            
            input_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
            
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
            
            mental_health_score = self.mental_health_score.get(emotion, 50)
            adjusted_score = int(mental_health_score + (confidence - 0.5) * 20)
            adjusted_score = max(0, min(100, adjusted_score))
            
            face_detected = face_rect is not None
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'intensity_level': intensity_level,
                'intensity_description': intensity_desc,
                'all_probabilities': all_probabilities,
                'mental_health_score': adjusted_score,
                'mental_health_note': self.mental_health_notes.get(emotion, ''),
                'emotion_zh': {
                    'Angry': '愤怒',
                    'Disgust': '厌恶',
                    'Fear': '恐惧',
                    'Happy': '开心',
                    'Sad': '悲伤',
                    'Surprise': '惊讶',
                    'Neutral': '中性'
                }.get(emotion, emotion),
                'face_detected': face_detected
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

model_path = "Models/Swin_Transformer/Swin_Transformer_best_model.pth"
fer_model = FERModelOptimized(model_path)

@app.route('/')
def index():
    return render_template('index_optimized.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        result = fer_model.predict(image)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Prediction failed'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)

