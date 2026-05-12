from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import os
import tempfile
import numpy as np

os.environ['DEEPFACE_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.deepface')

app = Flask(__name__)
CORS(app)

class FERDeepFace:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        self.emotion_intensity = {
            'angry': ['Slightly Angry', 'Angry', 'Very Angry', 'Extremely Angry'],
            'disgust': ['Slightly Disgusted', 'Disgusted', 'Very Disgusted', 'Extremely Disgusted'],
            'fear': ['Slightly Fearful', 'Fearful', 'Very Fearful', 'Extremely Fearful'],
            'happy': ['Slightly Happy', 'Happy', 'Very Happy', 'Extremely Happy'],
            'sad': ['Slightly Sad', 'Sad', 'Very Sad', 'Extremely Sad'],
            'surprise': ['Slightly Surprised', 'Surprised', 'Very Surprised', 'Extremely Surprised'],
            'neutral': ['Calm', 'Neutral', 'Very Neutral', 'Completely Neutral']
        }
        
        self.mental_health_score = {
            'happy': 85,
            'neutral': 70,
            'surprise': 60,
            'angry': 40,
            'fear': 35,
            'disgust': 30,
            'sad': 25
        }
        
        self.mental_health_notes = {
            'sad': '长时间的悲伤可能与抑郁有关，建议关注情绪变化，必要时寻求帮助',
            'angry': '频繁的愤怒可能与压力或焦虑有关，尝试放松技巧',
            'fear': '持续的恐惧可能是焦虑障碍的征兆，建议咨询专业人士',
            'happy': '积极的情绪对心理健康有益！继续保持',
            'neutral': '中性情绪是正常的情绪状态，注意观察情绪变化',
            'surprise': '惊讶是正常的情绪反应',
            'disgust': '厌恶情绪可能与压力有关'
        }
        
        self.emotion_zh = {
            'angry': '愤怒',
            'disgust': '厌恶',
            'fear': '恐惧',
            'happy': '开心',
            'sad': '悲伤',
            'surprise': '惊讶',
            'neutral': '中性'
        }
        
        print("DeepFace initialized!")
    
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
        try:
            from deepface import DeepFace
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            image.save(temp_path)
            
            result = DeepFace.analyze(
                img_path=temp_path,
                actions=['emotion'],
                enforce_detection=False
            )
            
            os.unlink(temp_path)
            
            if isinstance(result, list) and len(result) > 0:
                emotion_result = result[0]
            else:
                emotion_result = result
            
            dominant_emotion = emotion_result['dominant_emotion']
            emotion_confidences = emotion_result['emotion']
            
            confidence = float(emotion_confidences[dominant_emotion]) / 100.0
            
            intensity_level = self.get_intensity_level(confidence)
            intensity_desc = self.emotion_intensity[dominant_emotion][intensity_level]
            
            all_probabilities = {}
            for emo, conf in emotion_confidences.items():
                all_probabilities[emo] = float(conf) / 100.0
            
            mental_health_score = self.mental_health_score.get(dominant_emotion, 50)
            adjusted_score = int(mental_health_score + (confidence - 0.5) * 20)
            adjusted_score = max(0, min(100, adjusted_score))
            
            return {
                'emotion': dominant_emotion,
                'confidence': confidence,
                'intensity_level': intensity_level,
                'intensity_description': intensity_desc,
                'all_probabilities': all_probabilities,
                'mental_health_score': adjusted_score,
                'mental_health_note': self.mental_health_notes.get(dominant_emotion, ''),
                'emotion_zh': self.emotion_zh.get(dominant_emotion, dominant_emotion),
                'face_detected': True
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

fer_model = FERDeepFace()

@app.route('/')
def index():
    return render_template('index_deepface.html')

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
    app.run(debug=True, host='0.0.0.0', port=5003)

