from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import os
import tempfile

os.environ['DEEPFACE_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.deepface')

app = Flask(__name__)
CORS(app)

class FERDeepFace:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        self.emotion_intensity = {
            'angry': ['轻微愤怒', '愤怒', '非常愤怒', '极度愤怒'],
            'disgust': ['轻微厌恶', '厌恶', '非常厌恶', '极度厌恶'],
            'fear': ['轻微恐惧', '恐惧', '非常恐惧', '极度恐惧'],
            'happy': ['轻微开心', '开心', '非常开心', '极度开心'],
            'sad': ['轻微悲伤', '悲伤', '非常悲伤', '极度悲伤'],
            'surprise': ['轻微惊讶', '惊讶', '非常惊讶', '极度惊讶'],
            'neutral': ['平静', '中性', '非常平静', '完全平静']
        }
        
        self.mental_health_score = {
            'happy': 90,
            'neutral': 75,
            'surprise': 65,
            'angry': 45,
            'fear': 40,
            'disgust': 35,
            'sad': 30
        }
        
        self.mental_health_advice = {
            'happy': {
                'title': '保持积极心态',
                'suggestions': [
                    '继续保持积极的生活态度',
                    '与家人朋友分享你的快乐',
                    '记录让你开心的事情',
                    '继续做自己喜欢的活动'
                ],
                'color': '#4CAF50'
            },
            'neutral': {
                'title': '关注情绪变化',
                'suggestions': [
                    '中性情绪是正常的，注意观察自己的情绪变化',
                    '尝试做一些让自己放松的事情',
                    '保持规律的作息时间',
                    '适当进行运动'
                ],
                'color': '#9E9E9E'
            },
            'surprise': {
                'title': '适应变化',
                'suggestions': [
                    '惊讶是对新事物的正常反应',
                    '给自己时间适应变化',
                    '保持开放的心态',
                    '与他人分享你的感受'
                ],
                'color': '#FF9800'
            },
            'sad': {
                'title': '关爱自己',
                'suggestions': [
                    '允许自己感受悲伤，这是正常的情绪',
                    '与信任的人谈谈你的感受',
                    '做一些能让你感到安慰的事情',
                    '如果悲伤持续，考虑寻求专业帮助',
                    '保持规律的饮食和睡眠'
                ],
                'color': '#2196F3',
                'warning': '如果这种情绪持续超过两周，建议咨询心理医生'
            },
            'angry': {
                'title': '管理愤怒情绪',
                'suggestions': [
                    '愤怒是正常的，但要学会健康地表达',
                    '尝试深呼吸或冥想平静下来',
                    '找出愤怒的根源',
                    '用"我"语句表达感受，而非指责他人',
                    '考虑通过运动释放情绪'
                ],
                'color': '#F44336'
            },
            'fear': {
                'title': '面对恐惧',
                'suggestions': [
                    '恐惧是自我保护的本能',
                    '尝试逐步面对让你害怕的事物',
                    '学习放松技巧',
                    '与他人分享你的担忧',
                    '如果恐惧影响生活，寻求专业帮助'
                ],
                'color': '#9C27B0',
                'warning': '持续的恐惧可能是焦虑的表现，建议咨询专业人士'
            },
            'disgust': {
                'title': '理解厌恶情绪',
                'suggestions': [
                    '厌恶是一种自我保护机制',
                    '识别触发厌恶情绪的原因',
                    '保持个人卫生和环境整洁',
                    '尝试从不同角度看待事物',
                    '如果情绪强烈，尝试转移注意力'
                ],
                'color': '#795548'
            }
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
        
        self.emotion_emoji = {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happy': '😊',
            'sad': '😢',
            'surprise': '😲',
            'neutral': '😐'
        }
        
        print("DeepFace initialized successfully!")
    
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
            
            advice = self.mental_health_advice.get(dominant_emotion, {})
            
            return {
                'success': True,
                'emotion': dominant_emotion,
                'emotion_zh': self.emotion_zh.get(dominant_emotion, dominant_emotion),
                'emotion_emoji': self.emotion_emoji.get(dominant_emotion, '😊'),
                'confidence': confidence,
                'confidence_percent': round(confidence * 100, 1),
                'intensity_level': intensity_level,
                'intensity_description': intensity_desc,
                'all_probabilities': all_probabilities,
                'mental_health_score': adjusted_score,
                'advice': advice
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

fer_model = FERDeepFace()

@app.route('/')
def index():
    return render_template('index_final.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        result = fer_model.predict(image)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5004)

