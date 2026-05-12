#!/usr/bin/env python3

from inference_enhanced import FERInferenceEnhanced
import os
from PIL import Image, ImageDraw, ImageFont


def create_demo_image(emotion_type="happy"):
    """Create a simple demo image for testing"""
    width, height = 400, 400
    image = Image.new('RGB', (width, height), color='lightgray')
    draw = ImageDraw.Draw(image)
    
    draw.rectangle([50, 50, 350, 350], outline='black', width=3)
    draw.ellipse([100, 100, 150, 130], fill='black')
    draw.ellipse([250, 100, 300, 130], fill='black')
    
    if emotion_type == "sad":
        draw.arc([150, 250, 250, 300], start=180, end=360, fill='black', width=5)
    else:
        draw.arc([150, 200, 250, 280], start=0, end=180, fill='black', width=5)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    label = f"Demo {emotion_type.capitalize()}"
    draw.text((120, 360), label, fill='black', font=font)
    
    return image


def main():
    print("=" * 60)
    print("🎭 增强版面部表情识别 - Demo")
    print("支持情绪强度判断和心理健康提示")
    print("=" * 60)
    
    model_path = "Models/Swin_Transformer/Swin_Transformer_best_model.pth"
    
    print("\n[1/3] 加载模型...")
    fer = FERInferenceEnhanced(model_path)
    
    if fer.model is None:
        print("错误: 模型加载失败")
        return
    
    for emotion in ["happy", "sad"]:
        print(f"\n[2/3] 创建 {emotion} 测试图片...")
        test_image = create_demo_image(emotion)
        test_image_path = f"test_demo_{emotion}.jpg"
        test_image.save(test_image_path)
        print(f"测试图片已保存到: {test_image_path}")
        
        print(f"\n[3/3] 运行 {emotion} 推理...")
        result = fer.predict(test_image_path)
        
        if result:
            fer.print_result(result)
        
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
    
    print("\n" + "=" * 60)
    print("Demo 完成！")
    print("\n如何使用真实图片:")
    print("1. 准备一张面部图片 (JPG, PNG 等格式)")
    print("2. 使用代码:")
    print("   from inference_enhanced import FERInferenceEnhanced")
    print("   fer = FERInferenceEnhanced('Models/Swin_Transformer/Swin_Transformer_best_model.pth')")
    print("   result = fer.predict('path/to/your/image.jpg')")
    print("   fer.print_result(result)")
    print("=" * 60)


if __name__ == "__main__":
    main()

