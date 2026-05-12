from inference_enhanced import FERInferenceEnhanced

# 初始化增强版模型
fer = FERInferenceEnhanced('Models/Swin_Transformer/Swin_Transformer_best_model.pth')

# 预测并获取完整结果
result = fer.predict('images/sad.png')

# 打印美观的结果
fer.print_result(result)