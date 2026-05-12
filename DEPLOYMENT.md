# 🚀 部署指南

## 方案一：免费网站部署（推荐）

### 1. 使用 Render（免费）
- **网站**：https://render.com
- **优点**：免费、简单、自动部署
- **步骤**：
  1. 把代码推送到 GitHub
  2. 在 Render 上创建 Web Service
  3. 连接 GitHub 仓库
  4. 配置：
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `gunicorn app_final:app`
  5. 部署！

### 2. 使用 PythonAnywhere（免费）
- **网站**：https://www.pythonanywhere.com
- **优点**：简单、有免费额度
- **步骤**：
  1. 注册账号
  2. 上传代码文件
  3. 创建 Web App（Flask）
  4. 配置虚拟环境并安装依赖
  5. 启动！

### 3. 使用 Vercel（免费）
- **网站**：https://vercel.com
- **优点**：超快、全球CDN
- **步骤**：
  1. 推送到 GitHub
  2. 导入到 Vercel
  3. 配置环境变量
  4. 部署！

---

## 方案二：云服务器部署

### 使用阿里云/腾讯云/AWS
1. 购买云服务器（推荐 2核4G以上）
2. 安装 Python、Nginx
3. 上传代码
4. 使用 Gunicorn + Nginx 部署
5. 配置域名和SSL证书

---

## 方案三：小程序部署

### 微信小程序
1. 注册微信小程序账号
2. 使用微信开发者工具
3. 前端使用 WXML/WXSS 重写
4. 后端部署到云服务器
5. 调用后端API

---

## 📦 快速开始（Render示例）

### 1. 创建 requirements.txt
```txt
flask
flask-cors
deepface
opencv-python
pillow
numpy
pandas
matplotlib
seaborn
scikit-learn
gunicorn
tf-keras
```

### 2. 创建 Procfile
```txt
web: gunicorn app_final:app --timeout 120
```

### 3. 推送到 GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <你的GitHub仓库>
git push
```

### 4. 在 Render 上部署
- 访问 https://render.com
- 点击 "New +" → "Web Service"
- 连接你的 GitHub 仓库
- 配置：
  - Name: fer-emotion-app
  - Region: Singapore（新加坡，速度快）
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `gunicorn app_final:app --timeout 120`
  - Instance Type: Free
- 点击 "Create Web Service"

### 5. 完成！
- 等待几分钟部署
- 获得一个类似 `https://fer-emotion-app.onrender.com` 的网址
- 分享给朋友体验！

---

## ⚠️ 注意事项

1. **模型大小**：DeepFace 模型较大，首次加载需要时间
2. **内存要求**：建议至少 2GB 内存
3. **超时设置**：增加超时时间，因为模型加载需要时间
4. **HTTPS**：生产环境必须使用 HTTPS
5. **相机权限**：浏览器需要 HTTPS 才能访问相机

---

## 🎯 最简单的方案（推荐）

**使用 Render 免费部署**：
- ✅ 完全免费
- ✅ 自动 HTTPS
- ✅ 自动部署
- ✅ 全球CDN
- ✅ 简单易用

---

需要我帮你准备具体的部署文件吗？

