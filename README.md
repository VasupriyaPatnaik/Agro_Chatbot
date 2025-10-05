# 🌱 AgroBot - AI-Powered Crop Disease Recognition System

AgroBot is a comprehensive AI-powered agricultural assistant that helps farmers diagnose crop health issues through both **text-based symptom analysis** and **advanced image classification**. Using a trained Convolutional Neural Network (CNN) with 94.7% accuracy, it provides instant diagnoses, prevention tips, and treatment recommendations for various crop diseases.

## 🚀 Key Features

### 🤖 Dual Diagnosis Methods
- **💬 Text-Based Diagnosis**: Describe symptoms in natural language
- **📸 AI Image Classification**: Upload plant images for instant analysis
- **🧠 CNN Model**: 94.7% validation accuracy for disease detection

### � Multi-Language Support
- **English** - Full support
- **हिंदी (Hindi)** - Complete translation
- **తెలుగు (Telugu)** - Full localization
- **Real-time Translation**: Google Translate integration

### 🌾 Comprehensive Coverage
- **9+ Crop Types**: Tomato, Rice, Wheat, Maize, Potato, Cotton, Mango, Neem, Papaya
- **21+ Disease Entries**: From common deficiencies to complex diseases
- **3 Disease Classes**: Healthy, Powdery Mildew, Rust (for image classification)

### 🔐 User Management
- **Farmer Interface**: Diagnosis and consultation
- **Admin Dashboard**: Knowledge base management
- **Secure Authentication**: Password hashing and sessions

## 🎯 Supported Diseases & Issues

### 🍅 **Tomato**
- Fusarium wilt, Early blight, Nitrogen deficiency

### 🌾 **Rice** 
- Brown spot disease, Bacterial blight, Potassium deficiency, Tungro virus

### 🌿 **Wheat**
- Stripe rust, Loose smut, Nitrogen deficiency

### 🌽 **Maize**
- Turcicum leaf blight, Stem borer, Zinc deficiency

### 🥔 **Potato**
- Leaf roll virus, Late blight, Magnesium deficiency

### 🌱 **Cotton**
- Boll rot, Pink bollworm, Phosphorus deficiency

*Plus many more diseases with detailed prevention and treatment protocols*

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **AI/ML**: TensorFlow 2.18.1, Keras
- **Image Processing**: OpenCV, PIL, NumPy
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, JavaScript
- **Translation**: Google Translate API
- **Authentication**: Werkzeug security

## 📋 Prerequisites

- Python 3.10+
- 4GB+ RAM (for TensorFlow)
- Modern web browser
- Internet connection (for translations)

## 🚀 Installation & Setup

### 1. **Clone the Repository**
```bash
git clone https://github.com/VasupriyaPatnaik/Agro_Chatbot.git
cd Agro_Chatbot
```

### 2. **Create Virtual Environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux  
source venv/bin/activate
```

### 3. **Install Dependencies**
```bash
# Core dependencies
pip install flask googletrans==4.0.0rc1 werkzeug

# AI/ML dependencies (for image classification)
pip install tensorflow opencv-python numpy pillow scikit-learn

# Additional packages
pip install matplotlib seaborn
```

### 4. **Setup Kaggle (Optional - for dataset)**
- Place your `kaggle.json` in the project directory
- Used for downloading plant disease datasets

## 🎮 Usage

### **Quick Start**
```bash
# Start with AI image classification enabled
python app.py
```

### **Environment Options**
```bash
# Disable ML (basic mode)
set ENABLE_ML=false && python app.py

# Enable debug mode
set FLASK_DEBUG=true && python app.py
```

### **Access the Application**
```
🌐 Web Interface: http://127.0.0.1:5000
```

### **Login Credentials**
```
👨‍🌾 Farmer: farmer / farmer123
👨‍💼 Admin:  admin / admin123
```

## 🖥️ User Interfaces

### **Farmer Interface**
- Interactive chatbot with suggested symptoms
- Image upload for disease classification  
- Multi-language support
- Treatment recommendations

### **Admin Dashboard**
- Knowledge base management (CRUD operations)
- System statistics and ML status
- User management capabilities

## 📊 AI Model Performance

### **Training Results**
- **Validation Accuracy**: 94.7%
- **Training Accuracy**: 93.3%
- **Model Architecture**: CNN with BatchNormalization
- **Input Size**: 225×225×3 (RGB images)
- **Classes**: 3 (Healthy, Powdery, Rust)

### **Training Configuration**
- **Epochs**: 20 (with early stopping)
- **Batch Size**: 32
- **Data Augmentation**: Rotation, zoom, shift, brightness
- **Callbacks**: ReduceLROnPlateau, EarlyStopping

## 🔌 API Endpoints

### **Text Diagnosis**
```bash
POST /diagnose
Content-Type: application/json

{
  "text": "My tomato plants have yellow leaves with brown spots",
  "lang": "en",
  "location": "Maharashtra"
}
```

### **Image Prediction**
```bash
POST /predict
Content-Type: multipart/form-data

{
  "file": <image_file>,
  "lang": "en",
  "location": "Maharashtra"
}
```

### **Admin Operations**
```bash
GET    /admin/data     # View knowledge base
POST   /admin/data     # Add entry
PUT    /admin/data     # Update entry  
DELETE /admin/data     # Delete entry
```

## 📁 Project Structure

```
AgroBot/
├── app.py                          # Main Flask application
├── model.h5                        # Trained CNN model (94.7% accuracy)
├── model.ipynb                     # Jupyter notebook with training pipeline
├── knowledge_base.json             # Disease database (21+ entries)
├── agrobot.db                      # SQLite user database
├── requirements.txt                # Python dependencies
├── kaggle.json                     # Kaggle API credentials
├── templates/
│   ├── index.html                  # Farmer interface
│   ├── admin.html                  # Admin dashboard
│   └── login.html                  # Authentication page
├── uploads/                        # Temporary image storage
├── plant_leaf_disease_predictor/   # Training dataset
└── __pycache__/                    # Python cache
```

## 🧠 Model Training

The CNN model was trained using the included Jupyter notebook:

### **Dataset**
- **Source**: Kaggle plant disease recognition dataset
- **Classes**: Healthy, Powdery Mildew, Rust
- **Training Images**: ~1,300 images
- **Validation Images**: ~150 images

### **Architecture**
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(225,225,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(), 
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
```

### **Training Features**
- Data augmentation for better generalization
- Class weight balancing
- Learning rate scheduling
- Early stopping to prevent overfitting

## 🌍 Multi-Language Features

### **Supported Languages**
- **English**: Complete interface and responses
- **Hindi**: Full translation with proper Unicode support
- **Telugu**: Complete localization for Telugu-speaking farmers

### **Translation Capabilities**
- Real-time symptom translation
- Diagnosis result translation
- User interface localization
- Fallback to original text if translation fails

## 🔧 Configuration

### **Environment Variables**
```bash
ENABLE_ML=true           # Enable/disable TensorFlow
FLASK_SECRET_KEY=secret  # Session security key
FLASK_DEBUG=false        # Debug mode toggle
```

### **Customization Options**
- Modify `knowledge_base.json` to add new diseases
- Update `templates/` for UI customization
- Retrain model with new plant disease data
- Add new language translations

## 🛠️ Development Setup

### **Model Training Environment**
```bash
# Install additional training dependencies
pip install kaggle jupyter matplotlib seaborn

# Launch Jupyter for model training
jupyter notebook model.ipynb
```

### **Adding New Diseases**
1. Update `knowledge_base.json` with new entries
2. Follow the existing JSON schema
3. Include prevention and treatment information
4. Test with the application

## 🚨 Troubleshooting

### **Common Issues**

**TensorFlow Import Errors**
```bash
# Try without ML first
set ENABLE_ML=false && python app.py
```

**Missing Model File**
```bash
# Train new model using model.ipynb
# Or download pre-trained model.h5
```

**Translation Errors**
```bash
# Check internet connection
# Verify Google Translate API access
```

**Database Issues**
```bash
# Delete agrobot.db to reset
# Database will be recreated automatically
```

## 🧪 Testing

### **Test Image Classification**
1. Upload plant images through the web interface
2. Check terminal for classification confidence scores
3. Verify results match expected disease types

### **Test Text Diagnosis**
```javascript
// Example test inputs
"My tomato plants have yellow leaves"
"Rice showing brown spots on leaves"  
"Wheat plants with orange rust on stems"
```

## 🤝 Contributing

### **How to Contribute**
1. Fork the repository
2. Add new disease entries to knowledge base
3. Improve model accuracy with more training data
4. Add support for new languages
5. Enhance UI/UX features

### **Contribution Guidelines**
- Follow existing code style
- Add comprehensive comments
- Test new features thoroughly
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle** for plant disease datasets
- **TensorFlow/Keras** for deep learning framework
- **Flask** for web application framework
- **Agricultural experts** for domain knowledge
- **Farmers** for real-world testing and feedback

## 📞 Support & Contact

- **GitHub Issues**: Report bugs and feature requests
- **Email**: Technical support and questions
- **Documentation**: Check this README for common solutions

## 🎯 Future Enhancements

- [ ] Support for more crop types
- [ ] Mobile app development
- [ ] Integration with weather APIs
- [ ] Soil health analysis
- [ ] Pest identification features
- [ ] Offline mode capabilities
- [ ] Real-time chat with agricultural experts

---

**🌾 Empowering Farmers with AI Technology! 👨‍🌾👩‍🌾**

*Made with ❤️ for sustainable agriculture and food security*