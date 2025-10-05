# 🌱 AgroBot - Crop Symptom Assistant

AgroBot is an AI-powered chatbot that helps farmers diagnose crop health issues based on symptoms they describe. It provides instant diagnoses, prevention tips, and treatment recommendations for various crop diseases and deficiencies.

## Features

- 🤖 Interactive chatbot interface
- 🌾 Support for multiple crops (Tomato, Rice, Wheat, Maize, Potato, Cotton)
- 🔍 Symptom-based diagnosis
- 💡 Detailed prevention and treatment recommendations
- 🌱 Organic treatment options
- 📱 Responsive design for mobile and desktop
- ⚡ Fast and accurate diagnosis using a knowledge base

## Supported Crops & Issues

- **Tomato**: Yellow leaves, leaf curling, wilted plants, spots on leaves
- **Rice**: Brown spots, wilting, yellowing leaves, stunted growth
- **Wheat**: Yellow rust, black powdery heads, yellowing leaves
- **Maize**: Leaf blight, stem borer damage, yellowing leaves
- **Potato**: Leaf curling, dark lesions, yellowing leaves
- **Cotton**: Boll rot, leaf reddening, squares falling

## Installation

1. **Clone or download the project files**

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install basic dependencies**:
   ```bash
   pip install flask googletrans==4.0.0rc1 werkzeug
   ```

5. **Optional: Install ML dependencies for image classification**:
   ```bash
   # Only install if you need image classification features
   pip install tensorflow opencv-python numpy pillow
   ```

6. **Ensure you have the required files**:
   - `app.py` (Flask application)
   - `knowledge_base.json` (Crop disease database)
   - `templates/` folder (Web interface)
   - `model.h5` (Optional: AI model for image classification)

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **For image classification features (optional)**:
   ```bash
   set ENABLE_ML=true && python app.py
   ```

3. **Open your web browser** and go to:
   ```
   http://127.0.0.1:5000
   ```

4. **Login credentials**:
   - **Farmer account**: username: `farmer`, password: `farmer123`
   - **Admin account**: username: `admin`, password: `admin123`

5. **Interact with AgroBot**:
   - Enter the crop type (e.g., "Tomato")
   - Describe the symptoms (e.g., "yellow leaves with brown edges")
   - Click the send button or press Enter
   - Receive instant diagnosis and recommendations

## API Usage

AgroBot also provides a REST API endpoint for programmatic access:

```bash
POST /diagnose
Content-Type: application/json

{
  "crop": "Tomato",
  "symptom": "yellow leaves with brown edges"
}
```

Response:
```json
{
  "status": "success",
  "result": {
    "crop": "Tomato",
    "symptom": "yellow leaves starting from bottom with brown edges",
    "possible_issue": "Nitrogen deficiency",
    "severity": "Medium",
    "season": "All seasons, more common in rainy season",
    "prevention": "Ensure proper soil fertility with compost, rotate crops with legumes...",
    "treatment": "Apply urea (100-150kg/ha) or organic compost to restore nitrogen",
    "organic_treatment": "Apply compost tea or fish emulsion, use legume cover crops"
  }
}
```

## Project Structure

```
AgroBot/
├── app.py                 # Flask application
├── knowledge_base.json    # Crop disease knowledge base
├── venv/                  # Virtual environment (created)
├── templates/
│   └── index.html         # Web interface
└── README.md              # This file
```

## Knowledge Base Format

The `knowledge_base.json` file contains entries in the following format:

```json
{
  "crop": "Tomato",
  "symptom": "yellow leaves starting from bottom with brown edges",
  "possible_issue": "Nitrogen deficiency",
  "severity": "Medium",
  "season": "All seasons, more common in rainy season",
  "prevention": "Ensure proper soil fertility with compost...",
  "treatment": "Apply urea (100-150kg/ha) or organic compost...",
  "organic_treatment": "Apply compost tea or fish emulsion..."
}
```

## Customization

### Adding New Crop Information

1. Edit the `knowledge_base.json` file
2. Add new entries following the existing format
3. Restart the Flask application

### Modifying the Interface

Edit `templates/index.html` to:
- Change colors and styling
- Add new UI elements
- Modify the chatbot behavior

## Troubleshooting

### Common Issues

1. **Application won't start**:
   - Ensure Flask is installed: `pip install flask`
   - Check Python version: `python --version` (requires Python 3.6+)

2. **No diagnoses found**:
   - Verify `knowledge_base.json` exists in the same directory
   - Check the JSON format is valid

3. **Import errors**:
   - Make sure all dependencies are installed
   - Try reactivating the virtual environment

### Getting Help

If you encounter issues:
1. Check that all files are in the correct location
2. Verify the knowledge_base.json format is valid JSON
3. Ensure you're using a supported Python version

## Contributing

To contribute to AgroBot:
1. Add new crop disease information to `knowledge_base.json`
2. Follow the existing JSON format
3. Include detailed prevention and treatment information
4. Test your changes before submitting

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Farmers and agricultural experts who provided knowledge
- Flask framework for web application support
- Font Awesome for icons

## Support

For questions or support, please check:
1. The knowledge base file for existing information
2. Flask documentation for technical issues
3. Agricultural extension services for farming advice

---

**Happy Farming!** 🌾👨‍🌾👩‍🌾