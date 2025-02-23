# 📸 PhotosDuplicatesInspector

**PhotosDuplicatesInspector** clusters images based on similarity using a **pretrained SOTA model (SigLIP 2)** and provides a **web-based UI** for visualization, selection, and export. 🚀

---

## ✨ Features
✔️ **Clustering of images** based on a similarity threshold.  
✔️ **User interface** via a **Flask web app** for visualization.  
✔️ **Supports multiple models**, with **local model loading** to avoid re-downloading.  
✔️ **Customizable file extensions** for supported image types.  
✔️ **Exporting selected images** to a separate folder.  
✔️ **Estimation of processing time remaining**.  
✔️ **Handling of missing images**, including detailed error reports.  
✔️ **Modern UI design** with real-time selection updates.  
✔️ **Perfect for organizing family albums 📸📚** by identifying and grouping similar photos, helping to declutter and preserve memories.  

---

## 📥 Installation

### 🔹 Step 1: Create a Virtual Environment
It is **highly recommended** to use a virtual environment to isolate dependencies. Run the following commands based on your OS:

#### 💻 macOS / Linux
```bash
python3 -m venv venv  # Create a virtual environment
source venv/bin/activate  # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies
python3 photos_inspector.py  # Run the application
```

#### 🖥️ Windows (CMD / PowerShell)
```powershell
python -m venv venv  # Create a virtual environment
venv\Scripts\activate  # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies
python photos_inspector.py  # Run the application
```

---

## 🎨 UI Preview

### **Clustered Images View**
![Clustered Images](https://github.com/user-attachments/assets/cb7ec0d3-bf54-4945-96cd-a4e21c514528)

### **Selection Interface**
![Selection UI](https://github.com/user-attachments/assets/43b2af1d-953e-4398-99ef-667a85f38051)

### **Export Feature**
![Exporting Images](https://github.com/user-attachments/assets/5c26265a-66a3-46da-b79e-30fb4403c42b)

---

## 🚀 Running the Application
Once installed, activate your virtual environment and run:
```bash
python photos_inspector.py
```
Then open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 🛠️ Troubleshooting
- **Error: Module not found?** → Ensure dependencies are installed: `pip install -r requirements.txt`
- **Flask not found?** → Run: `pip install flask`
- **Port 5000 already in use?** → Use another port: `python photos_inspector.py --port 8080`

📌 *If you encounter any issues, feel free to open an issue in the repository!* 🚀

