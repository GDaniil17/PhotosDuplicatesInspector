# PhotosDuplicatesInspector

This project clusters images based on similarity using a **pretrained SOTA model (SigLIP 2)** and provides a **web-based UI** for visualization, selection, and export. 

## Features
- **Clustering of images** based on a similarity threshold.
- **User interface** via a **Flask web app** for visualization.
- **Supports multiple models**, with **local model loading** to avoid re-downloading.
- **Customizable file extensions** for supported image types.
- **Exporting selected images** to a separate folder.
- **Estimation of processing time remaining**.
- **Handling of missing images**, including detailed error reports.
- **Modern UI design** with real-time selection updates.

---

## Installation

### 1. **Create a Virtual Environment**
It is recommended to create a virtual environment to isolate dependencies.  
Run the following commands:

```bash
python -m venv venv  # Create a virtual environment
source venv/bin/activate  # Activate on macOS/Linux
venv\Scripts\activate  # Activate on Windows
pip install -r requirements.txt
python photos_inspector.py
python3 photos_inspector.py # for macOS
```

<img width="750" alt="Снимок экрана 2025-02-23 в 23 06 12" src="https://github.com/user-attachments/assets/cb7ec0d3-bf54-4945-96cd-a4e21c514528" />

<img width="1512" alt="Снимок экрана 2025-02-23 в 23 07 54" src="https://github.com/user-attachments/assets/43b2af1d-953e-4398-99ef-667a85f38051" />

<img width="1512" alt="Снимок экрана 2025-02-23 в 23 10 12" src="https://github.com/user-attachments/assets/5c26265a-66a3-46da-b79e-30fb4403c42b" />
