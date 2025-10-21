# 🚦 Smart Traffic Control System using YOLO and Canny Edge Detection

## 📘 **Overview**
The **Density-Based Smart Traffic Control System** is an AI-powered web application that automatically controls traffic light duration based on real-time vehicle density.  
It uses **YOLO (You Only Look Once)** for vehicle detection and **Canny Edge Detection** for density estimation.  
The system is built using **Streamlit** for the user interface and **SQLite3** for local data storage.

---

## 🎯 **Key Features**
- 🔹 **Dynamic Signal Control:** Allocates green light duration based on vehicle density.  
- 🔹 **AI-Based Detection:** Uses YOLOv8 for real-time vehicle identification.  
- 🔹 **Edge Detection:** Uses Canny Edge Detection to analyze vehicle density.  
- 🔹 **Live Webcam Support:** Stream live camera feed using Streamlit-WebRTC.  
- 🔹 **User Authentication:** Secure login and signup system with bcrypt encryption.  
- 🔹 **Data Storage:** Stores analysis history in an SQLite database.  
- 🔹 **Analytics Dashboard:** Displays interactive charts and downloadable reports using Altair and Pandas.  
- 🔹 **Multi-Input Support:** Works with uploaded images, video files, or live webcam feed.  

---

## 🧠 **Technology Stack**
| Category | Tool / Library |
|-----------|----------------|
| Programming Language | Python |
| Frontend Framework | Streamlit |
| Image Processing | OpenCV, Pillow |
| Object Detection | YOLOv8 (Ultralytics) |
| Edge Detection | Canny Algorithm |
| Database | SQLite3 |
| Authentication | bcrypt |
| Visualization | Altair, Pandas |
| Live Streaming | Streamlit-WebRTC |
| Other Libraries | NumPy, Base64, io, tempfile, av |

---

## ⚙️ **Installation Guide**

### **1. Clone or Download the Project**
```bash
git clone https://github.com/yourusername/smart-traffic-control.git
cd smart-traffic-control
```

### **2. Create a Virtual Environment**
```bash
python -m venv venv
```

### **3. Activate the Virtual Environment**
- **Windows**
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux**
  ```bash
  source venv/bin/activate
  ```

### **4. Install Required Libraries**
```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` file yet, create one with:
```txt
streamlit
opencv-python
numpy
pandas
altair
pillow
bcrypt
sqlite3-binary
ultralytics
streamlit-webrtc
av
```

---

## ▶️ **How to Run**
After installation, start the Streamlit app:
```bash
streamlit run "Main.py"
```

If Streamlit is not recognized, run it as:
```bash
python -m streamlit run "Main.py"
```

Then open the link shown in the terminal (usually http://localhost:8501).

---

## 🧭 **Usage Instructions**
1. **Login or Sign Up** using your credentials.  
2. Upload an **empty road image** as a reference.  
3. Choose one of the following input modes:
   - 📸 Upload Image  
   - 🎥 Upload Video  
   - 🔴 Live Webcam Feed  
4. The system will automatically:
   - Detect vehicles using YOLO.  
   - Calculate density via Canny Edge Detection.  
   - Suggest green signal duration.  
5. View and download your results in the **Analytics Dashboard**.

---

## 📊 **Outputs**
- Vehicle count per class (car, bus, bike, truck).  
- Calculated density percentage.  
- Allocated green time.  
- Real-time charts for analysis history.  
- Downloadable Excel reports.

---

## 🧩 **Project Structure**
```
├── Main.py
├── CannyEdgeDetector.py
├── traffic_data.db
├── background.png
├── traffic.png
├── README.md
└── requirements.txt
```

---

## 👥 **Contributors**
- **Kistipati Shiva Prasad Reddy (23EG510A03)**  
- **Mohammad Saniya (22EG110A42)**  
- **Challa Adithya (22EG110A13)**  
**Under the guidance of:** Mr. M. Manohar, Assistant Professor, Dept. of Data Science, Anurag University

---

## 📚 **Future Enhancements**
- Integration with IoT-based traffic sensors.  
- Centralized database for multi-junction control.  
- Mobile app version for field monitoring.  
- Real-time cloud dashboard with predictive analytics.

---

## 🏁 **License**
This project is developed for **academic purposes** under the Department of Data Science, Anurag University.
