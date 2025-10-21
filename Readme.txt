Smart Traffic Control System using Computer Vision
This project is a Python-based web application that uses real-time computer vision to analyze traffic density and dynamically adjust traffic light timings. It features a secure user authentication system, an analytics dashboard, and multiple modes for traffic data input.
✨ Features
* Secure User Authentication: Users can sign up and log in. All credentials are saved securely in a local SQLite database.
* Multiple Input Modes: Analyze traffic from three different sources:
   * 📷 Static Image Upload
   * 🎥 Pre-recorded Video Upload
   * 💻 Live Webcam Feed
* Intelligent Vehicle Isolation: Uses the YOLOv8 object detection model to accurately identify and isolate only vehicles (cars, trucks, buses, motorcycles) from the background.
* Dynamic Density Calculation: Applies the Canny Edge Detection algorithm to the isolated vehicles to generate a quantitative measure of traffic density.
* Adaptive Signal Timing: Automatically allocates green light duration based on the calculated real-time traffic density.
* Analytics Dashboard: Visualizes historical data to identify peak traffic hours and track user activity.
* Excel Report Generation: Allows logged-in users to download their personal analysis history as a .xlsx file.
🛠️ Technology Stack
* Language: Python 3.10+
* Web Framework: Streamlit
* Computer Vision: OpenCV, Ultralytics YOLOv8
* Database: SQLite
* Data Handling: Pandas, Altair
* Utilities: bcrypt (for password hashing), openpyxl (for Excel export)
🚀 Setup and Installation
Follow these steps to get the application running on your local machine.
1. Prerequisites
* Python 3.10 or later installed.
* pip (Python package installer).
2. Clone the Repository
(If applicable, otherwise just place all files in one folder)
git clone [https://github.com/your-username/smart-traffic-control.git](https://github.com/your-username/smart-traffic-control.git)
cd smart-traffic-control

3. Install Dependencies
It is recommended to use a virtual environment.
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install all required packages from requirements.txt
pip install -r requirements.txt

Note: A requirements.txt file can be created by running pip freeze > requirements.txt. For this project, it would contain:
streamlit
opencv-python-headless
ultralytics
pandas
altair
bcrypt
openpyxl
av

4. File Structure
Ensure the following files are in your main project directory (E:\SOURCE CODE\):
* app.py (The main application script)
* CannyEdgeDetector.py (Your custom Canny algorithm implementation)
* traffic.png (The image for the dashboard header)
* background.png (The background image for the login page)
5. Run the Application
Open your terminal in the project directory and run the following command:
streamlit run app.py

The application should automatically open in your web browser at http://localhost:8501.
📖 How to Use
1. Sign Up: On your first visit, navigate to the "Sign Up" page from the sidebar to create a new user account.
2. Login: Use your credentials to log in. You will be redirected to the main dashboard.
3. Configure Parameters: In the sidebar, you can adjust the Canny algorithm parameters and set the "Reference Image" (either by uploading a picture of an empty road or using the demo box).
4. Analyze Traffic:
   * In the "Traffic Analysis Tool" tab, choose your input method (Image, Video, or Webcam).
   * Provide the input, and the system will perform the analysis and display the results.
5. View Analytics:
   * Navigate to the "Analytics Dashboard" tab to see historical charts on peak hours and user contributions.
   * In the "My Analysis History" section, you can view your personal records and click the button to download them as an Excel file.