

---
# INVIGIL AI 
# AI Proctoring System: "AI Guardian for Uncompromised Online Exams" 

---
## About Me
Hi, I’m Kashish Shah—the developer behind this project. I’m passionate about AI and Machine Learning, and I enjoy working on projects that solve real-world problems and improve processes.

With this AI Proctoring System, I aimed to create a solution that ensures fairness and integrity in online exams. By using computer vision and real-time detection, the system automatically monitors for suspicious activity, preventing cheating while maintaining a smooth exam experience.

I believe in building efficient, scalable systems that have a positive impact. I'm always learning and looking for new opportunities to grow in the tech space. If you have feedback or would like to collaborate, feel free to reach out!





## Introduction

Welcome to the **AI Proctoring System**, the ultimate digital invigilator designed to ensure **uncompromised online exam integrity**. In an era of growing online learning and certification programs, maintaining exam security is critical. Our AI-powered system revolutionizes online assessments by using **real-time surveillance** and **intelligent detection** to monitor exam environments and instantly flag any attempts at cheating. With **computer vision**, **deep learning**, and **real-time object detection**, AI Proctor ensures that cheating never goes unnoticed. 

This autonomous proctoring tool is designed to not only identify suspicious activities but also **automatically warn** and even **terminate** the exam session when necessary. 

## 🔥 Why It’s a Game-Changer?

AI Proctor isn’t just another tool for online exam monitoring—it’s a complete solution that ensures **academic credibility** while **saving time and costs**. Here's what sets it apart:

- **Zero Compromise on Integrity**: Detects eye deviations, head movements, and unauthorized objects (phones, books, multiple faces).
- **Real-Time Surveillance**: The system monitors the exam environment in real-time with **computer vision** and **deep learning** techniques.
- **Autonomous Operation**: The system automatically takes action, issuing warnings and even terminating the exam if violations are detected.
- **Cost-Effective**: No need for human proctors, reducing manual supervision costs while maintaining high security.
  
## ⚙️ High-Level Technology

The AI Proctoring System leverages cutting-edge technologies and frameworks to provide seamless and accurate performance:

- **Computer Vision & Deep Learning**: 
   - OpenCV
   - Mediapipe
   - YOLOv3
   - Dlib

- **Real-Time Facial & Eye Tracking**: Detects unusual behavior such as suspicious eye movements or head turns using **real-time facial tracking**.
  
- **Intelligent Malpractice Detection**: Classifies objects (like phones, books, and Multiple faces) in real-time, using advanced AI models (e.g., **Ultralytics** for YOLOv3).

- **Automated Warning & Termination System**: The system can automatically trigger a **warning** when a violation occurs and **terminate the exam** if the violations are repeated, ensuring fair assessment.

- **Backend**: The system is powered by a **Flask backend**, enabling fast and scalable API calls for real-time processing.

- **Frontend**: Built with **HTML**, **CSS**, and **JavaScript** for an intuitive and responsive interface.

## 🚀 Features

### 1. **Real-Time Surveillance**
   - Monitors the candidate's eye and head movements to detect any suspicious behavior.
   - Tracks facial landmarks and identifies if the eyes wander or if there’s any head-turning.

### 2. **Malpractice Detection**
   - Recognizes unauthorized objects such as **phones**, **books**, and other materials.
   - Detects multiple faces in the frame, ensuring that only the registered candidate is visible.

### 3. **Warning & Termination Mechanism**
   - Sends automated warnings upon detection of malpractice.
   - After multiple violations, the system terminates the exam session, preventing further cheating attempts.

### 4. **Scalability**
   - AI Proctor is designed to scale across different platforms like universities, online certification programs, and corporate assessments.

### 5. **User-Friendly Interface**
   - Simple and intuitive frontend for both proctors and candidates to interact with the system.

## 📈 Business Benefits

- **Reduces Exam Malpractice**: Ensures that students are tested fairly without any unfair advantage.
- **Cost-Effective**: No need for manual proctors, reducing overhead costs.
- **Scalable Solution**: Ideal for online courses, universities, and certification programs that need a secure proctoring solution.
- **Fast and Efficient**: Automated monitoring leads to faster exam processes and enhanced security.

## 🔮 Future Innovations

We are continuously improving the AI Proctoring System. Future updates may include:

- **Identity Verification with Facial Recognition**: Enhances the security by verifying the candidate’s identity before starting the exam.
- **Group Proctoring**: Allows proctoring of group exams or multiple candidates in a single session.
- **Voice-based Anomaly Detection**: Detects anomalies such as whispering or speaking during the exam using voice detection algorithms.

## 🛠️ Installation & Setup

To use this AI Proctoring System locally, follow these steps:

### Prerequisites

Ensure you have the following installed on your machine:
- Python 3.7 or above
- `pip` for installing dependencies
- Docker (for containerization and deployment)
- OpenCV, Mediapipe, YOLOv3, Dlib, Flask

### Steps to Set Up

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/ai-proctoring-system.git
   cd ai-proctoring-system
   ```

2. **Install Dependencies**:

   Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   venv\Scripts\activate     # For Windows

   pip install -r requirements.txt
   ```

3. **Run the Application**:

   Run the system using Flask or Docker.

   - **Using Flask**:
     ```bash
     python app.py
     ```
   
   - **Using Docker**:
     Build and run the Docker container:
     ```bash
     docker build -t ai-proctor .
     docker run -p 5000:5000 ai-proctor
     ```

4. **Access the System**:
   Open a browser and go to `http://localhost:5000` to access the application.

## 📊 How It Works

The system operates in the following steps:

1. **Candidate Setup**: The candidate prepares their exam environment and starts the system.
2. **Real-Time Monitoring**: The system continuously tracks the candidate’s eye movements, head positions, and objects in the environment.
3. **Violation Detection**: If the system detects a violation (like eye movement, head turning, or unauthorized objects), it immediately issues a warning.
4. **Terminating the Exam**: After repeated violations, the system terminates the session, ensuring that the exam is fair and secure.

## 🔒 Privacy & Security

We prioritize the privacy and security of all participants. The system does not store any sensitive information (e.g., video feeds or personal data) beyond the exam session. All detected violations are logged for auditing purposes.

## 🤝 Contributing

We welcome contributions from developers, researchers, and security enthusiasts. If you'd like to contribute to the AI Proctoring System, feel free to fork the repository and submit a pull request.

## 💬 Support

For any issues, bugs, or pull requests, please open an issue on GitHub or contact me via email.
kashishshah.work@gmail.com


