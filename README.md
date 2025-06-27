# OBE-Result-Analysis
Outcome-Based Education Result Analysis Using a Fine-Tuned TinyLLaMA Model

# 📊 OBE Result Analysis Tool

A web-based tool powered by Gradio + Python to analyze ESA (Exam Score Assessment) data for Outcome-Based Education (OBE). It provides insightful visualizations and AI-generated insights for educators to improve question design and student performance.

## 🚀 Features

- ✅ **Automatic Excel Parsing** of ESA result files  
- ❌ **Analysis of Non-Attempted Questions** (with Bloom Level, CO, PI)  
- 🏆 **Top 5 Questions by % Attainment**  
- 📊 **Strongest vs Weakest Question Analysis**  
- 📈 **Attempt vs Score Analysis**  
- 🤖 **Insights Generated using LLM (Simulated)**  
- 🎨 **Modern, Responsive UI with Custom Styling**
## 📁 Input Format

Upload an Excel file (`.xlsx`) with the following structure:

- **First 5 rows**: Metadata  
  - Row 1: Question names  
  - Row 2: PI (Performance Indicator)  
  - Row 3: Bloom Level (L1, L2, etc.)  
  - Row 4: CO (Course Outcome)  
  - Row 5: Max Marks  

- **From row 6 onwards**: Student scores  
  - Column 1: Student name or ID  
  - Remaining columns: Marks for each question  

---

## 💡 Example Use Case

An academic department uploads ESA results. This tool highlights:

- ❌ Questions students avoid  
- ✅ Questions students perform best/worst on  
- 🧠 Intelligent feedback for instructors  

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy, Matplotlib, Seaborn  
- Gradio for the Web UI  
- Simulated LLM for insights  

---

## 🌐 Live Demo

👉 [Hugging Face Space](https://huggingface.co/spaces/sarvesh1818/OBE_Result_Analysis)
- Download the Sample.xls file
- Open the Application and Drop it

---

## 🧠 Future Scope

- Integration with real LLMs (e.g., GPT-4)  
- Exportable reports (PDF)  
- CO-PO attainment mapping  
- Multi-course support  

---

## 📃 License

**MIT License** © 2025 KLE Technological University – Dept. of CSE  
Created by **Sarvesh Sanikop**
