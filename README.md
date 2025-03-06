# 📄 Resume Ranking System 🚀

## 🔍 Overview

Finding the right candidate for a job can be time-consuming and challenging. The **Resume Ranking System** simplifies the process by evaluating resumes against a job description using **cutting-edge NLP techniques**. This system leverages **TF-IDF Vectorization, Semantic Similarity, and Skill Matching** to provide the best candidate recommendations.

## 🌟 Why Use This?

✔️ Saves time in manual resume screening\
✔️ Identifies the best-fit candidates instantly\
✔️ Provides a **fair and objective ranking system**\
✔️ Enhances recruitment efficiency with AI-powered analysis

## 🚀 Features

✅ **Upload multiple resumes (PDF format)**\
✅ **Extract text and preprocess it for ranking**\
✅ **Compare resumes using TF-IDF and Cosine Similarity**\
✅ **Use Sentence Transformers for deep Semantic Similarity**\
✅ **Extract skills and experience from resumes**\
✅ **Provide a ranked list of candidates based on job relevance**\
✅ **Sort resumes automatically, ensuring the best match is given a 100% score**


## 🛠️ Installation

### Prerequisites

Ensure you have **Python 3.8+** installed.ad

### **Step 1: Clone the Repository**  
```sh
git clone https://github.com/manmohansethi/Resume-Ranking-System.git
cd Resume-Ranking-System
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Project

Launch the Streamlit app with:

```sh
streamlit run app.py
```

## 🖥️ Usage

1. Open the web interface.
2. Upload resumes in **PDF format**.
3. Paste the **Job Description** in the text area.
4. Click **Process** to analyze and rank resumes.
5. View the **ranked resumes** based on similarity and relevance.

## 📜 How It Works

1. **Text Extraction:** Extracts text from uploaded resumes.
2. **Preprocessing:** Cleans text using tokenization, stemming, and stop-word removal.
3. **TF-IDF Similarity:** Measures keyword similarity between job description and resumes.
4. **Semantic Similarity:** Uses **Sentence Transformers** for deep context-based matching.
5. **Skill & Experience Matching:** Extracts skills and experience duration from resumes.
6. **Final Ranking:** Combines multiple metrics to rank resumes fairly and efficiently.
7. **Sorting:** Ensures the highest-ranked resume is assigned **100%**, with other scores normalized accordingly.

## 📊 Scoring Breakdown

The ranking is calculated using a weighted combination of different factors:

- **40% TF-IDF Similarity**
- **30% Semantic Similarity (Sentence Transformers)**
- **20% Skill Matching Score**
- **10% Experience Matching Score**

Scores are normalized on a scale of **0-100%**, ensuring fair ranking, with the best resume always scoring **100%**.

## 🔗 Dependencies

- **Streamlit** (UI Framework)
- **PyPDF2** (Extracts text from PDFs)
- **NLTK** (Text Processing)
- **Scikit-learn** (TF-IDF and Similarity Metrics)
- **Sentence Transformers** (Semantic Matching)
- **RAKE-NLTK** (Keyword Extraction)

## 🚀 Future Improvements

🔹 **Enhance Named Entity Recognition (NER)** for better skill extraction\
🔹 **Add support for Word & Excel resumes**\
🔹 **Improve Experience Parsing with advanced NLP techniques**\
🔹 **Deploy the app on Cloud platforms for global access**

## 👨‍💻 Contributing

Contributions are welcome! If you'd like to enhance this project, feel free to fork the repo and submit a PR. For major changes, please open an issue first.

## 📜 License

This project is licensed under the **MIT License**.

📬 Have suggestions or feedback? Feel free to reach out! 🚀

## 🌟 **Contributing**  
Want to improve this project? Contributions are welcome! Feel free to submit a **pull request**.

---

## 📧 **Contact**  
📌 **GitHub:** [@manmohansethi](https://github.com/manmohansethi)  
📌 **Email:** manmohansethi143@gmail.com  
