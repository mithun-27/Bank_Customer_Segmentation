

---


# 🏦 Bank Customer Segmentation using K-Means (Data Science Project)

This project performs customer segmentation for a bank using **K-Means Clustering**.  
It identifies different groups of customers based on their financial behaviour, account balance, spending patterns, and transaction activity.

This helps banks with:

- Targeted Marketing  
- Personalized Banking Services  
- Credit Risk Scoring  
- Customer Retention Strategy  

---

## 🚀 Tech Stack

- **Programming:** Python  
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Plotly  
- **Dashboard:** Streamlit  
- **Dataset Source:** Kaggle  

---

## 📂 Project Structure

```
bank_customer_segmentation/
│
├─ data/                     → Dataset (CSV from Kaggle)
├─ models/                   → (Optional) Model storage
├─ reports/                  → Generated charts & results
│   ├─ clusters_pca.png
│   ├─ centroids.csv
│   ├─ data_clustered.csv
│   ├─ elbow_sse.png
│   ├─ silhouette_scores.png
│   ├─ k_selection.csv
│   ├─ SUMMARY.md
│
├─ src/
│   ├─ **init**.py
│   ├─ config.py
│   ├─ data_utils.py
│   ├─ kmeans_segmentation.py
│   ├─ visualize.py
│   ├─ app_streamlit.py
│
├─ run_pipeline.py
├─ requirements.txt
└─ README.md
```


---

## 📊 Output Screenshots

### ✅ Streamlit Dashboard – Home View

<img width="1919" alt="Dashboard" src="https://github.com/user-attachments/assets/006171f9-2c6d-40db-ab70-0c52d77002e6" />


### 📍 Clustered Data Preview

<img width="1919" alt="Clustered Data" src="https://github.com/user-attachments/assets/6f9a899a-d71b-44f7-8a57-79171031f334" />


### 🎯 PCA Scatter Plot – Customer Segments (2D)

<img width="1919" alt="PCA Scatter Plot" src="https://github.com/user-attachments/assets/3a03d6cb-401e-4930-93e2-f2a3c399d2a4" />


### 📈 Elbow Method (SSE vs K)

<img width="1919" alt="Elbow Method" src="https://github.com/user-attachments/assets/493af14e-97c7-4989-bb41-854531e3f210" />


### 📉 Silhouette Score vs K

<img width="1919" alt="Silhouette Score" src="https://github.com/user-attachments/assets/ddf1ebbc-8b67-4992-8861-bca938801b4c" />

---

## 🧪 Run Locally (Windows)

### 1️⃣ Create Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
````

### 2️⃣ Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3️⃣ Add Kaggle Dataset

Download dataset from:
[https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation](https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation)

Place the CSV in:

```
bank_customer_segmentation/data/
```

### 4️⃣ Run Pipeline (Generates Clusters + Reports)

```powershell
python run_pipeline.py
```

### 5️⃣ Run Streamlit Dashboard

```powershell
streamlit run src/app_streamlit.py
```

---

## 🧠 Key Insights from the Project

* Optimal K identified using **Elbow Method + Silhouette Score**
* **PCA** used to reduce dimensions for 2D cluster visualization
* **Cluster centroids** help understand customer behaviour patterns
* Enables **customer profiling** for bank strategies

---

## 📌 Future Enhancements

* Add **Gaussian Mixture Models (GMM)** for soft clustering
* Generate **Automated Customer Segment Insights Report**
* Deploy Streamlit dashboard online
* Add **API endpoint** for real-time segmentation

---

## 🧑‍💻 Author

**Mithun S**
GitHub: [https://github.com/mithun-27](https://github.com/mithun-27)

LinkedIn: [https://www.linkedin.com/in/mithun-s-732939280](https://www.linkedin.com/in/mithun-s-732939280)

---
