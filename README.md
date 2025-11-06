# 🛒 E-Commerce Product Recommendation System (Big Data + Spark MLlib)

An end-to-end **Big Data project** that builds a **personalized product recommender system** using **PySpark (ALS)**, powered by **real e-commerce behavior data** and visualized through **Power BI and Streamlit** dashboards.

---

## 🚀 Project Overview

This project demonstrates scalable **data engineering** and **machine learning** skills by processing millions of user–product interactions and generating personalized recommendations using **Spark MLlib’s Alternating Least Squares (ALS)** algorithm.

### 🔍 Key Objectives
- Build a recommendation system for e-commerce products based on **user behavior (view, cart, purchase)**.
- Leverage **Spark distributed computing** for large-scale data preprocessing and model training.
- Showcase analytical and visualization skills via **Streamlit** and **Power BI dashboards**.

---

## 🧠 Tech Stack

| Category | Tools / Technologies |
|-----------|----------------------|
| Big Data Processing | **Apache Spark (PySpark)** |
| Machine Learning | **Spark MLlib (ALS)** |
| Storage Format | **Parquet, CSV** |
| Visualization | **Streamlit**, **Power BI** |
| Programming | **Python 3.12+** |
| Dataset | [E-Commerce Behavior Data (Kaggle)](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) |

---

```
## ⚙️ Project Architecture

📦 recommender_system/
│
├── preprocess.py               # Clean and transform raw e-commerce events
├── train_als_implicit.py       # Train ALS model & generate top-N recommendations
├── dashboard.py                # Streamlit dashboard for user-wise recommendations
│
├── work/
│   ├── ratings_parquet/        # Preprocessed Spark Parquet data
│   ├── out/
│   │   ├── analytics_summary/    # Product-level analytics (counts, averages)
│   │   ├── top_recommendations_pretty/ # Joined ALS results + metadata
│   │   └── model_metrics.csv     # RMSE and other KPIs
│
├── README.md
└── requirements.txt
```
---


## 🧩 Features

✅ **End-to-End Big Data Pipeline**  
From raw CSV → Spark DataFrames → Model training → Dashboard-ready outputs.  

✅ **Implicit Feedback Modeling**  
User engagement (view/cart/purchase) mapped to weighted ratings for realistic recommendation logic.  

✅ **Interactive Dashboards**  
Streamlit UI and Power BI visualizations for model insights and business metrics.  

✅ **Scalable and Efficient**  
Uses Parquet, partitioning, and memory-tuned Spark configuration to handle millions of records.

---
