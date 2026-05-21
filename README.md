# 🚗 Car Price Predictor

This repository contains our **Machine Learning course group project** for predicting used car selling prices using machine learning.

The project focuses on building a complete machine learning workflow, including data preprocessing, model training, model evaluation, final model selection, and a user-friendly web interface using Streamlit.

---

## 📌 Project Overview

The **Car Price Prediction System** predicts the estimated selling price of a used car based on important vehicle details such as:

- Manufacturing year
- Kilometers driven
- Fuel type
- Seller type
- Transmission type
- Owner type
- Mileage
- Engine capacity
- Max power
- Number of seats
- Car brand

The final system allows users to enter car details through a web interface and receive an estimated car price prediction.

---

## 👥 Group Model Development

As part of the group project, each group member trained and tested different machine learning models for car price prediction.

The purpose of this approach was to compare different regression models and identify the most suitable model for the final system.

The tested models included:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- CatBoost Regressor

After comparing the performance of the trained models, the **Random Forest Regressor** was selected as the final model for the real system.

---

## ✅ Final Model Used

The final model used in the real system is:

```text
Random Forest Regressor
