# What's That Smell? 👃
### AI-Based Gas Classification System Robust to Sensor Drift
(센서 노후화에 강건한 AI 기반 가스 분류 시스템)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Project_Complete-success)]()

[cite_start]**Team Name:** What's that smell? [cite: 56]
[cite_start]**Affiliation:** Inha University, Computer Engineering [cite: 51]
[cite_start]**Project Topic:** AI-Based Gas Classification System Robust to Sensor Drift [cite: 57]

---

## 📖 Project Overview (English)

[cite_start]Industrial environments such as battery manufacturing and petrochemical plants require continuous monitoring of hazardous gases[cite: 59]. However, gas sensors (MOX) suffer from **Sensor Drift**—a phenomenon where sensor responsiveness degrades over time due to aging. [cite_start]This causes conventional AI models trained on early data (Batch 1) to fail on future data (Batch 10)[cite: 59].

This project proposes a robust classification system that maintains high accuracy even as sensors age. [cite_start]We redefine sensor aging as a **Domain Shift** problem and solve it using **Domain Adversarial Neural Networks (DANN)** and **Physics-Based Feature Engineering**[cite: 59].

### Key Features
* [cite_start]**Drift Robustness:** Classifies 6 gas types (Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene) regardless of sensor age[cite: 63].
* [cite_start]**Sensor-Aware Feature Engineering:** Extracts physical features ($|DR|$, EMA Variance) to capture reaction kinetics independent of drift magnitude[cite: 59].
* [cite_start]**Domain Adaptation:** Utilizes DANN with a Gradient Reversal Layer (GRL) to learn time-invariant features, making the model insensitive to the "Batch ID" (sensor age)[cite: 39, 40].

---

## 📖 프로젝트 개요 (Korean)

[cite_start]산업 현장에서는 유해 가스(에탄올, 암모니아 등)를 지속적으로 모니터링해야 합니다[cite: 59]. [cite_start]하지만 가스 센서(MOX)는 시간이 지남에 따라 성능이 저하되는 **센서 노후화(Sensor Drift)** 현상이 발생하며, 이로 인해 초기에 학습된 AI 모델이 시간이 지난 후에는 작동하지 않는 문제가 발생합니다[cite: 14, 59].

본 프로젝트는 센서 교체 없이 AI 기술만으로 이 문제를 해결합니다. [cite_start]센서의 노후화를 단순한 오차가 아닌 **도메인 변화(Domain Shift)** 문제로 정의하고, **도메인 적대적 신경망(DANN)** 과 **물리적 특징 공학**을 결합하여 강건한 분류 시스템을 제안합니다[cite: 59].

### 핵심 기능
* [cite_start]**드리프트 강건성:** 센서가 노후화되어도 6종의 가스를 정확히 분류합니다[cite: 63].
* [cite_start]**센서 인식 특징 공학:** 반응의 크기($|DR|$)뿐만 아니라 반응 속도와 형태(EMA 분산)를 추출하여 노후화의 영향을 최소화합니다[cite: 27, 31].
* [cite_start]**도메인 적응(Domain Adaptation):** GRL(Gradient Reversal Layer)을 적용한 DANN 모델을 통해, 센서의 사용 시기(Batch)에 상관없는 공통 특징을 학습합니다[cite: 39].

---

## 🛠️ Methodology (적용 기술)

### 1. Feature Engineering (특징 공학)
Raw sensor data often contains "magnitude" noise caused by drift. [cite_start]We extract 8 features per sensor (128 total) to focus on the *shape* and *speed* of the chemical reaction[cite: 24, 25].
* [cite_start]**$|DR|$ (Absolute Delta Response):** Represents the reaction intensity[cite: 27].
* [cite_start]**EMA Variance:** Captures the trend and speed of the reaction curve[cite: 31].
* **Drift Indicator:** A ratio feature ($EMA / |DR|$) designed to remain stable despite sensor aging.

### 2. DANN (Domain Adversarial Neural Network)
[cite_start]We utilize a deep learning architecture that trains on two competing objectives[cite: 35, 36]:
1.  **Label Classifier:** Minimizes gas classification error (Make the model smart).
2.  **Domain Discriminator:** Maximizes the error in identifying the "Batch ID" (Make the model "forget" the sensor age).

[cite_start]This adversarial process forces the Feature Extractor to learn a latent space where data from **Batch 1 (New Sensor)** and **Batch 10 (Aged Sensor)** are indistinguishable.

---

## 📂 Repository Structure

```bash
├── data/
│   ├── batch1.dat ... batch10.dat   # Gas Sensor Drift Dataset (UCI/Kaggle)
│
├── models/
│   ├── dann_model.pth               # Trained DANN model weights
│
├── scripts/
│   ├── feature_engineering.py       # Basic feature extraction & Baseline model
│   ├── feature_engineering_grouping.py # Group-based specialized models
│   ├── benchmark_comparison.py      # Comparison of Baseline vs. Advanced methods
│   ├── train_dann.py                # Main DANN training loop (PyTorch)
│   ├── visualize.py                 # t-SNE visualization of raw data drift
│   └── visualize_latent.py          # t-SNE visualization of DANN latent space
│
├── README.md                        # Project Documentation
└── requirements.txt                 # Dependencies


📊 Results Summary
Baseline Model: Performance drops significantly on Batch 10 due to drift. (기존 모델은 센서 노후화 시 성능 급락) 

Feature Engineering: Improves RMSE and stability on aged data. (특징 공학 적용 시 노후 데이터에서도 성능 개선)

DANN Model: Successfully aligns domains, achieving high classification accuracy on unseen batches. (DANN 적용 시 배치 간 차이 제거 및 높은 정확도 달성)

👥 Team (팀원)- Computer Science

Jinwoo Jo (조진우) -팀장(Leader)

Taeyong Lee (이태용) 

Dongchul Ahn (안동철)
