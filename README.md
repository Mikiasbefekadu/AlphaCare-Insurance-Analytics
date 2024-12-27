# AlphaCare Insurance Analytics Project

## Project Overview
This project aims to analyze insurance claims data for AlphaCare to identify risk patterns and implement effective data version control using DVC (Data Version Control). The project is divided into two main tasks:

1. **Exploratory Data Analysis (EDA) and Visualization:**
   - Analyze claims data to uncover patterns based on geography and vehicle types.
   - Generate visual insights for informed decision-making.

2. **Data Version Control:**
   - Use DVC to manage data versions, ensuring reproducibility and efficient collaboration.

## Features
- **EDA and Visualizations:**
  - Analyze claim ratios by province.
  - Explore total claim amounts by vehicle type.
- **Data Version Control:**
  - Track changes to datasets with DVC.
  - Ensure consistent data sharing and reproducibility.

## Technologies Used
- **Programming Language:** Python
- **Libraries:**
  - `pandas`, `matplotlib`, `seaborn` for data analysis and visualization.
  - `DVC` for data version control.
- **Version Control:** Git

## Setup Instructions

### Prerequisites
- Python (>=3.8)
- Git (>=2.0)
- DVC (>=2.0)

## Project Structure
```
.
├── data
│   ├── MachineLearningRating_v3.txt
│   ├── ...
├── analysis
│   ├── eda_visualizations.ipynb
│   ├── ...
├── .dvc
├── .gitignore
├── README.md
└── requirements.txt
└──scripts
    ├── __init__
    ├──eda_module.py
└──tests
    ├──dummy_test
```

## Key Visualizations

### 1. Risk Rating by Province
- Gauteng province shows the highest risk, with elevated claim ratios compared to others.

### 2. Claims by Vehicle Type
- Heavy commercial vehicles contribute the highest total claim amounts.

## Contact
For further queries or collaboration, please reach out to [**Mikias Befekadu**] at [zelekemikias@gmail.com].
