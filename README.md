# Clinical Trial Keyword Recommendation System

A collaborative filtering system for recommending Medical Subject Headings (MeSH) terms in clinical trial documents using machine learning.

## 🎯 Overview

This project implements a collaborative filtering algorithm to recommend relevant keywords for clinical trial documents. The system processes clinical trial data from ClinicalTrials.gov and uses MeSH terms to provide intelligent keyword recommendations for researchers.

## 📊 Results

- **Mean Squared Error (MSE)**: 0.0147
- **Average Precision at K (AP@K)**: 0.4607
- **Dataset**: 55,634 clinical trial documents
- **Features**: 134 MeSH terms
- **Performance**: No overfitting detected

## 🏗️ System Architecture

### Data Engineering Module
- XML parsing from ClinicalTrials.gov
- MeSH term extraction and standardization
- TF-IDF utility matrix construction

### Machine Learning Module
- Collaborative filtering algorithm
- ReLU-based latent factor model
- Mini-batch gradient descent optimization
- L2 regularization

## 🚀 Getting Started

### Prerequisites
- Required packages (see requirements.txt)

### Installation
```bash
# Clone the repository
git clone https://github.com/2005-Aneeshdutt/Clinical-Trial-Keyword-Recommendation.git

# Navigate to project directory
cd Clinical-Trial-Keyword-Recommendation

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# Set Python path
export PYTHONPATH=$(pwd)

# Run complete pipeline
python tools/xml_to_bin.py
python scripts/build_mesh_counts.py
python scripts/split_data.py
python scripts/search_alpha.py
python scripts/final_training.py
python scripts/test_final_model.py

# Generate recommendations
python scripts/final_recommendations.py
```

## 📁 Project Structure
Clinical-Trial-Keyword-Recommendation/
├── clinical_trials/          # Clinical trial data processing
├── mesh/                     # MeSH term handling
├── ml/                       # Machine learning models
├── scripts/                  # Main execution scripts
├── analysis/                 # Results analysis
├── output/                   # Generated models and data
├── results/                  # Analysis results and plots
└── tools/                    # Utility scripts


🔬 Methodology
Data Processing: Extract MeSH terms from clinical trial XML documents
Feature Engineering: Create TF-IDF utility matrix
Model Training: Train collaborative filtering model with ReLU activation
Evaluation: Test using MSE and AP@K metrics
Recommendation: Generate keyword suggestions for new documents


📈 Performance
The model demonstrates excellent performance with consistent results across training, validation, and test sets:
Training MSE: 0.0147
Validation MSE: 0.0148
Test MSE: 0.0147
AP@K: 0.4602
🎓 Authors
N Aneesh Dutt - GitHub
Prachi Jha

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
ClinicalTrials.gov for providing the dataset
Medical Subject Headings (MeSH) thesaurus
Python scientific computing community
