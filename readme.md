# Loan Approval Optimization Project

This repository contains the complete implementation for a machine learning project focused on optimizing loan approval decisions using the LendingClub Loan Data dataset. The project involves exploratory data analysis (EDA), a supervised deep learning classifier for default prediction, an offline reinforcement learning (RL) agent for policy optimization, and a comparative analysis. It is designed to demonstrate end-to-end ML skills for a fintech loan approval scenario.

## About the Project

The goal is to build models that maximize financial returns by deciding whether to approve or deny loans based on historical data. Key components:
- **EDA and Preprocessing**: Data cleaning, feature selection, and engineering.
- **Deep Learning Model**: A neural network for predicting loan default probability.
- **Offline RL Agent**: Discrete CQL algorithm to learn an approval/denial policy.
- **Analysis**: Comparison of models and future recommendations.

This project uses a sampled subset of the accepted loans dataset for efficiency, but can be scaled to the full dataset.

### Built With
- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, torch, d3rlpy

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)
- Download the dataset from [Kaggle: LendingClub Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (use `accepted_2007_to_2018Q4.csv.gz`)

### Installation
1. Clone the repository:

git clone https://github.com/sahilshukla3003/LoanApprovalOptimizations.git
cd loan-approval-optimization


2. Create and activate a virtual environment (recommended):

python -m venv env
source env/bin/activate # On Unix/Mac

or
env\Scripts\activate # On Windows


3. Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn torch d3rlpy

4. Place the downloaded dataset in the project root or update the file path in the code.

## Usage

This project is structured as Jupyter Notebook-style Python scripts. Run them in a Jupyter environment or as standalone scripts.

1. **EDA and Preprocessing**:
- Run the preprocessing script to load, clean, and prepare the data.

2. **Train Deep Learning Model**:
- Builds and evaluates a neural network classifier.

3. **Train Offline RL Agent**:
- Frames the problem as offline RL and trains a Discrete CQL agent.

4. **View Results and Report**:
- Metrics: DL AUC ~0.71, F1 ~0.06; RL Policy Value ~3.19.
- The final report is in `Report.pdf` .

For full reproducibility:
- Adjust sampling size in the code for larger datasets.
- Run all scripts sequentially to generate models and outputs.

## Roadmap
- Integrate rejected loans data for improved RL simulation.
- Add hyperparameter tuning scripts.
- Explore hybrid DL-RL models.

## License
Distributed under the MIT License. 

## Authors
- Sahil Shukla - sahilshukla959@gmail.com 

## Acknowledgements
- LendingClub Dataset from Kaggle
- d3rlpy for offline RL
- PyTorch for deep learning

For any issues, open a GitHub issue or contact via email.



