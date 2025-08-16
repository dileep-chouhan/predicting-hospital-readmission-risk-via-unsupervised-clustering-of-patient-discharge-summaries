# Predicting Hospital Readmission Risk via Unsupervised Clustering of Patient Discharge Summaries

## Overview

This project aims to identify high-risk patient subgroups prone to hospital readmission using natural language processing (NLP) techniques and unsupervised clustering.  We analyze patient discharge summaries to extract relevant features and subsequently group patients into clusters based on their predicted readmission risk. This allows for the identification of specific patient characteristics associated with high readmission probability, enabling targeted interventions to improve patient outcomes and reduce healthcare costs. The analysis leverages unsupervised learning to discover hidden patterns in the data without prior knowledge of readmission status.


## Technologies Used

* Python 3.x
* Pandas
* Numpy
* Scikit-learn (for clustering algorithms)
* NLTK (for natural language processing)
* Matplotlib
* Seaborn


## How to Run

1. **Install Dependencies:**  Navigate to the project directory in your terminal and install the required Python libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Main Script:** Execute the main script using:

   ```bash
   python main.py
   ```

   Ensure that you have the necessary input data (discharge summaries) in the specified format and location as detailed within the code.


## Example Output

The script will print key analysis results to the console, including details about the clustering process (e.g., number of clusters, cluster characteristics).  Additionally, the script generates several visualization files (e.g., cluster distribution plots, etc.).  These plots are saved in the `./output` directory (this directory will be created if it doesn't exist) and provide a visual representation of the patient clusters and their characteristics.  The exact filenames and plot types may vary depending on the chosen clustering algorithm and visualization options within the code.