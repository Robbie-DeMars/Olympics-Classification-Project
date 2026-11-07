# Olympics Classification Project

Predicts Olympic medal outcomes (gold / silver / bronze / no medal) from athlete
features. Final project for Managing Data for and with ML, University of Utah.

- **Data:** merged Kaggle swim, track, and height/weight datasets, Maddison Project GDP data,
  and scraped Wikipedia data, unified by an entity-resolution pipeline (PySpark, GraphFrames).
- **Model:** XGBoost with class weighting; median imputation outperformed KNN.
- **Robbie's contribution:** classification model and dataset-completeness evaluation.

**Group:**

Logan Hunter

Quang Nguyen

Robbie DeMars

---

### Setup
1. **Clone the repository and enter the project folder:**
   ```bash
   git clone https://github.com/Robbie-DeMars/Olympics-Classification-Project.git
   cd Olympics-Classification-Project/olympics-classification-project
   ```
   The `clean_data` CSVs are stored with Git LFS, so install Git LFS before cloning.

2. **Create and activate virtual environment**

      2.1 Create virtual environment
      ```bash
      python -m venv venv
      ```
      2.2 Activate virtual environment based on OS
      
      **(Windows:)** ```.\venv\Scripts\activate```
      
      **(Linux / macOS:)** ```source venv/bin/activate```

3. **Install the environment:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Model
   ```bash
   cd 'Classification Model'
   python olympics_classification.py <dataset>
   ```

### \<dataset> can be the following:
```
final - All of our datasets combined
small_swim - Kaggle swimming dataset
small_track - Kaggle track dataset
small_weight_height - Kaggle weight height dataset
scraped - Our scraped dataset
```

### Examples of how to run one of these would be:
```bash
python olympics_classification.py final
```
or 
```bash
python olympics_classification.py small_weight_height
```

# Entity Resolution Pipeline (Linux only)

The ER pipeline merges and cleans the scraped Olympic CSVs.
It requires Linux, Java 11, and Python 3.11.

1. Install packages (Java11, Python 3.11, venv)
   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa -y
   sudo apt update
   sudo apt install python3.11 python3.11-venv python3.11-dev openjdk-11-jdk -y

2. Create and activate virtual env
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt

4. Run the ER pipeline
   ```bash
   python ER_pipeline.py
   
5. Outputs will appear in following folders:
   ```bash
   ./clean_data
   ./canonical
   
