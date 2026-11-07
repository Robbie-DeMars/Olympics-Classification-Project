# Olympics Classification Project
**Group:**

Logan Hunter, u1342460

Quang Nguyen, u1470229

Robbie DeMars, u1411323

---

### Setup
1. **Clone Repository From:**
   
   (https version of clone):
   ```bash
   git clone https://github.com/u1411323/olympics-classification-project
   ```
   (ssh version of clone):
   ```bash
   git@github.com:u1411323/olympics-classification-project.git
   ```

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
   
