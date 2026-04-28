# olympics classification project
Group:
Logan Hunter, u1342460
Quang Nguyen, u1470229
Robbie DeMars, u1411323

### Setup
download github repository from: 

# Inside olympics-classification-project:

requirements.txt has the requirements, if you want to install the environment another way, otherwise, using venv do the following:

python -m venv venv

# Windows:
.\venv\Scripts\activate 

# macOS / Linux:          # Im not sure this is correct because I am on Windows
source venv/bin/activate 

# To install the environment:
pip install -r requirements.txt

### Running the Model
cd '.\Classification Model\'
python olympics_classification.py <dataset>

# <dataset> can be the following:
final - All of our datasets combined
small_swim - Kaggle swimming dataset
small_track - Kaggle track dataset
small_weight_height - Kaggle weight height dataset
scraped - Our scraped dataset

# An example of how to run one of these would be:
python olympics_classification.py final

or 

python olympics_classification.py small_weight_height


# Entity Resolution Pipeline (Linux only)

The ER pipeline merges and cleans the scraped Olympic CSVs.
It requires Linux, Java 11, and Python 3.11.

1. Install packages (Java11, Python 3.11, venv)
   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa -y
   sudo apt update
   sudo apt install python3.11 python3.11-venv python3.11-dev openjdk-11-jdk -y

2. Create and activate virtual env
   ```
   python3.11 -m venv venv
   source venv/bin/activate

3. Install dependencies
   ```
   pip install -r requirements.txt

4. Run the ER pipeline
   ```
   python ER_pipeline.py
   
5. Outputs will appear in following folders:
   ```
   ./clean_data
   ./canonical
   ./readable_results.csv
   
