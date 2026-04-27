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
venv\Scripts\Activate

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


