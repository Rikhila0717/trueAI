1. Install conda and create a conda environment. 

[//]: # (install conda)
brew install --cask anaconda
brew install --cask miniconda

[//]: # (create a new project with conda env and activate conda)
conda create -n trueAI python=3.10 -y
conda activate trueAI
[//]: # (python3.12+ proved to be incompatible with a lot of LLM dependencies)

2. Install the required dependencies from requirements.txt and spaCy English model
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
3. 
