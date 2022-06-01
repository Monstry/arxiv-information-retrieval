## Environments setup

1. install graphviz from https://graphviz.org/download/

2. run `pip install -r requirements.txt` in the terminal, the python version is 3.7.13. If you are using conda, you can create a new environment using `conda env create -f conda_env.yaml`.

## How to run

STEP 1. run spider to get papers metadata, which will be stored in `papers/` and `data/`.

STEP 2. run `preprocess.py` to tokenize summary, build inverted_index and build author index.

STEP 3. run `main.py` to start Flask server.