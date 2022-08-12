# NLP_barriers

Goal: use hugging face transformers to predict Ea 

## Installation

```
# Create a new env
conda create -n huggingface python==3.9.13 -y

# activate env
conda activate huggingface

# install pytorch gpu version
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y

# install other packages
conda install -c rdkit rdkit==2021.09.3 -y
conda install seaborn jupyter -y
conda install -c anaconda joblib ipython -y
conda install -c conda-forge tqdm  -y
conda install -c anaconda scikit-learn 

# optional to view training results
pip install wandb
```
