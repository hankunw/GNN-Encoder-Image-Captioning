# GNN-Encoder-Image-Captioning

The model architecture is shown below: 
```
├── gnn_captioning
|   ├── config.py
│   ├── data.py
|   ├── models
│   ├── __init__.py
│   ├── evaluate.py
│   ├── run.py 
│   ├── train.py 
│   ├── utils.py  
├── project.ipynb
```
To run this project. Please follow below steps: 
### Step 1: 
Clone the github repository to local or your cloud server
```
git clone https://github.com/hankunw/GNN-Encoder-Image-Captioning.git
```
### Step 2: 
Download the datasets used in this project. 
[Flicker8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
[COCO dataset](https://www.kaggle.com/datasets/nikhil7280/coco-image-caption)
Please note that the coco dataset I used in this project is COCO Image Captioning Task 2014 val

### Step 3: 
After downloading datasets, please put them in the right place: 
Flicker8k dataset: put "Images" and "captions.txt" under gnn_captioning directory. 
COCO dataset: put coco images into "val2014" and captions into "annotations". Then put these two folders under gnn_captioning directory

The final directory looks like this: 
```
├── gnn_captioning
|   ├── val2014
|   ├── annotations
|   |   ├── captions_val2014.json
|   ├── captions.txt
|   ├── Images
|   ├── config.py
│   ├── data.py
|   ├── models
│   ├── __init__.py
│   ├── evaluate.py
│   ├── run.py 
│   ├── train.py 
│   ├── utils.py  
├── project.ipynb
```

### Step 4: 
Congratulations! Now let's move to environment. We will use python virtual environment to construct the environment (Linux/MacOS)
To create virtual environment, use below commands:
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip

```
If you use Windows, Please use below commands:
```
python -m venv venv
venv\Scripts\activate
venv\Scripts\python.exe -m pip install --upgrade pip
```
This project used torch==2.5.1+cu124, therefore, to avoid version conflict, please follow below instructions: 
```
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
Then please install corresponding version of pyg: 
```
pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install -q torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
```
Then install other packages used in this project:
```
pip install -r requirements.txt
```
### Step 5: 
After setting up the environment, you can run the projects as shown below:
if you want to train the model, please run below command
```
cd gnn_captioning
python run.py --mode train --model gcn 
```
note that there are three models that are avaliable to be trained: gcn, gat and vit. You can choose one of them 
After training, two plots will pump out and display the training trend. Meanwhile, one best model checkpoint will be saved under gnn_captioning directory. 
The saved checkpoint will have name of best_model_[gat, gcn, vit] (based on which model you are choosing).

to evaluate model, you need to specify the model checkpoint and run below command: 
```
cd gnn_captioning
python run.py --mode evaluate --model gat --testset coco --checkpoint \path\to\your\checkpoint
```
note that there are two test dataset avaliable: coco and flicker. you can choose one of them

### Note: I put the best model checkpoint under gnn_captioning/checkpoints. The file name is best_model_gat.pth. You can feel free to test it. 
### Note: The default ratio of flicker8k dataset is (0.8, 0.2, 0.0) for (training, val, test). You can refer to flicker_ratio in config.py to make mofifications. 
### Note: All used hyperparameters are in config.py, please feel free to try out with other hyperparameter sets. 

