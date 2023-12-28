### Read the REPORT.pdf file present in the root folder of this repo
### Setup
```
# Clone the repo or download and unzip it
git clone https://github.com/hritikakolkar/wysa.git
```
Download the weights from this [link](https://drive.google.com/file/d/1njV4qQAn0qJFNGGMRnad2uO1xyO6Uhvk/view?usp=sharing) unzip it and copy or move the 'weights' folder in the root folder of the project
```
# Creating Virtual Environment
python3 -m venv venv
# for debian
source venv/bin/activate
# install requirements.txt
pip install -r requirements.txt
```

### Check the results of Test Data
Go to data/results.xlsx

### Run App 
```
python app.py
```
You can also check the app running on my local instance hosted on gradio but you can only access it within 72 hours
[Live App check here](https://63d4caeef4ca3e7062.gradio.live/)

### To train model
```
# You just have to edit the config file and all the model training will work efficiently.
# Do edit the versions in the config file every time you run a new experiment
# You will require to provide wandb access token for experiment tracking

python train.py src/config/train.yaml
```

### For inference 
```
# do check inference.py for custom batch inference

python inference.py
```

### Do checkout the jupyter notebooks in notebooks folder which contains EDA, Model Training and Inference Notebooks

### Please do check the reports I generated in wandb
- [Model Version Comparison](https://api.wandb.ai/links/hritikakolkar/avm8upcx)
- [Finetuned Model Report](https://api.wandb.ai/links/hritikakolkar/70rvxa5h)