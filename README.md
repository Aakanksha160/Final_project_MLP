# Final_project_MLP
Credit card fraud poses significant financial risks, necessitating robust detection methods. Traditional approaches face challenges in leveraging unlabeled transaction data and incorporating categorical attributes effectively.

## **Usage**

### **Data processing**
1.Run unzip /data/Amazon.zip and unzip /data/YelpChi.zip to unzip the datasets;
2.Run python feature_engineering/data_process.py  to pre-process all datasets needed in this repo.

## **Training & Evalutaion**

To test implementations of MCNN please run

python main.py --method mcnn

Configuration files can be found in config/mcnn_cfg.yaml

## **Repo Structure**

The repository is organized as follows:

-models/: the pre-trained models for each method. The readers could either train the models by themselves or directly use our pre-trained models;

-data/: dataset files;

-config/: configuration files for different models;

-feature_engineering/: data processing;

-methods/: implementations of models;

-main.py: organize all models;

-requirements.txt: package dependencies;

## **Requirements**

python           3.7

scikit-learn     1.0.2

pandas           1.3.5

numpy            1.21.6

networkx         2.6.3

scipy            1.7.3

torch            1.12.1+cu113

dgl-cu113        0.8.1
