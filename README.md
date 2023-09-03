# MultiLoad-GAN_public
This repository includes the source code of the MultiLoad-GAN paper, a generative model to generate a group of load profiles considering spatial-temporal correlations.

***
NOTE: 

DUE THE PRIVACY ISSUE, WE ARE NOT ABLE TO SHARE THE ORIGINAL TRAINING DATA RIGHT NOW. CURRENTLY, YOU CAN ONLY RUN THE CODE WITH YOUR OWN DATA. AS SOON AS WE GET THE PERMISSION TO SHARE REAL DATA, WE WILL UPDATE IT.

We will keep updating the repository, making it easier to reproduce.

Please note that the paths in code are fixed as 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public', make sure to modify the path before trying to run the code.

Download data from https://drive.google.com/drive/folders/1uenITdDWMVU3MTGXlJ-VFnsTMd-Cwncq?usp=sharing
***

### Citation
If you use the data or code of this repository, please cite the paper:

Yi Hu, Yiyan Li, Lidong Song, Han Pyo Lee, PJ Rehm, Mattew Makdad, Edmond Miller, and Ning Lu, "MultiLoad-GAN: A GAN-Based Synthetic Load Group Generation Method Considering Spatial-Temporal Correlations," in IEEE Transactions on Smart Grid, doi: 10.1109/TSG.2023.3302192.

### Contact
Please send your questions and comments to the author at _hugh19flyer@gmial.com_.

### For a quick try
You can generate some synthetic data by MultiLoad-GAN or SingleLoad-GAN. We have the trained model in this repository. 

If you would like to try the synthetic generation method quickly, all you need to do is to run _evaluation.py_. Then you will get the generated data in dataset folder.

### Process dataset
In dataset folder, _GANData0.csv_ and _classiferSorted0.csv_ are the real data for training. Other files are the generated load data.
If you want to prepare the training data (recreate _GANData0.csv_) by yourself, _run newRiver.py_. In case you may not have access to the original full dataset, you can not run this part.

### Retrain MultiLoad-GAN
We have included the trained model in checkpoints. 
If you want to retrain MultiLoad-GAN, run _groupGAN.py_.
If you want to retrain the Automatic Data Augmentation process, run _train_model.py_.

### Retrain SingleLoad-GAN
We have included the trained model in SingleLoadGAN/checkpoints. 
If you want to retrain SingleLoad-GAN, run SingleLoad-GAN/singleUser_Train.py

### Retrain classifier
We have included the trained model in classifier/model. 
run classifier/classifier.py

### Show evaluation results
In _evaluation.py_, run the _eval()_ function.
