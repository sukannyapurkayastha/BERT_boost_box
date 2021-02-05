This repository contains the data and code for BERT Boost Box. The following steps are required to run the code:
* cd data [Contains data without noise in training set]
* Unzip the mask matrices 
  * unzip train_mask.zip
  * unzip valid_mask.zip
  * unzip test_mask.zip
* cd ../data_noise_added/ [Contains data with noise in training set]
* Unzip the mask matrices 
  * unzip train_mask.zip
  * unzip valid_mask.zip
  * unzip test_mask.zip
* mkdir ../checkpoints [All the checkpoints from the run along with the config files will be saved in this folder]
* mkdir ../Results [All the results from the run along with the config files will be saved in this folder]
* cd ../src
* Run the main.py file
  * python main.py --path ../data_noise_added/ --epochs 10 --patience 7 --batch_size 8 --alpha 0.5 --lr 1e-5 
  
    args  | Meaning
    ------------- | -------------
    epochs  | Number of Epochs
    patience  |  Number of epochs to wait before stopping training. Early stopping will stop training if the validation loss has not decreased after the patience. [patience < epochs]
    batch_size | Batch Size
    path | path to Dataset eg., ../data_noise_added/ to run noisy dataset code
    alpha | The hyperparameter to decide the contribution of the Cross-Entropy Loss (Loss_CE) and Mask Based loss (Loss_Mask) in the Total Loss. Total Loss = (1- α ) * Loss_CE + α * Loss_Mask  
    
