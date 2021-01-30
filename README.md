This repository contains the data and code for BERT Boost Box. The following steps are required to run the code:
* cd data
* Unzip the mask matrices 
  * unzip train_mask.zip
  * unzip valid_mask.zip
  * unzip test_mask.zip
* mkdir ../Results [All the outputs from the run will be saved in this folder]
* cd ../src
* Run the main.py file
  * python main.py --epochs 10 --patience 7 --batch_size 8 --alpha 0.5 --lr 1e-5 
  
    args  | Meaning
    ------------- | -------------
    epochs  | Number of Epochs
    patience  |  Number of epochs to wait before stopping training. Early stopping will stop training if the validation loss has not decreased after the patience. [patience < epochs]
    batch_size | Batch Size
    alpha | The hyperparameter to decide the contribution of the Cross-Entropy Loss (Loss_CE) and Mask Based loss (Loss_Mask) in the Total Loss. Total Loss = (1- α ) * Loss_CE + α * Loss_Mask  
    
