# ATMTCR
ATMTCR consists of two parts, the MHC encoder and the TCR encoder. The packages required for the python runtime environment are in the MHC-encoder and the requirements.txt in the root directory respectively
. Please refer to our paper for more details: 'Attention-aware contrastive learning for predicting T
cell receptor-antigen binding specificity.'

The CDR3 sequences used for the pre-training process are from the TCRdb database.
(http://bioinfo.life.hust.edu.cn/TCRdb/#/).
This is the first time that the model is pre-trained for 10 million CDR3 sequences, and the encoder obtained by the model, which can directly encode CDR3 sequences, is convenient for use in related tasks.




## MHC-encoder:Guided Tutorial
Command:
```
python mhc_encoder.py -input input.csv -library library -output output_dir -output_log test/output/output.log
```
* input.csv: input csv file with 3 columns named as "CDR3,Antigen,HLA": TCR-beta CDR3 sequence, peptide sequence, and HLA allele.\
* library: diretory to the downloaded library
* output_dir : diretory you want to save the output
* output_log : local directory to log file with CDR, Antigen, HLA information and predicted binding rank.\


## Example 
The example input file is under data/.\
Comand :
```
python MHC-encoder/mhc_encoder.py -input data/dataset_2.csv -library library -output test/output -
output_log test/output/output.log
```
The output for test_input.csv is under test/output.

## Output file example
MHC-encoder outputs a table with 4 columns: CDR3 sequences, antigens sequences, HLA alleles, and ranks for each pair of TCR/pMHC. The rank reflects the percentile rank of the predicted binding strength between the TCR and the pMHC with respect to the 10,000 randomly sampled TCRs against the same pMHC. A lower rank considered a good prediction. The sequences of 10,000 background TCRs can be fold under https://github.com/tianshilu/pMTnet/tree/master/library/bg_tcr_library. 


## TCR-encoder:Guided Tutorial
Need to change hyperparameters such as dataset in the code.
Pre-training with TCR encoder,The encoder file is 'TCR-encoder/results/model_transformer_state_dict.pkl'
Comand :
```
python TCR-encoder/main_train.py
```

Encoding of CDR3 sequences for downstream tasks using pre-trained models.

Comand :
```
python TCR-encoder/main_con.py
```

## Main:Guided Tutorial
After encoding by two encoders, downstream prediction can be performed using the master function
Comand :
```
python main.py
```
