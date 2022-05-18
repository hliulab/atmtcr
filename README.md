# ATMTCR
ATMTCR consists of two parts, the MHC encoder and the TCR encoder. The packages required for the python runtime environment are in the MHC-encoder and the requirements.txt in the root directory respectively
. Please refer to our paper for more details: 'Attention-aware contrastive learning for predicting T
cell receptor-antigen binding specificity.'

The CDR3 sequences used for the pre-training process are from the TCRdb database.
(http://bioinfo.life.hust.edu.cn/TCRdb/#/).
This is the first time that the model is pre-trained for 10 million CDR3 sequences, and the encoder obtained by the model, which can directly encode CDR3 sequences, is convenient for use in related tasks.







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
Note: We use the mainstream model netMHCpan as the MHC encoder to ensure that the MHC as well as the antigenic peptide encoding can preserve richer features

Command:
```
python MHC-encoder/mhc_encoder.py -input input.csv -library library -output output_dir -output_log test/output/output.log
```
* input.csv: input csv file with 3 columns named as "CDR3,Antigen,HLA": TCR-beta CDR3 sequence, peptide sequence, and HLA allele.\
* library: diretory to the downloaded library
* output_dir : diretory you want to save the output
* output_log : local directory to log file with CDR, Antigen, HLA information and predicted binding rank.\
## Main:Guided Tutorial
After encoding by two encoders, downstream prediction can be performed using the master function
Comand :
```
python main.py
```
