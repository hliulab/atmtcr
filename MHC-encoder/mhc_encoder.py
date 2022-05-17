import numpy as np
import pandas as pd
# import tensorflow as tf
import sys
import csv
import random
import os
from io import StringIO
import keras
from keras.layers import Input, Dense, concatenate, Dropout
from keras.models import Model, load_model
from keras import backend as K

args = sys.argv
file_dir=args[args.index('-input')+1] #input protein seq file
library_dir=args[args.index('-library')+1] #directory to downloaded library
#
model_dir=library_dir+'/h5_file'
aa_dict_dir=library_dir+'/Atchley_factors.csv' #embedding vector for tcr encoding
hla_db_dir=library_dir+'/hla_library/' #hla sequence
output_dir=args[args.index('-output')+1] #diretory to hold encoding and prediction output
output_log_dir=args[args.index('-output_log')+1] #standard output

# file_dir='data/dataset_1.csv'
#
# library_dir='library'
# #
# # model_dir=library_dir+'/h5_file'
# model_dir='library/h5_file'
# # aa_dict_dir=library_dir+'/Atchley_factors.csv' #embedding vector for tcr encoding
# aa_dict_dir='library/Atchley_factors.csv'
# # hla_db_dir=library_dir+'/hla_library/' #hla sequence
# hla_db_dir='library/hla_library'
# # output_dir=args[args.index('-output')+1] #diretory to hold encoding and prediction output
# output_dir='test/output'
# # output_log_dir=args[args.index('-output_log')+1] #standard output
# output_log_dir='test/output/output.log'
################################
# Reading Encoding Matrix #
################################
########################### Atchley's factors#######################
aa_dict_atchley=dict()
with open(aa_dict_dir,'r') as aa:
    aa_reader=csv.reader(aa)
    next(aa_reader, None)
    for rows in aa_reader:
        aa_name=rows[0]
        aa_factor=rows[1:len(rows)]
        aa_dict_atchley[aa_name]=np.asarray(aa_factor,dtype='float')
########################### One Hot ##########################   
aa_dict_one_hot = {'A': 0,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'K': 8,'L': 9,
           'M': 10,'N': 11,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'V': 17,
           'W': 18,'Y': 19,'X': 20}  # 'X' is a padding variable
########################### Blosum ########################## 
BLOSUM50_MATRIX = pd.read_table(StringIO(u"""                                                                                      
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  J  Z  X  *                                                           
A  5 -2 -1 -2 -1 -1 -1  0 -2 -1 -2 -1 -1 -3 -1  1  0 -3 -2  0 -2 -2 -1 -1 -5                                                           
R -2  7 -1 -2 -4  1  0 -3  0 -4 -3  3 -2 -3 -3 -1 -1 -3 -1 -3 -1 -3  0 -1 -5                                                           
N -1 -1  7  2 -2  0  0  0  1 -3 -4  0 -2 -4 -2  1  0 -4 -2 -3  5 -4  0 -1 -5                                                           
D -2 -2  2  8 -4  0  2 -1 -1 -4 -4 -1 -4 -5 -1  0 -1 -5 -3 -4  6 -4  1 -1 -5                                                           
C -1 -4 -2 -4 13 -3 -3 -3 -3 -2 -2 -3 -2 -2 -4 -1 -1 -5 -3 -1 -3 -2 -3 -1 -5                                                           
Q -1  1  0  0 -3  7  2 -2  1 -3 -2  2  0 -4 -1  0 -1 -1 -1 -3  0 -3  4 -1 -5                                                           
E -1  0  0  2 -3  2  6 -3  0 -4 -3  1 -2 -3 -1 -1 -1 -3 -2 -3  1 -3  5 -1 -5                                                           
G  0 -3  0 -1 -3 -2 -3  8 -2 -4 -4 -2 -3 -4 -2  0 -2 -3 -3 -4 -1 -4 -2 -1 -5                                                           
H -2  0  1 -1 -3  1  0 -2 10 -4 -3  0 -1 -1 -2 -1 -2 -3  2 -4  0 -3  0 -1 -5                                                          
I -1 -4 -3 -4 -2 -3 -4 -4 -4  5  2 -3  2  0 -3 -3 -1 -3 -1  4 -4  4 -3 -1 -5                                                           
L -2 -3 -4 -4 -2 -2 -3 -4 -3  2  5 -3  3  1 -4 -3 -1 -2 -1  1 -4  4 -3 -1 -5                                                           
K -1  3  0 -1 -3  2  1 -2  0 -3 -3  6 -2 -4 -1  0 -1 -3 -2 -3  0 -3  1 -1 -5                                                           
M -1 -2 -2 -4 -2  0 -2 -3 -1  2  3 -2  7  0 -3 -2 -1 -1  0  1 -3  2 -1 -1 -5                                                           
F -3 -3 -4 -5 -2 -4 -3 -4 -1  0  1 -4  0  8 -4 -3 -2  1  4 -1 -4  1 -4 -1 -5                                                           
P -1 -3 -2 -1 -4 -1 -1 -2 -2 -3 -4 -1 -3 -4 10 -1 -1 -4 -3 -3 -2 -3 -1 -1 -5                                                           
S  1 -1  1  0 -1  0 -1  0 -1 -3 -3  0 -2 -3 -1  5  2 -4 -2 -2  0 -3  0 -1 -5                                                           
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  2  5 -3 -2  0  0 -1 -1 -1 -5                                                           
W -3 -3 -4 -5 -5 -1 -3 -3 -3 -3 -2 -3 -1  1 -4 -4 -3 15  2 -3 -5 -2 -2 -1 -5                                                           
Y -2 -1 -2 -3 -3 -1 -2 -3  2 -1 -1 -2  0  4 -3 -2 -2  2  8 -1 -3 -1 -2 -1 -5                                                           
V  0 -3 -3 -4 -1 -3 -3 -4 -4  4  1 -3  1 -1 -3 -2  0 -3 -1  5 -3  2 -3 -1 -5                                                           
B -2 -1  5  6 -3  0  1 -1  0 -4 -4  0 -3 -4 -2  0  0 -5 -3 -3  6 -4  1 -1 -5                                                           
J -2 -3 -4 -4 -2 -3 -3 -4 -3  4  4 -3  2  1 -3 -3 -1 -2 -1  2 -4  4 -3 -1 -5                                                           
Z -1  0  0  1 -3  4  5 -2  0 -3 -3  1 -1 -4 -1  0 -1 -2 -2 -3  1 -3  5 -1 -5                                                           
X -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -5                                                           
* -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5  1                                                           
"""), sep='\s+').loc[list(aa_dict_one_hot.keys()), list(aa_dict_one_hot.keys())]
assert (BLOSUM50_MATRIX == BLOSUM50_MATRIX.T).all().all()

ENCODING_DATA_FRAMES = {
    "BLOSUM50": BLOSUM50_MATRIX,
    "one-hot": pd.DataFrame([
        [1 if i == j else 0 for i in range(len(aa_dict_one_hot.keys()))]
        for j in range(len(aa_dict_one_hot.keys()))
    ], index=aa_dict_one_hot.keys(), columns=aa_dict_one_hot.keys())
}

########################### HLA pseudo-sequence ##########################
#pMHCpan 
HLA_ABC=[hla_db_dir+'/A_prot.fasta',hla_db_dir+'/B_prot.fasta',hla_db_dir+'/C_prot.fasta',hla_db_dir+'/E_prot.fasta']
HLA_seq_lib={}
for one_class in HLA_ABC:
    prot=open(one_class)
    #pseudo_seq from netMHCpan:https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000796
    pseudo_seq_pos=[7,9,24,45,59,62,63,66,67,79,70,73,74,76,77,80,81,84,95,97,99,114,116,118,143,147,150,152,156,158,159,163,167,171]
    #write HLA sequences into a library
    #class I alles
    name=''
    sequence=''                                                                                                                        
    for line in prot:
        if len(name)!=0:
            if line.startswith('>HLA'):
                pseudo=''
                for i in range(0,33):
                    if len(sequence)>pseudo_seq_pos[i]:
                        pseudo=pseudo+sequence[pseudo_seq_pos[i]]
                HLA_seq_lib[name]=pseudo
                name=line.split(' ')[1]
                sequence=''
            else:
                sequence=sequence+line.strip()
        else:
            name=line.split(' ')[1]
########################################
# Input data encoding helper functions #
########################################
#################functions for TCR encoding####################
def preprocess(filedir):
    #Preprocess TCR files                                                                                                                 
    print('Processing: '+filedir)
    if not os.path.exists(filedir):
        print('Invalid file path: ' + filedir)
        return 0
    dataset = pd.read_csv(filedir, header=0)
    #Preprocess HLA_antigen files
    #remove HLA which is not in HLA_seq_lib; if the input hla allele is not in HLA_seq_lib; then the first HLA startswith the input HLA allele will be given     
    #Remove antigen that is longer than 15aa
    dataset=dataset.dropna()
    HLA_list=list(dataset['HLA'])
    ind=0
    index_list=[]
    for i in HLA_list:
        if len([hla_allele for hla_allele in HLA_seq_lib.keys() if hla_allele.startswith(str(i))])==0:
            index_list.append(ind)
            print('drop '+i)
        ind=ind+1
    dataset=dataset.drop(dataset.iloc[index_list].index)
    dataset=dataset[dataset.Antigen.str.len()<16]
    print(str(max(dataset.index)-dataset.shape[0]+1)+' antigens longer than 15aa are dropped!')
    dataset.to_csv("dataset.csv")
    TCR_list=dataset['CDR3'].tolist()
    antigen_list=dataset['Antigen'].tolist()
    HLA_list=dataset['HLA'].tolist()
    return TCR_list, antigen_list, HLA_list

def aamapping_TCR(peptideSeq,aa_dict):
    #Transform aa seqs to Atchley's factors.                                                                                              
    peptideArray = []
    if len(peptideSeq)>80:
        print('Length: '+str(len(peptideSeq))+' over bound!')
        peptideSeq=peptideSeq[0:80]
    for aa_single in peptideSeq:
        try:
            peptideArray.append(aa_dict[aa_single])
        except KeyError:
            print('Not proper aaSeqs: '+peptideSeq)
            peptideArray.append(np.zeros(5,dtype='float64'))
    for i in range(0,80-len(peptideSeq)):
        peptideArray.append(np.zeros(5,dtype='float64'))
    return np.asarray(peptideArray)

def hla_encode(HLA_name,encoding_method):
    #Convert the a HLA allele to a zero-padded numeric representation.
    if HLA_name not in HLA_seq_lib.keys():
        if len([hla_allele for hla_allele in HLA_seq_lib.keys() if hla_allele.startswith(str(HLA_name))])==0:
            print('cannot find'+HLA_name)
        HLA_name=[hla_allele for hla_allele in HLA_seq_lib.keys() if hla_allele.startswith(str(HLA_name))][0]
    if HLA_name not in HLA_seq_lib.keys():
        print('Not proper HLA allele:'+HLA_name)
    HLA_sequence=HLA_seq_lib[HLA_name]
    HLA_int=[aa_dict_one_hot[char] for char in HLA_sequence]
    while len(HLA_int)!=34:
        #if the pseudo sequence length is not 34, use X for padding
        HLA_int.append(20)
    result=ENCODING_DATA_FRAMES[encoding_method].iloc[HLA_int]
    # Get a numpy array of 34 rows and 21 columns
    return np.asarray(result)

def peptide_encode_HLA(peptide, maxlen,encoding_method):
    #Convert peptide amino acid sequence to numeric encoding
    if len(peptide) > maxlen:
        msg = 'Peptide %s has length %d > maxlen = %d.'
        raise ValueError(msg % (peptide, len(peptide), maxlen))
    peptide= peptide.replace(u'\xa0', u'')    #remove non-breaking space  
    o = list(map(lambda x: aa_dict_one_hot[x.upper()] if x.upper() in aa_dict_one_hot.keys() else 20 , peptide)) 
    #if the amino acid is not valid, replace it with padding aa 'X':20
    k = len(o)
    #use 'X'(20) for padding
    o = o[:k // 2] + [20] * (int(maxlen) - k) + o[k // 2:]
    if len(o) != maxlen:
        msg = 'Peptide %s has length %d < maxlen = %d, but pad is "none".'
        raise ValueError(msg % (peptide, len(peptide), maxlen))
    result=ENCODING_DATA_FRAMES[encoding_method].iloc[o]
    return np.asarray(result)

def TCRMap(dataset,aa_dict):
    #Wrapper of aamapping                                                                                                                 
    for i in range(0,len(dataset)):
        if i==0:
            TCR_array=aamapping_TCR(dataset[i],aa_dict).reshape(1,80,5,1)
        else:
            TCR_array=np.append(TCR_array,aamapping_TCR(dataset[i],aa_dict).reshape(1,80,5,1),axis=0)
    print('TCRMap done!')
    return TCR_array

def HLAMap(dataset,encoding_method):
    #Input a list of HLA and get a three dimentional array
    m=0
    for each_HLA in dataset:
        if m==0:
            HLA_array=hla_encode(each_HLA,encoding_method).reshape(1,34,21)
        else:
            HLA_array=np.append(HLA_array,hla_encode(each_HLA,encoding_method).reshape(1,34,21),axis=0)
        m=m+1
    print('HLAMap done!')
    return HLA_array

def antigenMap(dataset,maxlen,encoding_method):
    #Input a list of antigens and get a three dimentional array
    m=0
    for each_antigen in dataset:
        if m==0:
            antigen_array=peptide_encode_HLA(each_antigen,maxlen,encoding_method).reshape(1,maxlen,21)
        else:
            antigen_array=np.append(antigen_array,peptide_encode_HLA(each_antigen, maxlen,encoding_method).reshape(1,maxlen,21),axis=0)
        m=m+1
    print('antigenMap done!')
    return antigen_array

def pearson_correlation_f(y_true: object, y_pred: object) -> object:
    fsp = y_pred - K.mean(y_pred) #being K.mean a scalar here, it will be automatically subtracted from all elements in y_pred                
    fst = y_true - K.mean(y_true)
    devP = K.std(y_pred)
    devT = K.std(y_true)
    return K.mean(fsp*fst)/(devP*devT)

def pos_neg_acc(y_true,y_pred):
    #self-defined prediction accuracy metric
    positive_pred=y_pred[:,1]
    negative_pred=y_pred[:,0]
    diff=K.mean(K.cast(negative_pred<positive_pred,"float16"))
    return diff

def pos_neg_loss(y_true,y_pred):
    #self-defined prediction loss function 
    positive_pred=y_pred[:,1]
    negative_pred=y_pred[:,0]
    diff=K.mean(K.relu(1+negative_pred-positive_pred))+0.2*K.mean(K.square(negative_pred)+K.square(positive_pred))
    return diff


log_file=open(output_log_dir,'w')
sys.stdout=log_file
print('Mission loading.')

TCR_list,antigen_list,HLA_list=preprocess(file_dir)
antigen_array=antigenMap(antigen_list,15,'BLOSUM50')
HLA_array=HLAMap(HLA_list,'BLOSUM50')



HLA_antigen_encoder=load_model(model_dir+'/HLA_antigen_encoder_60.h5',custom_objects={'pearson_correlation_f': pearson_correlation_f})
HLA_antigen_encoder=Model(HLA_antigen_encoder.input,HLA_antigen_encoder.layers[-2].output)
HLA_antigen_encoded_result=HLA_antigen_encoder.predict([antigen_array,HLA_array])# 后两列的编码结果
np.save("HLA_antigen_encoded_result_test.npy", HLA_antigen_encoded_result)

print('Encoding Accomplished.\n')
