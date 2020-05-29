import warnings
warnings.filterwarnings('ignore')
import keras
import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from scipy import stats
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from difflib import ndiff,Differ,SequenceMatcher
import string
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.class_weight import compute_class_weight
from timeit import default_timer as timer
import sys

from srcT.Common import ConfigFile as CF

#if len(sys.argv)!=2:
#    print("Usage python train.py <path-to-train-data>")
#    sys.exit(1)
#data=pd.read_csv(sys.argv[1],encoding='latin')
data=pd.read_csv(CF.fnameSingleL_Train, encoding='latin')

print("Preprocessing...")
tic = timer()
insert=[0]*len(data['sourceLineAbs'])
delete=[0]*len(data['sourceLineAbs'])
replace=[0]*len(data['sourceLineAbs'])
for i in range(len(data['sourceLineAbs'])):
    x=data['sourceLineAbs'][i].split(' ')
    y=data['targetLineAbs'][i].split(' ')
    sm=SequenceMatcher(None,x,y)
    repl_str=[]
    del_str=[]
    ins_str=[]
    for opcodes in sm.get_opcodes():
        if opcodes[0]=='equal':
            continue
        elif opcodes[0]=='replace':
            tmp_str=''
            for st in x[opcodes[1]:opcodes[2]]:
                tmp_str+='- '+ st + '\n'
            for st in y[opcodes[3]:opcodes[4]]:
                tmp_str+='+ '+st + '\n'
            repl_str.append(tmp_str[:-1])
        elif opcodes[0]=='insert':
            tmp_str=''
            for st in y[opcodes[3]:opcodes[4]]:
                tmp_str+='+ '+st + '\n'
            ins_str.append(tmp_str[:-1])
        elif opcodes[0]=='delete':
            tmp_str=''
            for st in x[opcodes[1]:opcodes[2]]:
                tmp_str+='- '+st + '\n'
            del_str.append(tmp_str[:-1])
    insert[i]=ins_str
    delete[i]=del_str
    replace[i]=repl_str
insert=np.array(insert)
delete=np.array(delete)
replace=np.array(replace)

errset=np.array(data['newErrSet'])
diffset=[]
for i in range(len(replace)):
    tmp_str=''
    tmp_str+=errset[i]+'\n'
    for repl in replace[i]:
        tmp_str+=repl+'\n'
    for dlt in delete[i]:
        tmp_str+=dlt+'\n'
    for ins in insert[i]:
        tmp_str+=ins+'\n'
    diffset.append(tmp_str[:-1])
diffset=np.array(diffset)

encoder=LabelEncoder()
labels=encoder.fit_transform(diffset)

one_hot_labels=keras.utils.to_categorical(labels)

counts=one_hot_labels.sum(axis=0)

del_items=[]
for i in range(len(counts)):
    if counts[i]<2:
        del_items.append(i)

del_indexes=one_hot_labels[:,del_items].any(1)

diffset=diffset[~del_indexes]

new_encoder = LabelEncoder()
new_int_labels = new_encoder.fit_transform(diffset)

new_labels=keras.utils.to_categorical(new_int_labels)

src_abs_lines=np.array(data['sourceLineAbs'])
tgt_abs_lines=np.array(data['targetLineAbs'])
src_abs_lines=src_abs_lines[~del_indexes]
tgt_abs_lines=tgt_abs_lines[~del_indexes]

insert=insert[~del_indexes]
delete=delete[~del_indexes]
replace=replace[~del_indexes]

errset=errset[~del_indexes]

feature_vector=[]
for i in range(len(src_abs_lines)):
    tmp_lst=[]
    for err in errset[i].split(' '):
        tmp_lst.append(err.split(';')[0])
    tmp_line=src_abs_lines[i].split(' ')
    for abst in tmp_line:
        tmp_lst.append(abst)
    for ind in range(len(tmp_line)-1):
        tmp_lst.append(tmp_line[ind]+' '+tmp_line[ind+1])
    feature_vector.append(tmp_lst)
feature_vector=np.array(feature_vector)

tkn=keras.preprocessing.text.Tokenizer(filters='')
tkn.fit_on_texts(feature_vector)
encoded_vec=tkn.texts_to_matrix(feature_vector)

print("Preprocessing Done.")
print("Training Repair Class Classifiers...")
def train_lin_classifier(X,Y):
    lin_svm=LinearSVC()
    lin_svm.fit(X,Y)
    return lin_svm

def train_logreg(X,Y):
    logreg=LogisticRegression()
    logreg.fit(X,Y)
    return logreg

coarse_one=[]
for i in range(len(src_abs_lines)):
    if replace[i]!=[] and insert[i]==[] and delete[i]==[]:
        coarse_one.append(1)
    else:
        coarse_one.append(0)


coarse_one=np.array(coarse_one)

def train_non_lin_classifier(X,Y):
    model=Sequential()
    model.add(Dense(128,input_shape=(X.shape[1],),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128,input_shape=(X.shape[1],),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X,Y,epochs=10,batch_size=10,verbose=False)
    return model

repl_or_not=train_non_lin_classifier(encoded_vec,coarse_one)

def predict_repl(repl_or_not,X):
    p=repl_or_not.predict(X.reshape(1,X.shape[0]))[0][0]
    if p>0.5:
        return 1
    return 0

noRepl_ind=np.where(coarse_one==0)
noRepl_diffset=diffset[noRepl_ind]
noRepl_replace=replace[noRepl_ind]
noRepl_delete=delete[noRepl_ind]
noRepl_insert=insert[noRepl_ind]
noRepl_encoded_vec=encoded_vec[noRepl_ind]

coarse_two=[]
for i in range(len(noRepl_encoded_vec)):
    if noRepl_replace[i]==[] and noRepl_insert[i]!=[] and noRepl_delete[i]==[]:
        coarse_two.append(1)
    elif noRepl_replace[i]==[] and noRepl_insert[i]==[] and noRepl_delete[i]!=[]:
        coarse_two.append(2)
    else:
        coarse_two.append(0)
coarse_two=np.array(coarse_two)

def train_insdel(X,Y):
    model=Sequential()
    model.add(Dense(128,input_shape=(X.shape[1],),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128,input_shape=(X.shape[1],),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(Y.shape[1],activation='softmax'))
    model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X,Y,epochs=10,batch_size=10,verbose=False)
    return model

ctwo_oh=keras.utils.to_categorical(coarse_two)

ins_del_model=train_insdel(noRepl_encoded_vec,ctwo_oh)

def predict_insdel(ins_del_model,X):
    p=ins_del_model.predict(X.reshape(1,X.shape[0]))[0]
    return np.argmax(p)

repl_ind=np.where(coarse_one==1)
repl_diffset=diffset[repl_ind]
repl_encoded_vec=encoded_vec[repl_ind]

repl_encoder=LabelEncoder()
repl_int_labels=repl_encoder.fit_transform(repl_diffset)
repl_oh_labels=keras.utils.to_categorical(repl_int_labels)
repl_oh_labels.shape

def train_repl_class(X,Y):
    model=Sequential()
    model.add(Dense(256,input_shape=(X.shape[1],),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256,input_shape=(X.shape[1],),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(Y.shape[1],activation='softmax'))
    model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X,Y,epochs=15,batch_size=10,verbose=False)
    return model

repl_class_model=train_repl_class(repl_encoded_vec,repl_oh_labels)

def predict_repl_class(X):
    p=qwe.predict(X.reshape(1,X.shape[0]))[0]
    return np.argmax(p)

k_c = 25
repl_clusters=[0]*repl_oh_labels.shape[1]
for i in range(repl_oh_labels.shape[1]):
    train_X=repl_encoded_vec[np.where(repl_int_labels==i)]
    repl_clusters[i]=KMeans(math.ceil(train_X.shape[0]/k_c))
    repl_clusters[i].fit(train_X)

ins_ind=np.where(coarse_two==1)
ins_diffset=noRepl_diffset[ins_ind]
ins_encoded_vec=noRepl_encoded_vec[ins_ind]

ins_encoder=LabelEncoder()
ins_int_labels=ins_encoder.fit_transform(ins_diffset)
ins_oh_labels=keras.utils.to_categorical(ins_int_labels)
ins_oh_labels.shape

ins_class_model=train_logreg(ins_encoded_vec,ins_int_labels)

ins_clusters=[0]*ins_oh_labels.shape[1]
for i in range(ins_oh_labels.shape[1]):
    train_X=ins_encoded_vec[np.where(ins_int_labels==1)]
    ins_clusters[i]=KMeans(math.ceil(train_X.shape[0]/k_c))
    ins_clusters[i].fit(train_X)

del_ind=np.where(coarse_two==2)
del_diffset=noRepl_diffset[del_ind]
del_encoded_vec=noRepl_encoded_vec[del_ind]

del_encoder=LabelEncoder()
del_int_labels=del_encoder.fit_transform(del_diffset)
del_oh_labels=keras.utils.to_categorical(del_int_labels)
del_oh_labels.shape

del_class_model=train_logreg(del_encoded_vec,del_int_labels)

del_clusters=[0]*del_oh_labels.shape[1]
for i in range(del_oh_labels.shape[1]):
    train_X=del_encoded_vec[np.where(del_int_labels==i)]
    del_clusters[i]=KMeans(math.ceil(train_X.shape[0]/k_c))
    del_clusters[i].fit(train_X)

rest_ind=np.where(coarse_two==0)
rest_diffset=noRepl_diffset[rest_ind]
rest_encoded_vec=noRepl_encoded_vec[rest_ind]

rest_encoder=LabelEncoder()
rest_int_labels=rest_encoder.fit_transform(rest_diffset)
rest_oh_labels=keras.utils.to_categorical(rest_int_labels)
rest_oh_labels.shape

rest_class_model=train_logreg(rest_encoded_vec,rest_int_labels)

rest_clusters=[0]*rest_oh_labels.shape[1]
for i in range(rest_oh_labels.shape[1]):
    train_X=rest_encoded_vec[np.where(rest_int_labels==i)]
    rest_clusters[i]=KMeans(math.ceil(train_X.shape[0]/k_c))
    rest_clusters[i].fit(train_X)


print("Repair Class Classification Training Done.")
print("Training OvA Classifiers...")
src_edits=[]
for i,line in enumerate(src_abs_lines):
    line=line.split(' ')
    tmp_edit=[0]*len(line)
    sm=SequenceMatcher(None,line,tgt_abs_lines[i].split(' '))
    for opcodes in sm.get_opcodes():
        if opcodes[0]!='equal':
            if opcodes[0]=='insert':
                if opcodes[1]==len(line):
                    tmp_edit[opcodes[1]-1]=1
                else:
                    tmp_edit[opcodes[1]]=1
            else:
                for j in range(opcodes[1],opcodes[2]):
                    tmp_edit[j]=1
    src_edits.append(tmp_edit)

src_bigram=[]
edit_bigram=[]
for i in range(len(src_abs_lines)):
    tmp_lst=[]
    edit_lst=[]
    tmp_line=src_abs_lines[i].split(' ')
    for ind in range(len(tmp_line)-1):
        if src_edits[i][ind]==1 or src_edits[i][ind+1]==1:
            edit_lst.append(1)
        else:
            edit_lst.append(0)
        tmp_str=''
        tmp_str+=tmp_line[ind]+' '+tmp_line[ind+1]
        tmp_lst.append(tmp_str)
    src_bigram.append(tmp_lst)
    edit_bigram.append(edit_lst)
src_bigram=np.array(src_bigram)
edit_bigram=np.array(edit_bigram)

edit_tkn = keras.preprocessing.text.Tokenizer(filters='')

edit_tkn.fit_on_texts(src_bigram)

encoded_bigram=edit_tkn.texts_to_matrix(src_bigram)

sparse_edit_pos=np.zeros((src_bigram.shape[0],len(edit_tkn.word_index)+1))
for i in range(src_bigram.shape[0]):
    if src_bigram[i]==[]:
        sparse_edit_pos[i][0]=1
    for j in range(len(src_bigram[i])):
        if edit_bigram[i][j]==1:
            idx=edit_tkn.word_index[src_bigram[i][j].lower()]
            sparse_edit_pos[i][idx]=1

def train_multiple_rankers(oh_labels,int_labels,bigram,edit_pos):
    models=[0]*oh_labels.shape[1]
    accuracies=[]
    glb_cnt=0
    test_cnt=0
    for i in range(oh_labels.shape[1]):
        models[i]=OneVsRestClassifier(DecisionTreeClassifier())
        train_X=bigram[np.where(int_labels==i)]
        train_Y=edit_pos[np.where(int_labels==i)]
        models[i].fit(train_X,train_Y)
#        print(i)
    return models,accuracies

flat_rankers,flat_accuracies=train_multiple_rankers(new_labels,new_int_labels,encoded_bigram,sparse_edit_pos)

idx_to_bigram={}
for key,val in edit_tkn.word_index.items():
    idx_to_bigram[val]=key
print("OvA Classification Done.")

toc=timer()
print("Time Taken: "+str(toc-tic))
if not os.path.exists('./macer'):
    os.makedirs('./macer')
with open('macer/edit_pos_tokenizer','wb') as file:
    pickle.dump(edit_tkn,file)
with open('macer/feature_tokenizer','wb') as file:
    pickle.dump(tkn,file)
with open('macer/idx_to_bigram','wb') as file:
    pickle.dump(idx_to_bigram,file)
with open('macer/new_encoder','wb') as file:
    pickle.dump(new_encoder,file)
with open('macer/flat_rankers','wb') as file:
    pickle.dump(flat_rankers,file)
with open('macer/repl_or_not','wb') as file:
    pickle.dump(repl_or_not,file)
with open('macer/ins_del_model','wb') as file:
    pickle.dump(ins_del_model,file)
with open('macer/repl_class_model','wb') as file:
    pickle.dump(repl_class_model,file)
with open('macer/repl_encoder','wb') as file:
    pickle.dump(repl_encoder,file)
with open('macer/ins_class_model','wb') as file:
    pickle.dump(ins_class_model,file)
with open('macer/ins_encoder','wb') as file:
    pickle.dump(ins_encoder,file)
with open('macer/del_class_model','wb') as file:
    pickle.dump(del_class_model,file)
with open('macer/del_encoder','wb') as file:
    pickle.dump(del_encoder,file)
with open('macer/rest_class_model','wb') as file:
    pickle.dump(rest_class_model,file)
with open('macer/rest_encoder','wb') as file:
    pickle.dump(rest_encoder,file)
with open('macer/repl_clusters','wb') as file:
    pickle.dump(repl_clusters,file)
with open('macer/ins_clusters','wb') as file:
    pickle.dump(ins_clusters,file)
with open('macer/del_clusters','wb') as file:
    pickle.dump(del_clusters,file)
with open('macer/rest_clusters','wb') as file:
    pickle.dump(rest_clusters,file)
print("Model Saved.")