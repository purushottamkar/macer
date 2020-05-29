import sys

if len(sys.argv)!=2:
    print("Usage: python testPredAtK.py <PredK>")
    sys.exit(1)

from timeit import default_timer as timer
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
from difflib import ndiff,Differ,SequenceMatcher
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import warnings
warnings.filterwarnings('ignore')

from srcT.Common import ConfigFile as CF

tic = timer()
with open('macer/edit_pos_tokenizer','rb') as file:
    edit_tkn=pickle.load(file)
with open('macer/feature_tokenizer','rb') as file:
    tkn=pickle.load(file)
with open('macer/idx_to_bigram','rb') as file:
    idx_to_bigram=pickle.load(file)
with open('macer/new_encoder','rb') as file:
    new_encoder=pickle.load(file)
with open('macer/flat_rankers','rb') as file:
    flat_rankers=pickle.load(file)
with open('macer/repl_or_not','rb') as file:
    repl_or_not=pickle.load(file)
with open('macer/ins_del_model','rb') as file:
    ins_del_model=pickle.load(file)
with open('macer/repl_class_model','rb') as file:
    repl_class_model=pickle.load(file)
with open('macer/repl_encoder','rb') as file:
    repl_encoder=pickle.load(file)
with open('macer/ins_class_model','rb') as file:
    ins_class_model=pickle.load(file)
with open('macer/ins_encoder','rb') as file:
    ins_encoder=pickle.load(file)
with open('macer/del_class_model','rb') as file:
    del_class_model=pickle.load(file)
with open('macer/del_encoder','rb') as file:
    del_encoder=pickle.load(file)
with open('macer/rest_class_model','rb') as file:
    rest_class_model=pickle.load(file)
with open('macer/rest_encoder','rb') as file:
    rest_encoder=pickle.load(file)
with open('macer/repl_clusters','rb') as file:
    repl_clusters=pickle.load(file)
with open('macer/ins_clusters','rb') as file:
    ins_clusters=pickle.load(file)
with open('macer/del_clusters','rb') as file:
    del_clusters=pickle.load(file)
with open('macer/rest_clusters','rb') as file:
    rest_clusters=pickle.load(file)


def test(src_line,errs,predAtK,tgt_line):
    '''Compare with ideal predicted target line'''
    global crrct
    tmp_bigram=create_bigram(src_line)

    enc_tmp_bigram=edit_tkn.texts_to_matrix(tmp_bigram)

    tmp_feat_vector=create_feat_vector(errs,src_line)

    enc_tmp_feat_vector=tkn.texts_to_matrix(tmp_feat_vector)

    repl_p=repl_or_not.predict(enc_tmp_feat_vector)[0][0]

    noRepl = ins_del_model.predict(enc_tmp_feat_vector)[0]

    start=timer()

    repl_pred=repl_class_model.predict(enc_tmp_feat_vector)
    ins_pred=ins_class_model.predict_proba(enc_tmp_feat_vector)
    del_pred=del_class_model.predict_proba(enc_tmp_feat_vector)
    rest_pred=rest_class_model.predict_proba(enc_tmp_feat_vector)

    end=timer()

#     Globals.corr_cls+=end - start

    # msk=get_repl_mask()
    # repl_dist=np.delete(repl_dist,msk,1)

    start=timer()

    repl_dist=get_dist(repl_clusters,enc_tmp_feat_vector)
    repl_pred=0.2*repl_dist + 0.8*repl_pred
    repl_pred=repl_pred * repl_p
            
    ins_dist=get_dist(ins_clusters,enc_tmp_feat_vector)
    msk=get_ins_mask()
    ins_dist=np.delete(ins_dist,msk,1)
    ins_pred=0.2*ins_dist + 0.8*ins_pred
    ins_pred=ins_pred*(1-repl_p)*noRepl[1]
    
    del_dist=get_dist(del_clusters,enc_tmp_feat_vector)
    msk=get_del_mask()
    del_dist=np.delete(del_dist,msk,1)
    del_pred=0.2*del_dist + 0.8*del_pred
    del_pred=del_pred*(1-repl_p)*noRepl[2]
    
    rest_dist=get_dist(rest_clusters,enc_tmp_feat_vector)
    msk=get_rest_mask()
    rest_dist=np.delete(rest_dist,msk,1)
    rest_pred=0.2*rest_dist + 0.8*rest_pred
    rest_pred=rest_pred*(1-repl_p)*noRepl[0]
    

    rp=re=ins=de=0
    
    sorted_repl_pred=sorted(repl_pred[0],reverse=True)
    sorted_ins_pred=sorted(ins_pred[0],reverse=True)
    sorted_del_pred=sorted(del_pred[0],reverse=True)
    sorted_rest_pred=sorted(rest_pred[0],reverse=True)

    end=timer()

#     Globals.rerank+= end - start
    
    targetLines=[]

    for i1 in range(predAtK):
        if sorted_repl_pred[rp]>=sorted_del_pred[de] and sorted_repl_pred[rp]>=sorted_ins_pred[ins] and sorted_repl_pred[rp]>=sorted_rest_pred[re]:
            repl_p=1
            edit=np.where(repl_pred[0]==sorted_repl_pred[rp])
            rp+=1
        elif sorted_ins_pred[ins]>=sorted_del_pred[de] and sorted_ins_pred[ins]>=sorted_repl_pred[rp] and sorted_ins_pred[ins]>=sorted_rest_pred[re]:
            repl_p=0
            noRepl=1
            edit=np.where(ins_pred[0]==sorted_ins_pred[ins])
            ins+=1
        elif sorted_del_pred[de]>=sorted_ins_pred[ins] and sorted_del_pred[de]>=sorted_repl_pred[rp] and sorted_del_pred[de]>=sorted_rest_pred[re]:
            repl_p=0
            noRepl=2
            edit=np.where(del_pred[0]==sorted_del_pred[de])
            de+=1
        elif sorted_rest_pred[re]>=sorted_del_pred[de] and sorted_rest_pred[re]>=sorted_ins_pred[ins] and sorted_rest_pred[re]>=sorted_repl_pred[rp]:
            repl_p=0
            noRepl=0
            edit=np.where(rest_pred[0]==sorted_rest_pred[re])
            re+=1


        start=timer()
        if repl_p==1:
            what_to_edit=repl_encoder.inverse_transform(edit[0][:1])
        else:
            if noRepl==1:
                what_to_edit=ins_encoder.inverse_transform(edit[0][:1])
            elif noRepl==2:
                what_to_edit=del_encoder.inverse_transform(edit[0][:1])
            else:
                what_to_edit=rest_encoder.inverse_transform(edit[0][:1])

        edit=new_encoder.transform(what_to_edit)
        edit_pos=np.zeros(shape=enc_tmp_bigram.shape)
        ones=np.where(enc_tmp_bigram[0]==1)
        for one in ones[0]:
            edit_pos[0][one]=flat_rankers[edit[0]].estimators_[one].predict(enc_tmp_bigram)
        # edit_pos=flat_rankers[edit[0]].predict(enc_tmp_bigram)

        end=timer()
#         Globals.bigram_rank+= end - start
        tmp_diff=what_to_edit[0].split('\n')[1:]
        pred_bigrams=get_predicted_bigrams(edit_pos,idx_to_bigram)
        where_to_edit=get_predicted_edit_pos(pred_bigrams,tmp_bigram)

        start=timer()
        where_to_edit=sorted(where_to_edit)
        add=[]
        dl=[]
        for token in tmp_diff:
            if token.startswith('-'):
                dl.append(token[2:])
            elif token.startswith('+'):
                add.append(token[2:])
        spcl_flg=0
        if '17;\n+ (\n+ )' in what_to_edit[0]:
            i=0
            while i< len(where_to_edit)-1:
                if where_to_edit[i] != where_to_edit[i+1]-1:
                    where_to_edit.remove(where_to_edit[i])
                    i-=1
                else:
                    where_to_edit.remove(where_to_edit[i+1])
                i+=1
            if len(where_to_edit)>1:
                split_line=src_line.split(' ')
                for l in range(len(split_line)-1):
                    tmp_str=split_line[l]+' '+split_line[l+1]
                    if tmp_str == tmp_bigram[0][where_to_edit[-1]]:
                        split_line[l]=split_line[l]+' )'
                        break
                for l in range(len(split_line)-1):
                    tmp_str=split_line[l]+' '+split_line[l+1]
                    if tmp_str == tmp_bigram[0][where_to_edit[-2]]:
                        split_line[l]=split_line[l]+' ('
                        spcl_flg=1
                        break
                target_line=''
                for l in range(len(split_line)):
                    target_line+=split_line[l]+' '
                target_line=target_line[:-1]    
#                 targetLines.append(target_line.split(' ')[:-1])
                if target_line == tgt_line:
                    crrct+=1
                    return
        if '15;\n- )\n+ )' in what_to_edit[0]:
            i=0
            while i< len(where_to_edit)-1:
                if where_to_edit[i] == where_to_edit[i+1]-1:
                    where_to_edit.remove(where_to_edit[i+1])
                i+=1
            if len(where_to_edit)>1:
                split_line=src_line.split(' ')
                mask=[0]*len(split_line)
                for l in range(len(split_line)-1):
                    tmp_str=split_line[l]+' '+split_line[l+1]
                    if tmp_str == tmp_bigram[0][where_to_edit[-1]]:
                        s=split_line[l].replace(')','')
                        mask[l]=1
                        break
                for l in range(len(split_line)-1):
                    tmp_str=split_line[l]+' '+split_line[l+1]
                    if tmp_str == tmp_bigram[0][where_to_edit[-2]]:
                        split_line[l]=split_line[l]+' )'
                        spcl_flg=1
                        break
                target_line=''
                for l in range(len(split_line)):
                    if mask[l]==0:
                        target_line+=split_line[l]+' '
                target_line=target_line[:-1]
#                 targetLines.append(target_line.split(' ')[:-1])
                if target_line == tgt_line:
                    crrct+=1
                    return
        if spcl_flg==1:
            continue
        if add==[]:
            split_line=src_line.split(' ')
            where_to_edit=sorted(where_to_edit,reverse=True)
            mask=[0]*len(split_line)
            for k in range(len(dl)):
                flg=0
                if len(split_line)==1:
                        s=split_line[0].replace(dl[k],'')
                        if s=='':
                            mask[0]=1
                else:
                    for j in where_to_edit:
                        for l in range(len(split_line)-1):
                            tmp_str=split_line[l]+' '+split_line[l+1]
                            if tmp_str==tmp_bigram[0][j]:
                                s=split_line[l].replace(dl[k],'')
                                if s=='':
                                    mask[l]=1
                                    flg=1
                                    where_to_edit.remove(j)
                                else:
                                    s=split_line[l+1].replace(dl[k],'')
                                    if s=='':
                                        mask[l+1]=1
                                        flg=1
                                        where_to_edit.remove(j)
                                break
                        if flg==1:
                            break
            target_line=''
            for l in range(len(split_line)):
                if mask[l]!=1:
                    target_line+=split_line[l]+' '
            target_line=target_line[:-1]
#             targetLines.append(target_line.split(' ')[:-1])
            if target_line == tgt_line:
                crrct+=1
                return
        elif dl==[]:
            target=[]
            add_all=''
            for x in add:
                add_all+=x+' '
            add_all=add_all[:-1]
            if tmp_bigram[0]==[]:
                target_line=add_all+' '+src_line
#                 targetLines.append(target_line.split(' ')[:-1])
                if target_line == tgt_line:
                    crrct+=1
                    return
                target_line=src_line+ ' ' +add_all
#                 targetLines.append(target_line.split(' ')[:-1])
                if target_line == tgt_line:
                    crrct+=1
                    return
            else:
                for j in where_to_edit:
                    if j==0:
                        edited_bigram=add_all+' '+tmp_bigram[0][j]
#                         targetLines.append(ins_bigram_to_line(tmp_bigram,edited_bigram,j).split(' ')[:-1])
                        if ins_bigram_to_line(tmp_bigram,edited_bigram,j) == tgt_line:
                            crrct+=1
                            return
                    if j-1 not in where_to_edit:
                        edited_bigram=tmp_bigram[0][j].split(' ')[0]+' '+add_all+' '+tmp_bigram[0][j].split(' ')[1]
#                         targetLines.append(ins_bigram_to_line(tmp_bigram,edited_bigram,j).split(' ')[:-1])
                        if ins_bigram_to_line(tmp_bigram,edited_bigram,j) == tgt_line:
                            crrct+=1
                            return
                    edited_bigram=tmp_bigram[0][j]+' '+add_all
#                     targetLines.append(ins_bigram_to_line(tmp_bigram,edited_bigram,j).split(' ')[:-1])
                    if ins_bigram_to_line(tmp_bigram,edited_bigram,j) == tgt_line:
                        crrct+=1
                        return
        else:
            split_line=src_line.split(' ')
            
            mask=[0]*len(split_line)
            if len(add)==len(dl):
                for x,y in zip(add,dl):
                    flg=0
                    for j in where_to_edit:
                        for l in range(len(split_line)-1):
                            tmp_str=split_line[l]+' '+split_line[l+1]
                            if tmp_str==tmp_bigram[0][j] and mask[l]==0 and mask[l+1]==0:
                                s=split_line[l].replace(y,x)
                                if s!=split_line[l]:
                                    mask[l]=s
                                    flg=1
                                    where_to_edit.remove(j)
                                else:
                                    s=split_line[l+1].replace(y,x)
                                    if s!=split_line[l+1]:
                                        mask[l+1]=s
                                        flg=1
                                        where_to_edit.remove(j)
                                break
                        if flg==1:
                            break
                target_line=''
                if tmp_bigram[0]!=[]:
                    for l in range(len(split_line)):
                        if mask[l]!=0:
                            target_line+=mask[l]+' '
                        else:
                            target_line+=split_line[l]+' '
                    target_line=target_line[:-1]
            else:
                add_all=''
                for x in add:
                    add_all+=x+' '
                add_all=add_all[:-1]
                split_line=src_line.split(' ')
                for k in range(len(dl)-1):
                    flg=0
                    for j in where_to_edit:
                        for l in range(len(split_line)-1):
                            tmp_str=split_line[l]+' '+split_line[l+1]
                            if tmp_str==tmp_bigram[0][j]:
                                s=split_line[l].replace(dl[k],'')
                                if s=='':
                                    mask[l]=1
                                    flg=1
                                    where_to_edit.remove(j)
                                else:
                                    s=split_line[l+1].replace(dl[k],'')
                                    if s=='':
                                        mask[l+1]=1
                                        flg=1
                                        where_to_edit.remove(j)
                                break
                        if flg==1:
                            break
                flg=0
                for j in where_to_edit:
                    for l in range(len(split_line)-1):
                        tmp_str=split_line[l]+' '+split_line[l+1]
                        if tmp_str==tmp_bigram[0][j]:
                            s=split_line[l].replace(dl[-1],add_all)
                            if s!=split_line[l] and mask[l]!=1:
                                mask[l]=s
                                flg=1
                                where_to_edit.remove(j)
                            else:
                                s=split_line[l+1].replace(dl[-1],add_all)
                                if s!=split_line[l+1]:
                                    mask[l+1]=s
                                    flg=1
                                    where_to_edit.remove(j)
                            break
                    if flg==1:
                        break
                target_line=''
                for l in range(len(split_line)):
                    if mask[l]!=0 and mask[l]!=1:
                        target_line+=mask[l]+' '
                    elif mask[l]!=1:
                        target_line+=split_line[l]+' '
                target_line=target_line[:-1]
#             targetLines.append(target_line.split(' ')[:-1])
            if target_line == tgt_line:
                crrct+=1
                return
            add_all=''
            for x in add:
                add_all+=x+' '
            add_all=add_all[:-1]
            split_line=src_line.split(' ')
            mask=[0]*len(split_line)
            for k in range(len(dl)):
                flg=0
                i=0
                while i < len(where_to_edit):
                    for l in range(len(split_line)-1):
                        tmp_str=split_line[l]+' '+split_line[l+1]
                        if tmp_str == tmp_bigram[0][where_to_edit[i]]:
                            s=split_line[l].replace(dl[k],'')
                            if s=='':
                                mask[l]=1
                                flg=1
                            s=split_line[l+1].replace(dl[k],'')
                            if s=='':
                                mask[l+1]=1
                                flg=1
                            break
                    if flg==1:
                        break
                    i+=1
            for j in where_to_edit:
                for l in range(len(split_line)-1):
                    tmp_str= tmp_bigram[0][j]
                    if tmp_str ==  split_line[l]+ ' '+split_line[l+1]:
                        if mask[l]!=1:
                            split_line[l]=split_line[l]+ ' '+ add_all
#                             targetLines.append(make_target_line(split_line,mask).split(' ')[:-1])
                            if make_target_line(split_line,mask) == tgt_line:
                                crrct+=1
                                return
                        if mask[l+1]!=1:
                            split_line[l+1]=split_line[l+1]+ ' '+ add_all
#                             targetLines.append(make_target_line(split_line,mask).split(' ')[:-1])
                            if make_target_line(split_line,mask) == tgt_line:
                                crrct+=1
                                return

        end=timer()
#         Globals.fixer+= end - start
#     return targetLines    

def make_target_line(split_line,mask):
    target_line=''
    for l in range(len(split_line)):
        if mask[l]==0:
            target_line+=split_line[l]+' '
    target_line=target_line[:-1]
    return target_line
def create_bigram(src_line):
    tmp_bigram=[]
    tmp_lst=[]
    tmp_line=src_line.split(' ')
    for ind in range(len(tmp_line)-1):
        tmp_str=''
        tmp_str+=tmp_line[ind]+' '+tmp_line[ind+1]
        tmp_lst.append(tmp_str)
    tmp_bigram.append(tmp_lst)
    return tmp_bigram

def create_feat_vector(errs,src_line):
    tmp_feat_vector=[]
    tmp_lst=[]
    for err in errs.split(' '):
        tmp_lst.append(err.split(';')[0])
    tmp_line=src_line.split(' ')
    for abst in tmp_line:
        tmp_lst.append(abst)
    for ind in range(len(tmp_line)-1):
        tmp_lst.append(tmp_line[ind]+' '+tmp_line[ind+1])
    tmp_feat_vector.append(tmp_lst)
    return tmp_feat_vector

def get_predicted_bigrams(specific_prediction,idx_to_bigram):
    predicted_bigrams=[]
    for x in np.where(specific_prediction[0]==1)[0]:
        if idx_to_bigram.get(x)!=None:
            predicted_bigrams.append(idx_to_bigram[x])
        else:
            predicted_bigrams.append(-1)
    return predicted_bigrams

def get_predicted_edit_pos(predicted_bigrams,act_bigram):
    pred_edit_pos=[]
    for x in range(len(predicted_bigrams)):
        if predicted_bigrams[x]==-1:
            pred_edit_pos.append(-1)
            continue
        for y in range(len(act_bigram[0])):
            if predicted_bigrams[x]==act_bigram[0][y].lower():
                pred_edit_pos.append(y)
    return pred_edit_pos

def ins_bigram_to_line(tmp_bigram,edited_bigram,j):
    target_line=''
    if j==0:
        target_line+=edited_bigram+' '
    else:
        target_line+=tmp_bigram[0][0]+' '
    for x in range(1,len(tmp_bigram[0])):
        if x==j:
            target_line+=edited_bigram.split(' ',maxsplit=1)[-1]+' '
        else:
            target_line+=tmp_bigram[0][x].split(' ',maxsplit=1)[-1]+' '
    target_line=target_line[:-1]
    return target_line

def filter_bigrams(tmp_diff,tmp_enc_src_bigram,idx_to_bigram):
    idxs=[]
    for tkns in tmp_diff:
        if tkns.startswith('-'):
            for k in range(1,tmp_enc_src_bigram.shape[1]):
                if idx_to_bigram.get(k)!=None:
                    if tkns[2:].lower() in idx_to_bigram.get(k):
                        if str(k) not in idxs:
                            idxs.append(str(k))
    tmp_tkn=keras.preprocessing.text.Tokenizer(filters='')
    tmp_tkn.fit_on_texts(idxs)
    filt_repl_enc_src_bigram=np.zeros((1,len(tmp_tkn.word_index)+1))
    for idx in idxs:
        filt_repl_enc_src_bigram[0][tmp_tkn.word_index[idx]]=tmp_enc_src_bigram[0][int(idx)]
    tmp_idx_to_bigram=get_idx_to_bigram(idx_to_bigram,tmp_tkn.word_index)
    return tmp_idx_to_bigram,filt_repl_enc_src_bigram

def get_repl_mask():    
    msk=[]
    for i in range(len(repl_clusters)):
        if i not in repl_class_model.classes_:
            msk.append(i)
    msk=np.array(msk)
    return msk

def get_ins_mask():    
    msk=[]
    for i in range(len(ins_clusters)):
        if i not in ins_class_model.classes_:
            msk.append(i)
    msk=np.array(msk)
    return msk

def get_del_mask():    
    msk=[]
    for i in range(len(del_clusters)):
        if i not in del_class_model.classes_:
            msk.append(i)
    msk=np.array(msk)
    return msk

def get_rest_mask():    
    msk=[]
    for i in range(len(rest_clusters)):
        if i not in rest_class_model.classes_:
            msk.append(i)
    msk=np.array(msk)
    return msk

def get_dist(clusters,enc_tmp_feat_vector):
    clst=[]
    for i in range(len(clusters)):
        if clusters[i].cluster_centers_.shape[0]==1:
            clst.append(clusters[i].cluster_centers_[0])
        else:
            clst_dist=euclidean_distances(enc_tmp_feat_vector,clusters[i].cluster_centers_)[0]
            ind=np.where(clst_dist==min(clst_dist))[0][:1]
            clst.append(clusters[i].cluster_centers_[ind][0])
    clst=np.array(clst)
    dist=euclidean_distances(enc_tmp_feat_vector,clst)
    dist=np.exp((-1/2)*(dist**2))
    return dist

def predict_repl(repl_or_not,X):
    p=repl_or_not.predict(X)[0][0]
    if p>0.5:
        return 1
    return 0

def predict_insdel(ins_del_model,X):
    p=ins_del_model.predict(X)[0]
    return np.argmax(p)

test_data=pd.read_csv(CF.fnameSingleL_Test,encoding="ISO-8859-1")
crrct=0
k = int(sys.argv[1])
for i in range(len(test_data)):
    test(test_data['sourceLineAbs'][i],test_data['newErrSet'][i],k,test_data['targetLineAbs'][i])
    if i!=0 and i%100 == 0: 
        print('\t...',i,'/',len(test_data),'Completed')
toc = timer()
print("Time Taken: "+str(round(toc-tic,1))+"s")
print("Pred@"+str(k)+": "+str(round(crrct/len(test_data),3)))
