import re, pandas as pd
import os

from srcT.DataStruct.Code import Code
from srcT.Common import ConfigFile as CF, Helper as H

#region: Read error-IDs (post creation)

def getAllErrs(fname):
    allErrs = {}
    if not os.path.exists(fname):
        saveAllErrs(allErrs)

    df = pd.read_csv(fname, index_col=False)
    for i, row in df.iterrows():
        allErrs[row['msg']] = row['id']

    return allErrs

def saveAllErrs(allErrs):
    data = []
    for msg in allErrs:
        iden = allErrs[msg]
        data.append((iden, msg))

    df = pd.DataFrame(data=data, columns=['id', 'msg'])
    df.to_csv(CF.fname_newErrIDs, index=False)

def replIDs(msg):
    msg = re.sub(r'\'(.*?)\'', 'ID', msg)
    msg = re.sub('\d+', 'NUM', msg)

    return msg

def getErrIDs(allErrs, codeObj, lineNum=None):
    eids = []

    for ce in codeObj.getSevereErrors():
        if lineNum is None or ce.line == lineNum: # If filter by lineNum
            msg = replIDs(ce.msg)

            if msg not in allErrs:
                allErrs[msg] = len(allErrs)+1
                saveAllErrs(allErrs)
            eids.append(allErrs[msg])

    return sorted(eids)

def getErrSet(allErrs, codeObj, lineNum=None):
    eids = getErrIDs(allErrs, codeObj, lineNum)
    return set(eids)

def getErrSetStr(allErrs, codeObj, lineNum=None):
    errSet = getErrSet(allErrs, codeObj, lineNum)
    return H.joinList(errSet, ';') + ';'

#endregion

#region: Create Error IDs for first time use

def createClass(fnameDataset):
    '''Given a dataset (CSV) file, replace old error-IDs (obtained using regex) with new ones (obtained using Clang LLVM)'''
    df = pd.read_csv(fnameDataset, encoding = "ISO-8859-1")
    allErrs = getAllErrs(CF.fname_newErrIDs)
    classes, classesRepeat, newErrSets = [], [], []
    mult = 10

    for i, row in df.iterrows():
        oldClass = row['errSet_diffs']
        codeObj = Code(row['sourceText'])

        newErrsetStr = getErrSetStr(allErrs, codeObj)
        newClass = newErrsetStr +'\n'+ H.joinList(oldClass.splitlines()[1:])

        newErrSets.append(newErrsetStr)
        classes.append(newClass)

        if i >= len(df)*mult/100:
            print(str(mult) +'%', end=' ', flush=True)
            mult += 10

    df['class'] = classes
    df['newErrSet'] = newErrSets
    df.to_csv(fnameDataset, index=False)

#endregion

#region: Main

if __name__=='__main__':
    print('Creating Training-Set Classes')
    createClass(CF.fnameSingleL_Train)

    print('\nCreating Testing-Set Classes')
    createClass(CF.fnameSingleL_Test)

#endregion
