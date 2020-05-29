import csv, time, inspect
from itertools import tee
from srcT.Common import ConfigFile as CF

def errorLog(listMsgs):
    return
    f=open(CF.filenameErrorLog, 'a')
    f.write(time.asctime() +'\t') 
    f.write('FileName='+ inspect.stack()[2][1] +'\t')
    f.write('FuncName='+ inspect.stack()[2][3] +'\t')
    f.write(joinLL(listMsgs, '=', '\t') + '\n')
    f.close()

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def NoneAnd(bool1, bool2):
    '''Return and of 2 bools, provided no-one is none'''
    if bool1 is None and bool2 is None:
        return None
    if bool1 is None:
        return bool2
    if bool2 is None:
        return bool1
    
    return bool1 and bool2

def NoneOr(bool1, bool2):
    '''Return and of 2 bools, provided no-one is none'''
    if bool1 is None and bool2 is None:
        return None
    if bool1 is None:
        return bool2
    if bool2 is None:
        return bool1
    
    return bool1 or bool2

def joinList(li, joinStr='\n', func=str):
    return joinStr.join([func(i) for i in li]) 

def joinLL(lists, joinStrWord=' ', joinStrLine='\n', func=str):
    listStrs = [joinList(li, joinStrWord, func) for li in lists]
    return joinList(listStrs, joinStrLine, func) 

def toInt(stri):
    if stri is None:
        return 0
    return int(stri)

def stringifyL(li):
    return [str(token) for token in li]

def stringifyLL(lists):
    return [stringifyL(li) for li in lists]

def readCSV(fname):
    f = open(fname, 'rU')
    freader = csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    lines = list(freader)
    f.close()
    headers = [i.strip() for i in lines[0]]

    return headers, lines[1:]

def writeCSV(fname, headers, lines):    
    fwriter = csv.writer(open(fname, 'w'), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
    fwriter.writerow(headers)
    fwriter.writerows(lines)

