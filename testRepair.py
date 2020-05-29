import sys

def error_args():
    print("Usage: python testRepair.py <dataset/path-to-test-file> <PredK>")
    print("Eg1: python data/input/fig_1a.c 5")
    print("Eg2: python testRepair.py tracer_single 5")
    print("Eg3: python testRepair.py deepfix 5")
    sys.exit(1)


if len(sys.argv)!=3:
    error_args()
    

from srcT.DataStruct.Code import Code
from srcT.DataStruct import ClusterError
from srcT.Common import ConfigFile as CF, Helper as H
from srcT.Symbolic import AbstractWrapper, ConcreteWrapper, ConcreteToken
from srcT.Prediction import Predict, Globals

import pandas as pd, copy
from timeit import default_timer as timer

#region: Global edits
activeLocalization=True

useTracers_errLoc = True # Use tracer's loc? Line-1, Line, Line+1
flagErrSet_Line = False # Pass only line specific err-sets to Macer?

AllErrs = ClusterError.getAllErrs(CF.fname_newErrIDs)
#endregion

#region: Accuracy

def checkRelevant(predText, predErrAbsLines, trgtText, trgtErrAbsLines):    
    trgtLL = [line.split() for line in trgtErrAbsLines.splitlines()]
    predLL=[]
    for line in predErrAbsLines:
        if line!=[]:
            predLL.append(line)
    isRelevant = trgtLL == predLL

    if isRelevant==False:
        tgt_text= [line.split() for line in trgtText.splitlines()]
        pred_text= [line.split() for line in predText.splitlines()]
        if tgt_text == pred_text:
            isRelevant=True

    return isRelevant

def checkRelevant2(predAbsLine, trgtAbsLine):
    return predAbsLine == trgtAbsLine

def calcAccuracy(actLinesStr, predLineNums, trgtText, trgtErrAbsLines, predErrAbsLines, predErrLines, predText):
    # isLocated
    isLocated = True    
    
    try:
        for actLineNum in actLinesStr.splitlines():
            if int(actLineNum) not in predLineNums:
                isLocated = False
    except ValueError as e:
        isLocated = False

    # isRelevant
    isRelevant = checkRelevant(predText, predErrAbsLines, trgtText, trgtErrAbsLines)

    # isCompiled 
    predCodeObj = Code(predText)    
    isCompiled = predCodeObj.getNumErrors() == 0

    return isLocated, isRelevant, isCompiled

#endregion

#region: 3-Phase
def errLoc(activeLocalization, srcCodeObj, actLinesStr, useTracers_errLoc=False):
    '''If errorLocalization is active, return compiler reported line.
Else, return the ideal (source-target text diff) lines'''
    if activeLocalization:
        predLines = srcCodeObj.getCE_lines()

        if useTracers_errLoc:
            prevLines = [line-1 for line in predLines]
            nextLines = [line+1 for line in predLines]
            return predLines + prevLines + nextLines

        return predLines

    return actLinesStr.split('\n')

def repairErrLine(srcCodeObj, repairLines, repairAbsLines, srcAbsLine, trgtLine, trgtAbsLine, errSetLine, lineNum, predErrAbsLines, predErrLines, predAtK):
    '''Pred@K and concretize the best line (with least errors)'''
    isConcretized, isExactMatch = None, None
    bestPredAbsLine, bestPredLine = None, None
    bestPredAbsLines, bestPredLines = repairAbsLines, repairLines

    prePredCodeObj = Code(H.joinList(repairLines))
    minNumErrs = prePredCodeObj.getNumErrors()

    for predAbsLine in Predict.predictAbs(srcAbsLine, errSetLine, trgtAbsLine, predAtK):
        # Create copy of previous obtained repairLines, and replace with predictedLines
        predLines, predAbsLines = copy.deepcopy(repairLines), copy.deepcopy(repairAbsLines)
        predAbsLines[lineNum - 1] = H.joinList(predAbsLine, joinStr=' ')
        
        # Concretize the predicted abstract fix
        predLine, tempIsConcretized = ConcreteWrapper.attemptConcretization(srcCodeObj, lineNum, predAbsLine)
        predLines[lineNum - 1] = H.joinList(predLine, joinStr=' ')

        # Concretization success?
        isConcretized = H.NoneAnd(isConcretized, tempIsConcretized) 
        tempIsExactMatch = checkRelevant2(predAbsLine, trgtAbsLine)         
        isExactMatch = H.NoneOr(isExactMatch, tempIsExactMatch) 

        # Find best prediction
        predCodeObj = Code(H.joinList(predLines))    
        if minNumErrs is None or predCodeObj.getNumErrors() < minNumErrs:
            minNumErrs = predCodeObj.getNumErrors()
            bestPredAbsLines, bestPredLines = predAbsLines, predLines
            bestPredAbsLine, bestPredLine = predAbsLine, predLine

    return bestPredAbsLine, bestPredLine, bestPredAbsLines, bestPredLines, isConcretized, isExactMatch
        

def runPerLine(srcCodeObj, srcLines, trgtLines, srcAbsLines, trgtAbsLines, errSet, lineNums, predAtK):     
    '''For each compiler error line, call predErrLine'''   
    srcErrLines, srcErrAbsLines = [], []
    predErrLines, predErrAbsLines = [], []
    repairLines, repairAbsLines = copy.deepcopy(srcLines), copy.deepcopy(srcAbsLines)
    isConcretized, isExactMatch = None, None

    # For each compiler flagged lineNums            
    for lineNum in lineNums: 
        lineNum=int(lineNum)

        if lineNum <= min([len(srcLines), len(srcAbsLines)]): # If compiler returned valid line-num
            srcLine, srcAbsLine = srcLines[lineNum - 1], srcAbsLines[lineNum - 1] # lineNum-1 since off-by-one                
            trgtLine, trgtAbsLine = None, None
            if lineNum <= min([len(trgtLines), len(trgtAbsLines)]) and lineNum > 0:
                trgtLine, trgtAbsLine =  trgtLines[lineNum-1], trgtAbsLines[lineNum-1]      
            srcErrLines.append(srcLine), srcErrAbsLines.append(srcAbsLine)
            
            # Use ErrSet at line=lineNum? Or at program-level
            errSetLine = errSet
            if flagErrSet_Line:
                errSetLine = ClusterError.getErrSetStr(AllErrs, srcCodeObj, lineNum=lineNum)

            # Predict@K the concrete repair line 
            predAbsLine, predLine, repairAbsLines, repairLines, tempIsConcretized, tempIsExactMatch = repairErrLine(srcCodeObj, \
                repairLines, repairAbsLines, srcAbsLine, trgtLine, trgtAbsLine, errSetLine, lineNum, \
                predErrAbsLines, predErrLines, predAtK)

            # Concretization success?
            isConcretized = H.NoneAnd(isConcretized, tempIsConcretized)
            isExactMatch =  H.NoneAnd(isExactMatch, tempIsExactMatch)

            # Record the predicted abstract and concrete line
            if predAbsLine is not None:
                predErrAbsLines.append(predAbsLine)
                predErrLines.append(predLine)

    predText = H.joinList(repairLines)
    return predText, srcErrLines, predErrLines, srcErrAbsLines, predErrAbsLines, isConcretized, isExactMatch

#endregion

#region: Main functions

def run(df, predAtK):
    startTime = timer()
    columns = ['id', 'sourceText', 'targetText', 'predText', 'actLineNums', 'predLineNums', \
            'actSourceLine', 'localSourceLine', 'targetLine', 'predLine', \
            'actSourceAbsLine', 'localSourceAbsLine', 'targetAbsLine', 'predAbsLine', \
            'errSet', 'isLocated', 'isRelevant', 'isConcretized', 'isExactMatch', 'isCompiled']
    results = []    #True to turn on localization Module, False to turn off
    #allErrors = ClusterError.getAllErrs()

    # For each erroneous code
    for i, row in df.iterrows(): 
        srcID, trgtID = str(row['id']) + '_source', str(row['id']) + '_target'
        srcText, trgtText = str(row['sourceText']), str(row['targetText'])
        trgtErrLines, trgtErrAbsLines = str(row['targetLineText']).strip(), str(row['targetLineAbs']).strip()
        actLinesStr = str(row['lineNums_Text'])        
            
        # Parse the source/erroneous code
        srcCodeObj, trgtCodeObj = Code(srcText, codeID=srcID), Code(trgtText, codeID=trgtID)
        srcLines, trgtLines = srcText.splitlines(), trgtText.splitlines()
        errSet = ClusterError.getErrSetStr(AllErrs, srcCodeObj)

        # Fetch its abstraction
        srcAbsLines = AbstractWrapper.getProgAbstraction(srcCodeObj)
        trgtAbsLines = AbstractWrapper.getProgAbstraction(trgtCodeObj)        

        #Fetch Line numbers
        lineNums = errLoc(activeLocalization, srcCodeObj, actLinesStr, useTracers_errLoc)

        if srcCodeObj.getNumErrors() > 0: # If there are errors
            # Run prediction on all erroneous lines
            predText, srcErrLines, predErrLines, srcErrAbsLines, predErrAbsLines, isConcretized, isExactMatch  = \
                runPerLine(srcCodeObj, srcLines, trgtLines, srcAbsLines, trgtAbsLines,errSet,lineNums,predAtK)        

            # Calculate accuracy and log it                    
            isLocated, isRelevant, isCompiled = calcAccuracy(actLinesStr, lineNums, \
                trgtText, trgtErrAbsLines, predErrAbsLines, predErrLines, predText)

            results.append((row['id'], srcText, trgtText, predText, actLinesStr, H.joinList(lineNums), \
                row['sourceLineText'], H.joinList(srcErrLines), trgtErrLines, H.joinLL(predErrLines), \
                row['sourceLineAbs'], H.joinLL(srcErrAbsLines), trgtErrAbsLines, H.joinLL(predErrAbsLines), errSet, \
                H.toInt(isLocated), H.toInt(isRelevant), H.toInt(isConcretized), H.toInt(isExactMatch), H.toInt(isCompiled)))

        if i!=0 and i%100 == 0: 
            print('\t...',i,'/',len(df),'Completed')            
            # break
        
    endTime = timer()
    print('\n#Programs=', len(df), 'Time Taken=', round(endTime - startTime, 2), '(s)')
    return pd.DataFrame(results, columns=columns)

def runTest(fname, predAtK):
    df_data = pd.read_csv(fname, encoding = "ISO-8859-1")

    df_results = run(df_data, predAtK)
    df_results.to_csv(CF.pathOut + 'results_PredAt_'+str(predAtK)+'.csv')
    
    print('-'*20, '\n', 'Pred@', str(predAtK) + '\n' + '-'*20, '\n')
    print('Repair accuracy:', round(df_results['isCompiled'].mean(), 3))

def repairProgram(fname, predAtK):
    srcText = open(fname).read()
    srcCodeObj = Code(srcText)
    srcAbsLines = AbstractWrapper.getProgAbstraction(srcCodeObj)
    errSet = ClusterError.getErrSetStr(AllErrs, srcCodeObj)
    lineNums = errLoc(True, srcCodeObj, '', useTracers_errLoc)

    predText, srcErrLines, predErrLines, srcErrAbsLines, predErrAbsLines, isConcretized, isExactMatch  = \
                runPerLine(srcCodeObj, srcText.splitlines(), [], srcAbsLines, [], errSet, lineNums, predAtK)  

    print('-'*20 + '\nOriginal Code\n' + '-'*20 + '\n' + srcText)
    print('-'*20 + '\nMACER\'s Repair\n' + '-'*20 + '\n' + predText)
    print('\nCompiled Successfully? ', Code(predText).getNumErrors() == 0)
    

if __name__ == '__main__':
    predAtK = int(sys.argv[2])
    if sys.argv[1] == 'tracer_single':
        fname = CF.fnameSingleL_Test
    elif sys.argv[1] == 'deepfix':
        fname = CF.fnameDeepFix_Test
    else:
        fname = sys.argv[1]

    if fname.split('.')[-1] == 'csv':
        runTest(fname, predAtK)
    elif fname.split('.')[-1] == 'c':
        repairProgram(fname, predAtK)
    else:
        print("Expected .c file or .csv file as 2nd argument: python testRepair.py <dataset/path-to-test-file> <PredK>")
        error_args()
#endregion
