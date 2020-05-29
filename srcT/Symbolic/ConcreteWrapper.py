from srcT.Symbolic import ConcreteToken, CigarSrcTrgt
from srcT.Symbolic import AbstractWrapper
from srcT.Common import ConfigFile as CF, Helper as H

import traceback
import collections, sys, edlib, re, traceback
from clang.cindex import *

# --- Helper ---
def printIndicesSpell(cst_indices, trgtSpells):
    print('trgtToken =',trgtSpells)
    print(cst_indices, '\n')

def printSrcTokenAbs(cst_indices, srcTrgtCigar):
    if cst_indices.iTrgtAbs < len(srcTrgtCigar.trgtAbs): print('trgtAbs  =', srcTrgtCigar.trgtAbs[cst_indices.iTrgtAbs])
    if cst_indices.iSrcAbs < len(srcTrgtCigar.srcLine): print('srcTokenSpell =',srcTrgtCigar.srcLine[cst_indices.iSrcAbs])
    if cst_indices.iSrcAbs < len(srcTrgtCigar.srcAbs): print('srcTokenAbs =', srcTrgtCigar.srcAbs[cst_indices.iSrcAbs])

# --- The 4 stooges (Equal, Ins, Del and Repl) ---
def handleEquality(cst_indices):
    '''If equal, add currToken spell. And increment both source & target abs counters by numAbs'''
    trgtToken = cst_indices.getCurr_SrcLine()
    cst_indices.incSrc(trgtToken, appendOther=False)  # Since both source and target consumed
    cst_indices.incTrgt(trgtToken, appendOther=False) # Don't append source or target with blanks

    return [trgtToken]

def handleDeletion(cst_indices, appendOther=True):
    '''If deletion, don't add any target spell. Just increment source abs counters by numAbs'''
    trgtToken = cst_indices.getCurr_SrcLine()
    cst_indices.incSrc('', appendOther) # Target spell is empty (deletion), but append Source (deletion consumes source)

    return [] # Ignore trgtTokens, since Deletion. But consume that many numAbs

def handleInsertion(cst_indices, appendOther=True):
    '''If insert, guess and append the spelling of predicted Abs. Inc only target-abs counter by 1'''
    trgtToken = ConcreteToken.guessConcreteSpell(cst_indices.getCurr_TrgtAbs(), cst_indices.srcTrgtCigar.symbTable)
    cst_indices.incTrgt(trgtToken, appendOther) # Consume one target Abs (hence appendOther with blank)

    return [trgtToken] # Consumed 1 target abstract token

def handleReplacement(cst_indices):
    '''If Replace, perform Delete + Insert'''
    delTrgtTokens = handleDeletion (cst_indices, appendOther=False) # Since both source and target consumed
    insTrgtTokens = handleInsertion(cst_indices, appendOther=False) # Don't append source or target with blanks

    # Return "inserted" target tokens (since deleted ones are useless)
    # And the number of abstract tokens deleted, to inc freq count (since just one token inserted anyways)
    return insTrgtTokens # Consume 1 target abstract token

# --- Concretize ---
def concretizeCToken(cst_indices):
    callFunc = None
    #printSrcTokenAbs(cst_indices)

    if cst_indices.compareOp == '=': 
        callFunc = handleEquality

    elif cst_indices.compareOp == 'D': 
        callFunc = handleDeletion
        
    elif cst_indices.compareOp == 'I':
        callFunc = handleInsertion 

    elif cst_indices.compareOp == 'X':
        callFunc = handleReplacement

    else:
        raise Exception('No clue on how to handle this compareOp: {}'.format(cst_indices.compareOp))
    
    trgtTokens = callFunc(cst_indices)
    cst_indices.decFreq() # Consumed currNumAbs (matched) abstract tokens
    cst_indices.srcTrgtCigar.trgtLine.extend(trgtTokens)  

    isConcretized = True
    for trgtToken in trgtTokens:
        if trgtToken is None or ConcreteToken.STR_LITERAL in trgtToken or ConcreteToken.STR_TYPE in trgtToken:
            isConcretized = False
    return isConcretized

def concretizeLine(srcTrgtCigar):
    isConcretized = True
    cst_indices = CigarSrcTrgt.CST_Indices(srcTrgtCigar) # For every CToken, there can be one or more AbsTokens

    for freq, compareOp in re.findall('(\d+)(.)', srcTrgtCigar.cigar): # Cigar eg: 13=1I2=1X5=
        # Alternating Frequency and Compare-Operator (=, I, D)
        cst_indices.freq, cst_indices.compareOp = int(freq), compareOp

        while cst_indices.freq>0: # 'freq' number of times, do what the compareOp says
            # Handle the conversion of currCToken to spell
            tempIsConcretized = concretizeCToken(cst_indices)            
            isConcretized = isConcretized and tempIsConcretized # Don't use "and concretizeCToken(cst_indices)" - shortHand issue!

            if cst_indices.freq == freq:
                raise Exception('Cigar not consumed. Freq is the same. Infinite loop!')  
            else:
                freq = cst_indices.freq
            #printIndicesSpell(cst_indices, trgtTokens)        

    return isConcretized

def attemptConcretization(srcCodeObj, lineNum, predAbsLine):
    try:
        srcLine, srcAbsLine, symbTable = AbstractWrapper.getAbstractAtLine(srcCodeObj, lineNum)
        cigar = CigarSrcTrgt.lineUp_SrcTrgAbs(srcAbsLine, predAbsLine)
        srcTrgtCigar = CigarSrcTrgt.CST_Params(-1, srcLine, srcAbsLine, predAbsLine, cigar, symbTable)

        # Try to concretize the Target Abstraction
        isConcretized = concretizeLine(srcTrgtCigar)
        return srcTrgtCigar.trgtLine, isConcretized

    # If not possible, note down the failure cases
    except Exception as e:
        exception = repr(e)
        traceback.print_exc()
        return exception, False