import re, string, sys, os, time

#from clang.cindex import *
from srcT.Common import ConfigFile as CF, Helper as H
#from srcT.Symbolic.AbstractToken import *

STR_LITERAL = 'TokenKind.LITERAL_'
STR_TYPE = 'TypeKind.'

def guessLiteral(literalType):
    '''Return a hard-code literal of matching type'''
    if literalType == 'INT':
        return '0'
    elif literalType == 'DOUBLE':
        return '0.0'
    elif literalType == 'CHAR':
        return 'c'
    elif literalType == 'STRING':
        return 'str'
    else:
        return STR_LITERAL + literalType
    
def guessTypeKind(typeKind, symbTable):
    '''Check SymbTable, and return a variable having typeKind'''
    return symbTable.getVar_MatchingType(typeKind)

def checkRnnPreProcess(trgtAbs):
    if trgtAbs == '<start>':
        return '{'
    elif trgtAbs == '<stop>':
        return '}'
    return None

def guessConcreteSpell(trgtAbs, symbTable):
    '''Mainly, handle LITERAL or TypeKind abstractions. 
    For rest, simply returns the trgtAbs (as is) - assume its probably punctuation/keyword'''

    spell = None
    if trgtAbs.startswith(STR_LITERAL):
        spell = guessLiteral(trgtAbs[len(STR_LITERAL):])

    elif trgtAbs.startswith(STR_TYPE):
        spell = guessTypeKind(trgtAbs, symbTable)

    else:
        rnnPreProcess = checkRnnPreProcess(trgtAbs)
        
        if rnnPreProcess != None:
            spell = rnnPreProcess
        else: spell = trgtAbs # If neither Literal, nor a type, return the abstract itself (probably punctuation/keyword)

    return spell