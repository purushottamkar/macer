from srcT.Symbolic import AbstractToken 
import traceback
from srcT.Common import ConfigFile as CF, Helper as H

import collections, sys
from clang.cindex import *

class SymbTable:
    def __init__(self):
        self.dictBlockVarType = {} # Spelling -> Type mapping (for user-defined vars/funcs): In case Clang-goofUps
        self.blockLevel = 0 # Current block level for maintaining symbol-table (each func/block statement inc/dec it)

        self.dictBlockVarType[0]={}

    def checkBlockLevel(self, spell, cursor):
        if spell == '{':
            self.incBlock()
        elif spell == '}':
            self.decBlock()
        #elif cursor.kind == CursorKind.FUNCTION_DECL and cursor.type.kind == TypeKind.FUNCTIONPROTO:
        #    self.incBlock()

    def insertToken(self, spell, cursor):
        myType = cursor.type.kind
        if myType == TypeKind.INVALID or len(self.lookup(spell)) > 0: # Ignore INVALID types and those spell already in table
            return # don't add them to Symb Table!
        elif myType == TypeKind.FUNCTIONPROTO: # If function, add the return/argument types as well to Abstraction
            returnTypes, argTypes = [], []
            #returnTypes = [cursor.result_type.kind]
            #returnTypes = ['<RETURN_TYPE>'] + returnTypes + ['</RETURN_TYPE>']
            #argTypes = [arg.type.kind for arg in cursor.get_arguments()]
            #argTypes = ['<'+str(CursorKind.PARM_DECL)+'>'] + argTypes + ['</'+str(CursorKind.PARM_DECL)+'>']
            self.dictBlockVarType[self.blockLevel][spell] = ['TypeKind.FUNCTIONCALL'] + returnTypes + argTypes 
        else:
            self.dictBlockVarType[self.blockLevel][spell] = [myType]

        #print '--Symb Table: Insert --\n',spell,self.dictBlockVarType

    def lookup(self, spell):
        #print '--Symb Table: Lookup--\n',spell, self.dictBlockVarType

        for level in range(self.blockLevel, -1, -1):
            if spell in self.dictBlockVarType[level]:
                return self.dictBlockVarType[level][spell]

        return []
    
    def getVar_MatchingType(self, typeKind):
        '''Fetch the first variable of matching typeKind'''
        
        for level in range(self.blockLevel, -1, -1):
            for spell, types in self.dictBlockVarType[level].items():
                if str(types[0]) == typeKind: # If the first typeKind (in list of abstractions) is the same
                    return spell

        return None

    def incBlock(self):
        self.blockLevel += 1
        self.dictBlockVarType[self.blockLevel] = {} # Create empty mapping for new BlockLevel
    
    def decBlock(self):
        self.dictBlockVarType[self.blockLevel] = {} # Delete the mapping from old BlockLevel
        #self.blockLevel -= 1
        if self.blockLevel!=0: # If already at min (block-0), don't dec - probably incorrect brace closing } 
            self.blockLevel -= 1


def getLineAbstraction(codeObj, codeClangTokens, tokenNum, lineNum, symbTable):
    ''' Abstract all clangTokens present in the curr lineNum'''
    flagIsDirective = False
    lineAbs, lineCTokens = [], []
    
    while tokenNum<len(codeClangTokens) and codeClangTokens[tokenNum].location.line == lineNum: 
        # continue until all tokens of this line are exhaused
        cToken = AbstractToken.CToken(codeClangTokens[tokenNum], codeObj, flagIsDirective)
        abstractTokens, flagIsDirective = cToken.getAbstractTokens(symbTable)

        lineAbs.extend(abstractTokens)
        lineCTokens.append(cToken)

        tokenNum += 1
        
    return lineAbs, lineCTokens, tokenNum

def getProgAbstractTokenSymbTab(codeObj, lineNumBreak=None):
    ''' Abstract all clangTokens line by line.
If breakLineNum is provided, run abstraction until lineNumBreak, and return the abstract tokens at that lineNum, along with the Symbol Table until that line '''
    absLines, cTokenLines = [], []
    symbTable = SymbTable()
    codeClangTokens = codeObj.getTokens()
    
    tokenNum = 0 # Current token number being abstracted
    lineNum = 1 # Current line number being abstracted

    while tokenNum < len(codeClangTokens): 
        lineAbs, lineCTokens, tokenNum = getLineAbstraction(codeObj, codeClangTokens, tokenNum, lineNum, symbTable)
        absLines.append(lineAbs)
        cTokenLines.append(lineCTokens)        

        if lineNumBreak != None and lineNum == lineNumBreak:
            return lineAbs, lineCTokens, symbTable
        lineNum += 1       

    # Handle the case lineNum = empty-line at end of code (Eg, forgot a closing brace)
    if lineNumBreak != None and lineNumBreak >= lineNum: # If the lineNumber to break on is beyond the lineNums encountered
        return [], [], symbTable # in program, return empty abstract/C-Tokens 

    absLinesStr = [[str(absToken) for absToken in line] for line in absLines]
    return absLinesStr, cTokenLines, symbTable

def getProgAbstraction(codeObj):
    absLines, cTokenLines, symbTable = getProgAbstractTokenSymbTab(codeObj)
    return absLines

def getAbstractAtLine(codeObj, lineNum):
    srcAbsObjs, srcCTokens, symbTable = getProgAbstractTokenSymbTab(codeObj, lineNum)

    srcAbs = H.stringifyL(srcAbsObjs)
    srcLine = [absObj.spell for cTok in srcCTokens for absObj in cTok.abstractTokens]
    return srcLine, srcAbs, symbTable

def getCTokenLines(codeText):
    codeObj = Code(codeText)
    absLines, cTokenLines, symbTable = getProgAbstractTokenSymbTab(codeObj)
    return cTokenLines

def printProgAbstraction():
    codeText = open(CF.inputPath + 'temp.c').read()
    codeObj = Code(codeText)
    absLines = getProgAbstraction(codeObj)
    for line in absLines:
        print(H.joinList(line, ' '))

def writeTypeKind():
    path = './final_all_noindent_singleL/'      
    nameRead = path + 'subset-srcTrgtPairs.csv'
    nameWrite = path + 'TokenKind.csv'
    headers, lines = H.readCSV(nameRead)
    writeH = ['spell', 'kind', 'cursorTypeKind']
    dictSpell = collections.defaultdict(lambda :{})

    count = 0
    for line in lines:
        srcText = line[headers.index('sourceText')]
        codeObj = Code(srcText)
        for token in codeObj.getTokens():
            cToken = CToken(token, codeObj)

            dictSpell[cToken.spell][str(cToken.kind) + '!@#$%' + str(cToken.cursorType)] = 0

        count += 1
        print(count, line[headers.index('sourceID')])
    
    writeL = [[spell, kindType.split('!@#$%')[0], kindType.split('!@#$%')[1]] 
        for spell in dictSpell for kindType in dictSpell[spell]]
    H.writeCSV(nameWrite, writeH, writeL)

def writeAbstractions():
    path = './final_all_noindent_singleL/'      
    nameRead = path + 'subset-srcTrgtPairs'
    nameWrite = nameRead + '_newAbs'

    headers, lines = H.readCSV(nameRead + '.csv')
    headers += ['NEW_SrcAbs', 'NEW_TrgtAbs']
    writeLines = []

    count = 0
    for line in lines[:10]:
        writeLine = line
        srcText = line[headers.index('sourceText')]
        trgtText = line[headers.index('targetText')]

        for text, hname in zip([srcText, trgtText], ['', '']):
            codeObj = Code(text)
            absLines = getProgAbstraction(codeObj)
            writeLine.append(H.joinLL(absLines))

        count += 1
        print(count, line[headers.index('sourceID')])
        writeLines.append(writeLine)

    H.writeCSV(nameWrite + '.csv', headers, writeLines)

if __name__=='__main__':
    #writeTypeKind()
    printProgAbstraction()
    #writeAbstractions()
