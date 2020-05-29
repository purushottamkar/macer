'''
Created on 29-Jan-2016

@author: umair
'''

import base64, clang.cindex, collections, traceback
from subprocess import Popen, PIPE, STDOUT
from srcT.DataStruct import CompError

from srcT.Common import ConfigFile as CF, Helper as H
from clang.cindex import Config
from clang.cindex import Diagnostic

class Code:
    '''Constructor takes: base64 content and compilation error objects. AutoAddCEs => Add CEs using the TU (clang api) object'''
    index = clang.cindex.Index.create()
        
    def __init__(self, codeText, codeID=None, indent=False):
        '''Needs a codeText (plain text code)'''
        self.codeText = codeText
        self.codeID = codeID
        self.notIndented = indent # True => trigger self.codeIndent() on calling clangParse

        self.tu = None
        self.ces = []
    
    def codeIndent(self):
        '''Pretty format the student's codeText (multi statements into multi lines)'''
        p = Popen(['indent', '-linux'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        self.codeText,err = p.communicate(input=self.codeText.replace('\r','\n'))
        
        if err!='' and err!=None:
            pass 
            #print err
    
    def clangParse(self):
        if self.notIndented: 
            self.codeIndent()
            self.notIndented = False

        try:
            filename = 'temp.c'
            self.tu = Code.index.parse(filename, 
                args=CF.ClangArgs, unsaved_files=[(filename, self.codeText)])
        except Exception:             
            traceback.print_exc()
            print(self.codeText)
    
    def getTU(self):
        if self.tu == None:
            self.clangParse()
            self.addCEsTU()
        
        if self.tu is None: raise Exception('tu is None!')
        return self.tu
    
    def delTU_CEs(self):
        self.tu = None
        self.ces = [] 

    def getTokens(self):
        tu = self.getTU()
        if tu != None:
            return list(self.getTU().cursor.get_tokens())
        else: return None

    def getTokenLines(self):
        allTokens = self.getTokens()
        lineNum, index = 1, 0
        tempLine, tokenLines = [], []

        while index < len(allTokens):
            token = allTokens[index]
            if token.location.line == lineNum:
                tempLine.append(token)
                index += 1 
            else:
                tokenLines.append(tempLine)
                tempLine = []
                lineNum += 1

        if len(tempLine)!=0: # Add the leftover tokens
            tokenLines.append(tempLine)

        return tokenLines

    def getTokenSpellLines(self):
        tokenLines = self.getTokenLines()
        return [[token.spelling for token in line] for line in tokenLines]

    def getTokensAtLine(self, lineNum):
        fileTokens = self.getTokens()
        if fileTokens != None:
            return [t for t in fileTokens if t.location.line==lineNum]
        return None

    def getCEs(self):
        '''Get all compilation errors'''
        if len(self.ces) == 0:
            tu = self.getTU()
        
        return self.ces

    def getCE_lines(self):
        '''Get all line numbers, where compilation error occurred'''
        return list(set([ce.line for ce in self.getSevereErrors()]))


    def getCEsAtLine(self, lineNum):
        if self.getCEs() != None:
            return [ce for ce in self.ces if ce.line==lineNum]
        return None

    def addCEsTU(self):
        if len(self.ces) == 0 and self.tu != None:
            for diag in self.tu.diagnostics:
                ce = CompError.CompError()
                ce.initDiagnostics(diag)
                ce.findTokens(self.tu)
                
                self.ces.append(ce)
                #print ce
    
    def getSevereErrors(self):
        return [ ce for ce in self.getCEs()
                    if ce.severity == Diagnostic.Error or ce.severity == Diagnostic.Fatal
               ]
    
    def getWarnings(self):
        return [ce for ce in self.getCEs() if ce.severity == Diagnostic.Warning]

    def getNumErrors(self):
        return len(self.getSevereErrors())

    def getNumWarnings(self):
        return len(self.getWarnings())

    def cesToString(self):
        return '\n'.join([str(ce) for ce in self.getCEs()])
            
        
    def checkErrLineExists(self, givenCE):
        '''Check if the givenCE exists in this particular, at the level of line (ignore pos, just check for line and msg equivalence)'''
        for ce in self.getCEs():
            if ce.compareErrLine(givenCE):
                return True
        return False
    
    def compareErrsDiffLine(self, codeObj, lineNum):
        '''Given a lineNum, return #errsDiff btw self and codeObj at that lineNum'''
        ces1, ces2 = self.getCEs(), codeObj.getCEs()
        numErrSelf, numErrObj = 0,0
        if ces1 == None or ces2 == None:    return None
        
        lineSelfCEs = collections.defaultdict(lambda: [])
        for ce in self.ces: lineSelfCEs[ce.line].append(ce)
            
        lineObjCEs = collections.defaultdict(lambda: [])
        for ce in codeObj.ces: lineObjCEs[ce.line].append(ce)
        
        if lineNum in lineSelfCEs: numErrSelf = len(lineSelfCEs[lineNum])
        if lineNum in lineObjCEs: numErrObj = len(lineObjCEs[lineNum])
        
        return numErrSelf - numErrObj
            
         
    def compareErrs(self, codeObj):
        ces1, ces2 = self.getCEs(), codeObj.getCEs()
        numErr1, numErr2 = 0,0
        index1, index2 = 0,0
        flag = True
        if ces1 == None or ces2 == None:    return None
        
        while index1 < len(self.ces) or index2 < len(codeObj.ces):
            # Ignore note type of errors (present only in command line compile mode)        
            '''
            if  index1 < len(self.ces) and self.ces[index1].typeErr == 'note': 
                index1 += 1 
                continue             
            
            if index2 < len(codeObj.ces) and codeObj.ces[index2].typeErr == 'note':
                index2 += 1 
                continue
            '''
                        
            # No point in continuing compare if codeObj (or self) ran out of ces. Inc num errs & index (ordering imp - typeErr check needs to be above)
            if index1 >= len(self.ces):
                index2 += 1
                numErr2 += 1
                flag = False
                continue 
            
            if index2 >= len(codeObj.ces):
                index1 += 1
                numErr1 += 1
                flag = False
                continue
            
            if not self.ces[index1].compareErr(codeObj.ces[index2]):
                flag = False
                
            index1 += 1
            index2 += 1
            numErr1 += 1
            numErr2 += 1
            
        return flag, numErr1-numErr2
                                
