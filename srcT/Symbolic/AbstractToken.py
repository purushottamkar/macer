import re, string, sys, os, time

from clang.cindex import *
from srcT.Common import ConfigFile as CF, Helper as H
from srcT.DataStruct.Code import Code 

class AbstractToken:
    def __init__(self, name, spell, cToken):
        self.name = name
        self.spell = spell
        self.cToken = cToken

    def __str__(self):
        return self.name

class CToken: 
    ''' Given a clangToken, stores the relevant info in an object, and a list of derived abstractTokens'''
    def __init__(self, clangToken, codeObj, flagIsDirective=False):
        self.codeID = codeObj.codeID
        self.clangToken = clangToken
        self.cursor = Cursor.from_location(codeObj.getTU(), clangToken.location)
        self.spell = clangToken.spelling
        self.lineNum = clangToken.location.line
        self.kind = clangToken.kind
        self.cursorType = self.cursor.type.kind 

        self.flagIsDirective = False
        if flagIsDirective or self.spell == '#': # Set to true if '#' token found now, or previously in the same line
            self.flagIsDirective = True # indicating directive decl in the line (#include<>)
        
        self.abstractTokens = []
        
    def __str__(self):
        return H.joinList(self.abstractTokens, ' ')

    def addAbstract(self, name, spell):
        self.abstractTokens.append(AbstractToken(str(name), spell, self))

    def extractFormatSpec(self, strLit):
        patternFormat = r'\%[-+]?[\d]*(?:\.[\d]*)?(?:lli|lld|llu|ld|li|lf|Lf|lu|hi|hu|d|c|e|E|f|g|G|i|l|o|p|s|u|x|X|n)'
        patternPunct  = r'\\(?:a|b|f|n|r|t|v|\\|\'|\"|\?)'
        patternSplit  = '(' + patternFormat + '|' + patternPunct + ')'
        return re.findall(patternSplit, self.spell)

    def getAbstractIdentifier(self, symbTable):
        '''If Identifier, then add the type of identifier as Abstract token (except for special cases)'''
        if self.flagIsDirective: # If directive declaration (#include<>), add actual spellings to abstraction (and not invalid-types)
            self.addAbstract(self.spell, self.spell)
        elif self.spell in CF.IncludeIdentifiers: # Handle specials like printf
            self.addAbstract(self.spell, self.spell)
        else:  # All other cursorTypes
            #print '-getAbstractIdentifier-\n', self.spell, self.cursorType
        
            symbTable.insertToken(self.spell, self.cursor) # Check & Add unknown variable/func declaration to Symbol-Table
            symbTypes = symbTable.lookup(self.spell) # try to fetch type from sybmTable

            if len(symbTypes)>0: # If lookup success, add the symbType as the abstraction
                list(map(self.addAbstract, symbTypes, [self.spell]*len(symbTypes)) )
                # Add self.spell as Concretization of all AbstractTypes

                # Log error in case SymbTable and Clang differ in claimed Type
                if len(symbTypes)==1 and self.cursorType!=TypeKind.INVALID and self.cursorType!=TypeKind.FUNCTIONPROTO:
                    # Unless the type is INVALID or FUNCTION
                    if symbTypes[0]!=self.cursorType:
                        H.errorLog([['CodeID', self.codeID], ['AbstractToken SymbTab & Clang mismatch type', str(symbTypes[0]) +' and '+ str(self.cursorType)], ['lineNum', self.lineNum], ['spell', self.spell]])
                    
            else: # Otherwise, If symbTable doesn't have the type, insert the cursorType (probably INVALID type)
                self.addAbstract(self.cursorType, self.spell)      
                
    
    def getAbstractLiteral(self):
        flagIsString = False
        quotes = ['\'', '"']

        if self.cursorType == TypeKind.CONSTANTARRAY:
            flagIsString = True
        
        elif len(self.spell)>=2 and self.spell[0] in quotes and self.spell[-1] in quotes:
            flagIsString = True
        
        if flagIsString: # TypeKind.CONSTANTARRAY or TypeKind.INT with single quotes - char or Invalids with double quotes
            self.addAbstract(self.spell[0], self.spell[0]) # Add First Quote

            intermediateStr = self.spell[1:-1]
            if len(intermediateStr) > 0: # Ignore 0 length LITERAL, to differentiate those cases when nothing exists inside quotes
                formatSpecs = self.extractFormatSpec(intermediateStr) # If String, abstract format spec (%d), special chars,... 

                if len(formatSpecs)>0: # If format specifiers present, add them instead of Char/String
                    list(map(self.addAbstract, formatSpecs, formatSpecs))
                elif len(intermediateStr) == 1: # Character: Otherwise, if no formatSpecs
                    self.addAbstract(str(self.kind) + '_CHAR', intermediateStr)  #Add a placeholder Literal_Char
                else: # String - if len(intermediateStr) >= 1
                    self.addAbstract(str(self.kind) + '_STRING', intermediateStr) # Else, add a placeholder Literal_String
                
            self.addAbstract(self.spell[-1], self.spell[-1]) # Add Last Quote
            
        elif isInt(self.spell): # If actually an integer literal
            self.addAbstract(str(self.kind) + '_INT', self.spell ) # Add a placeholder Literal_Int
        elif isFloat(self.spell): # If actually a float literal
            self.addAbstract(str(self.kind) + "_DOUBLE", self.spell) # Add a placeholder Literal_Int
        else: # If neither String, nor int/float: add cursorType (can't abstract - mostly Invalid)
            self.addAbstract(str(self.kind) +'_'+ str(self.cursorType), self.spell) 
            if self.cursorType!=TypeKind.INVALID: # Log the "special" type of Literal (unless its INVALID)
                H.errorLog([['CodeID', self.codeID], ['AbstractToken new literal-type', str(self.kind) +'_'+ str(self.cursorType)],
                ['lineNum', self.lineNum], ['spell', self.spell]])

    def getAbstractPunct(self, symbTable):
        if self.spell == '{':
            self.addAbstract('{', self.spell) 
            #self.addAbstract('<start>', self.spell) 
        elif self.spell == '}':
            self.addAbstract('}', self.spell)   
            #self.addAbstract('<stop>', self.spell)   
        else:
            self.addAbstract(self.spell, self.spell) # Punctuation => Add the spelling as it is
        symbTable.checkBlockLevel(self.spell, self.cursor) # Check if '{' or '}'

    def getAbstractTokens(self, symbTable):
        if self.kind == TokenKind.COMMENT:
            pass # Don't add any abstract tokens for comments
        elif self.kind == TokenKind.PUNCTUATION: 
            self.getAbstractPunct(symbTable)    
        elif self.kind == TokenKind.KEYWORD:
            self.addAbstract(self.spell, self.spell) # Keyword => Retain as it is
        elif self.kind == TokenKind.IDENTIFIER:
            self.getAbstractIdentifier(symbTable) # Lookup symbTable, or add the cursorType
        elif self.kind == TokenKind.LITERAL:
            self.getAbstractLiteral() # Figure out what kind of Literal, and format specs

        return self.abstractTokens, self.flagIsDirective

def isInt(stri):
    try:
        int(stri)
        return True
    except ValueError:
        return False

def isFloat(stri):
    try:
        float(stri)
        return True
    except ValueError:
        return False

