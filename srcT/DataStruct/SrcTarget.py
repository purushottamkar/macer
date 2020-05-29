import base64, collections, traceback
from srcT.Common import ConfigFile as CF, Helper as H
from Code import Code

class SrcTarget:    
        
    def __init__(self, row):
        self.row = row                
        indent = False

        self.event_name = row[0]
        self.assignID = row[1]
        self.srcCodeID = int(row[2])

        self.srcCodeText = base64.b64decode(row[3])
        self.sourceCodeObj = Code(self.srcCodeText, indent=indent, codeID=self.srcCodeID)

        self.sourceTime = row[4]
        self.sourceErrors = row[5]


        targetData = row[6]
        self.targetCodeID = None
        self.targetCodeObj = None

        if targetData != None:
            self.targetCodeID, self.targetTime, targetContents = targetData.split(',')
            self.targetCodeID = int(self.targetCodeID)
            self.targetCodeText = base64.b64decode(targetContents)
            self.targetCodeObj = Code(self.targetCodeText, indent=indent, codeID=self.targetCodeID)

    