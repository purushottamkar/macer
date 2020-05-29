import pandas as pd
from srcT.Common import ConfigFile as CF

def addDummyCols(cols):
    df = pd.read_csv(CF.fnameDeepFix_Test, encoding="ISO-8859-1")

    if 'id' not in df.columns:
        df['id'] = df['code_id']
        del df['code_id']

    if 'sourceText' not in df.columns:
        df['sourceText'] = df['code']
        del df['code']

    df['targetText'] = ""

    df['sourceLineText'] = ""
    df['targetLineText'] = ""

    df['sourceLineAbs'] = ""
    df['targetLineAbs'] = ""

    df['lineNums_Text'] = ""

    df.to_csv(CF.fnameDeepFix_Test, index=False)
    # sourceAbs	targetAbs	sourceLineText	targetLineText	sourceLineAbs	targetLineAbs	lineNums_Text	lineNums_Abs	diffText_ins	diffText_del	diffText_repl	diffAbs_ins	diffAbs_del	errSet_diffs	sourceErrorPrutor	sourceErrorClangParse	ErrSet	class	newErrSet


if __name__=='__main__':    
    cols = []
    addDummyCols(cols)
