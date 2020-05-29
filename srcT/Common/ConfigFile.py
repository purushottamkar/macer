import os

# ------------ Clang Args ----------------- #
ClangArgs = ['-static', '-Wall', '-funsigned-char', '-Wno-unused-result', '-O', '-Wextra', '-std=c99', "-I/usr/lib/clang/6.0/include", '-I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/']

# ------------ PATHS ---------------------- # 

full_path = os.path.realpath(__file__)
currPath, configFilename = os.path.split(full_path)

pathHome = os.path.expanduser('~') + '/'
pathBase = currPath + '/../../'

pathData =  pathBase + "data/"
pathInput = pathData + 'input/' 
pathOut = pathData + 'output/'

fname_newErrIDs = pathInput + 'newErrorIDs.csv' # improved error cluster IDs - using Clang LLVM
filenameErrorLog = pathOut + "err.log"

# --------- Abstraction ------------#
fnameIncludeIdent = pathInput + 'include_Identifiers.csv'
IncludeIdentifiers = [line.strip() for line in open(fnameIncludeIdent).read().split('\n') 
                                    if line.strip()!=''] 

# --------- Tracer Source-Target pairs ------------#
pathTracer = pathBase + "tracer/"
pathSingleL = pathTracer + 'data/dataset/singleL/'
fnameSingleL_Train = pathSingleL + 'singleL_Train+Valid.csv'
fnameSingleL_Test = pathSingleL + 'singleL_Test.csv'

# --------- DeepFix dataset ------------#
pathDeepFix = pathBase + "prutor-deepfix-09-12-2017/"
fnameDeepFix_Test = pathDeepFix + 'deepfix_test.csv'
