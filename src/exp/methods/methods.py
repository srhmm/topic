from exp.util.ty import ValidMethodType
from exp.methods.methods_changes import TopChangesContMethod, OracleChangesContMethod, TopChangesTimeMethod, \
    OracleChangesTimeMethod, JCIPCMethod, CDNODMethod, UTIGSPMethod, GIESMethod, JCIFCIMethod, JPCMCIMethod, \
    MGLINGAMMethod
from exp.methods.methods_cont import TopContMethod, OracleContMethod, PCMethod, GESMethod, FCIMethod, CAMMethod, GLOBEMethod, \
    SCOREMethod, NOTEARSMethod, ICALINGAMMethod, DirectLINGAMMethod, GranDAGMethod, CAMUVMethod, DAGGNNMethod, \
    RESITMethod, GGESMethod, LINGAMMethod, SCORE2Method, NOGAMMethod, DASMethod, CAM2Method, GESNLMethod, GESMargMethod, \
    R2SORTMethod, VARSORTMethod, RANDSORTMethod
from exp.methods.methods_time import TopTimeMethod, OracleTimeMethod, VarLINGAMMethod, PCMCIPLUSMethod, DyNOTEARSMethod, \
    RhinoMethod, CDNODTMethod


def id_to_method(idf: int):
    METHODS = [
        TopContMethod, OracleContMethod,
        TopTimeMethod, OracleTimeMethod,
        TopChangesContMethod, OracleChangesContMethod,
        TopChangesTimeMethod, OracleChangesTimeMethod,
        PCMethod, GESMethod, GGESMethod, FCIMethod, CAMMethod, CAMUVMethod, RESITMethod, GLOBEMethod, SCOREMethod,
        ICALINGAMMethod, DirectLINGAMMethod, LINGAMMethod, NOTEARSMethod, GranDAGMethod, DAGGNNMethod,
        VarLINGAMMethod, PCMCIPLUSMethod, DyNOTEARSMethod, RhinoMethod,
        JPCMCIMethod, JCIPCMethod, JCIFCIMethod, CDNODMethod, CDNODTMethod, UTIGSPMethod, GIESMethod, MGLINGAMMethod,
        DASMethod, NOGAMMethod, SCORE2Method, CAM2Method, GESNLMethod, GESMargMethod,R2SORTMethod, VARSORTMethod, RANDSORTMethod
    ]
    for method in METHODS:
        if idf == method.ty().value:
            return method
    raise ValueError("no method for this ID")


def ids_continuous_methods_extended(comp):
    if not comp:
        return [
            ValidMethodType.TopCont.value ]
    return [
        ValidMethodType.TopCont.value,
        ValidMethodType.OracleCont.value,
        ValidMethodType.PC.value,
        ValidMethodType.GES.value,
        ValidMethodType.FCI.value,
        ValidMethodType.CAM.value,
        ValidMethodType.CAM_UV.value,
        ValidMethodType.SCORE.value,
        ValidMethodType.RESIT.value,
        ValidMethodType.ICA_LINGAM.value,
        ValidMethodType.Direct_LINGAM.value,
        ValidMethodType.GLOBE.value,
        ValidMethodType.NOTEARS.value
    ]


def ids_continuous_methods(comp):
    if not comp:
        return [
            ValidMethodType.TopCont.value ]
    return [
        ValidMethodType.SCORE2.value,
        ValidMethodType.DAS.value,
        ValidMethodType.NOGAM.value,
        ValidMethodType.TopCont.value,
        ValidMethodType.R2SORT.value,
        ValidMethodType.VARSORT.value,
        ValidMethodType.RANDSORT.value,
       # ValidMethodType.PC.value,
       #ValidMethodType.CAM2.value,
       # ValidMethodType.GES.value,
       # ValidMethodType.GES_CV.value,
      #  ValidMethodType.GES_MARG.value,
       # ValidMethodType.CAM.value,
       # ValidMethodType.RESIT.value,
      #  ValidMethodType.SCORE.value,
      #  ValidMethodType.ICA_LINGAM.value,
      #  ValidMethodType.Direct_LINGAM.value,
        #ValidMethodType.GLOBE.value,
      #  ValidMethodType.NOTEARS.value
    ]


def ids_time_continuous_methods(comp):
    if not comp:
        return [
            ValidMethodType.TopTime.value]
    return [
        ValidMethodType.TopTime.value,
        ValidMethodType.PCMCI.value,
        ValidMethodType.DyNOTEARS.value,
        ValidMethodType.CDNODT.value,
        ValidMethodType.VarLINGAM.value]


def ids_change_methods(comp):
    if not comp:
        return [
            ValidMethodType.TopChangesCont.value]
    return [
        ValidMethodType.TopChangesCont.value,
        ValidMethodType.MGLINGAM.value,
        ValidMethodType.UTIGSP.value,
        ValidMethodType.CDNOD.value,
        ValidMethodType.JCI_PC.value,
        ValidMethodType.JCI_FCI.value
    ]


def ids_time_change_methods(comp):
    if not comp:
        return [
            ValidMethodType.TopChangesTime.value]
    return [
        ValidMethodType.TopChangesTime.value,
        ValidMethodType.JPCMCI.value,
        ValidMethodType.PCMCI.value,
        ValidMethodType.DyNOTEARS.value,
        ValidMethodType.VarLINGAM.value
    ]
