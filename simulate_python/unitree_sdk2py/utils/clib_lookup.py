import os
import ctypes

_IS_WINDOWS = os.name == "nt"

clib = None if _IS_WINDOWS else ctypes.CDLL(None, use_errno=True)

def CLIBCheckError(ret, func, args):
    if ret < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return ret

def CLIBLookup(name, resType, argTypes):
    if clib is None:
        raise NotImplementedError("C library lookups are not supported on Windows")

    func = clib[name]
    func.restype = resType
    func.argtypes = argTypes
    func.errcheck = CLIBCheckError
    return func
