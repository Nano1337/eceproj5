# % Wrapper class to solve sparse systems using the Biconjugate Gradient method compiled in a dll
# % EECE 8395: Engineering for Surgery
# % Fall 2023
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu

from ctypes import*
import numpy as np
# Instructions for recompiling bicg.dll using ming++ or g++
# #C:\path\to\mingw64\bin\x86_64-w64-mingw32-g++.exe -o IterativeSparseSolver.dll -mdll -static-libstdc++ -static BICG.cpp
# #put the resulting dll file in the same directory as GraphCut.py
# on Mac using clang++ or g++ from XCode
# g++ -dynamiclib -DNDEBUG -o IterativeSparseSolver.dylib BICG.cpp
# A similar approach with different compile options should permit creating static .o library for linux using g++

class bicg:
    def __init__(self, dllpath='C:\\Users\\noblejh\\Box Sync\\2023_EECE8395_VISE\\EECE8395Code\\Module6_Modeling\\Demo\\IterativeSparseSolver.dll'):
        self.mydll = cdll.LoadLibrary(dllpath)

    def Solve(self, r, c, I, J, V, b, tol = 1e-5, maxiter=1000, itol=1):
        n = np.size(I)
        x = np.random.default_rng(0).normal(size=c)
        self.mydll.LinBCGSolve(c_int32(r), c_int32(c), c_int32(n),
                               I.ctypes.data_as(POINTER(c_int32)),
                               J.ctypes.data_as(POINTER(c_int32)),
                               V.ctypes.data_as(POINTER(c_double)),
                               b.ctypes.data_as(POINTER(c_double)),
                               x.ctypes.data_as(POINTER(c_double)),
                               c_double(tol), c_int32(maxiter), c_int32(itol))
        return x

    def SolveA(self, r, c, A, b, tol = 1e-5, maxiter=1000, itol=1):
        n = np.size(A.data)
        x = np.random.default_rng(0).normal(size=c)
        self.mydll.LinBCGSolveA(c_int32(r), c_int32(c), c_int32(n),
                               A.indptr.ctypes.data_as(POINTER(c_int32)),
                               A.indices.ctypes.data_as(POINTER(c_int32)),
                               A.data.ctypes.data_as(POINTER(c_double)),
                               b.ctypes.data_as(POINTER(c_double)),
                               x.ctypes.data_as(POINTER(c_double)),
                               c_double(tol), c_int32(maxiter), c_int32(itol))
        return x
