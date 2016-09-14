/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda
 *
 * Copyright (c) 2010-2016 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
package jcuda.jcublas;

/**
 * Different GEMM algorithms
 */
public class cublasGemmAlgo
{
    public static final int CUBLAS_GEMM_DFALT         = -1;
    public static final int CUBLAS_GEMM_ALGO0         =  0;
    public static final int CUBLAS_GEMM_ALGO1         =  1;
    public static final int CUBLAS_GEMM_ALGO2         =  2;
    public static final int CUBLAS_GEMM_ALGO3         =  3;
    public static final int CUBLAS_GEMM_ALGO4         =  4;

    /**
     * Returns the String identifying the given cublasGemmAlgo
     *
     * @param n The cublasGemmAlgo
     * @return The String identifying the given cublasGemmAlgo
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUBLAS_GEMM_DFALT: return "CUBLAS_GEMM_DFALT";
            case CUBLAS_GEMM_ALGO0: return "CUBLAS_GEMM_ALGO0";
            case CUBLAS_GEMM_ALGO1: return "CUBLAS_GEMM_ALGO1";
            case CUBLAS_GEMM_ALGO2: return "CUBLAS_GEMM_ALGO2";
            case CUBLAS_GEMM_ALGO3: return "CUBLAS_GEMM_ALGO3";
            case CUBLAS_GEMM_ALGO4: return "CUBLAS_GEMM_ALGO4";
        }
        return "INVALID cublasGemmAlgo: " + n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cublasGemmAlgo()
    {
        // Private constructor to prevent instantiation.
    }

}
