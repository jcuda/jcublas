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
    public static final int CUBLAS_GEMM_DFALT = -1;
    public static final int CUBLAS_GEMM_DEFAULT = -1;
    public static final int CUBLAS_GEMM_ALGO0 = 0;
    public static final int CUBLAS_GEMM_ALGO1 = 1;
    public static final int CUBLAS_GEMM_ALGO2 = 2;
    public static final int CUBLAS_GEMM_ALGO3 = 3;
    public static final int CUBLAS_GEMM_ALGO4 = 4;
    public static final int CUBLAS_GEMM_ALGO5 = 5;
    public static final int CUBLAS_GEMM_ALGO6 = 6;
    public static final int CUBLAS_GEMM_ALGO7 = 7;
    public static final int CUBLAS_GEMM_ALGO8 = 8;
    public static final int CUBLAS_GEMM_ALGO9 = 9;
    public static final int CUBLAS_GEMM_ALGO10 = 10;
    public static final int CUBLAS_GEMM_ALGO11 = 11;
    public static final int CUBLAS_GEMM_ALGO12 = 12;
    public static final int CUBLAS_GEMM_ALGO13 = 13;
    public static final int CUBLAS_GEMM_ALGO14 = 14;
    public static final int CUBLAS_GEMM_ALGO15 = 15;
    public static final int CUBLAS_GEMM_ALGO16 = 16;
    public static final int CUBLAS_GEMM_ALGO17 = 17;
    public static final int CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99;
    public static final int CUBLAS_GEMM_DFALT_TENSOR_OP = 99;
    public static final int CUBLAS_GEMM_ALGO0_TENSOR_OP = 100;
    public static final int CUBLAS_GEMM_ALGO1_TENSOR_OP = 101;
    public static final int CUBLAS_GEMM_ALGO2_TENSOR_OP = 102;
    public static final int CUBLAS_GEMM_ALGO3_TENSOR_OP = 103;
    public static final int CUBLAS_GEMM_ALGO4_TENSOR_OP = 104;

    /**
     * Private constructor to prevent instantiation
     */
    private cublasGemmAlgo()
    {
        // Private constructor to prevent instantiation
    }

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUBLAS_GEMM_DEFAULT: return "CUBLAS_GEMM_DEFAULT";
            case CUBLAS_GEMM_ALGO0: return "CUBLAS_GEMM_ALGO0";
            case CUBLAS_GEMM_ALGO1: return "CUBLAS_GEMM_ALGO1";
            case CUBLAS_GEMM_ALGO2: return "CUBLAS_GEMM_ALGO2";
            case CUBLAS_GEMM_ALGO3: return "CUBLAS_GEMM_ALGO3";
            case CUBLAS_GEMM_ALGO4: return "CUBLAS_GEMM_ALGO4";
            case CUBLAS_GEMM_ALGO5: return "CUBLAS_GEMM_ALGO5";
            case CUBLAS_GEMM_ALGO6: return "CUBLAS_GEMM_ALGO6";
            case CUBLAS_GEMM_ALGO7: return "CUBLAS_GEMM_ALGO7";
            case CUBLAS_GEMM_ALGO8: return "CUBLAS_GEMM_ALGO8";
            case CUBLAS_GEMM_ALGO9: return "CUBLAS_GEMM_ALGO9";
            case CUBLAS_GEMM_ALGO10: return "CUBLAS_GEMM_ALGO10";
            case CUBLAS_GEMM_ALGO11: return "CUBLAS_GEMM_ALGO11";
            case CUBLAS_GEMM_ALGO12: return "CUBLAS_GEMM_ALGO12";
            case CUBLAS_GEMM_ALGO13: return "CUBLAS_GEMM_ALGO13";
            case CUBLAS_GEMM_ALGO14: return "CUBLAS_GEMM_ALGO14";
            case CUBLAS_GEMM_ALGO15: return "CUBLAS_GEMM_ALGO15";
            case CUBLAS_GEMM_ALGO16: return "CUBLAS_GEMM_ALGO16";
            case CUBLAS_GEMM_ALGO17: return "CUBLAS_GEMM_ALGO17";
            case CUBLAS_GEMM_DEFAULT_TENSOR_OP: return "CUBLAS_GEMM_DEFAULT_TENSOR_OP";
            case CUBLAS_GEMM_ALGO0_TENSOR_OP: return "CUBLAS_GEMM_ALGO0_TENSOR_OP";
            case CUBLAS_GEMM_ALGO1_TENSOR_OP: return "CUBLAS_GEMM_ALGO1_TENSOR_OP";
            case CUBLAS_GEMM_ALGO2_TENSOR_OP: return "CUBLAS_GEMM_ALGO2_TENSOR_OP";
            case CUBLAS_GEMM_ALGO3_TENSOR_OP: return "CUBLAS_GEMM_ALGO3_TENSOR_OP";
            case CUBLAS_GEMM_ALGO4_TENSOR_OP: return "CUBLAS_GEMM_ALGO4_TENSOR_OP";
        }
        return "INVALID cublasGemmAlgo: "+n;
    }
}

