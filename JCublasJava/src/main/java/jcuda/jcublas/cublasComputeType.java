/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda
 *
 * Copyright (c) 2010-2020 Marco Hutter - http://www.jcuda.org
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
 * <pre>
 * Enum for compute type
 *
 * - default types provide best available performance using all available hardware features
 *   and guarantee internal storage precision with at least the same precision and range;
 * - _PEDANTIC types ensure standard arithmetic and exact specified internal storage format;
 * - _FAST types allow for some loss of precision to enable higher throughput arithmetic.
 * </pre>
 */
public class cublasComputeType
{
    /**
     * half - default 
     */
    public static final int CUBLAS_COMPUTE_16F = 64;
    /**
     * half - pedantic 
     */
    public static final int CUBLAS_COMPUTE_16F_PEDANTIC = 65;
    /**
     * float - default 
     */
    public static final int CUBLAS_COMPUTE_32F = 68;
    /**
     * float - pedantic 
     */
    public static final int CUBLAS_COMPUTE_32F_PEDANTIC = 69;
    /**
     * float - fast, allows down-converting inputs to half or TF32 
     */
    public static final int CUBLAS_COMPUTE_32F_FAST_16F = 74;
    /**
     * float - fast, allows down-converting inputs to bfloat16 or TF32 
     */
    public static final int CUBLAS_COMPUTE_32F_FAST_16BF = 75;
    /**
     * float - fast, allows down-converting inputs to TF32 
     */
    public static final int CUBLAS_COMPUTE_32F_FAST_TF32 = 77;
    /**
     * double - default 
     */
    public static final int CUBLAS_COMPUTE_64F = 70;
    /**
     * double - pedantic 
     */
    public static final int CUBLAS_COMPUTE_64F_PEDANTIC = 71;
    /**
     * signed 32-bit int - default 
     */
    public static final int CUBLAS_COMPUTE_32I = 72;
    /**
     * signed 32-bit int - pedantic 
     */
    public static final int CUBLAS_COMPUTE_32I_PEDANTIC = 73;

    /**
     * Private constructor to prevent instantiation
     */
    private cublasComputeType()
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
            case CUBLAS_COMPUTE_16F: return "CUBLAS_COMPUTE_16F";
            case CUBLAS_COMPUTE_16F_PEDANTIC: return "CUBLAS_COMPUTE_16F_PEDANTIC";
            case CUBLAS_COMPUTE_32F: return "CUBLAS_COMPUTE_32F";
            case CUBLAS_COMPUTE_32F_PEDANTIC: return "CUBLAS_COMPUTE_32F_PEDANTIC";
            case CUBLAS_COMPUTE_32F_FAST_16F: return "CUBLAS_COMPUTE_32F_FAST_16F";
            case CUBLAS_COMPUTE_32F_FAST_16BF: return "CUBLAS_COMPUTE_32F_FAST_16BF";
            case CUBLAS_COMPUTE_32F_FAST_TF32: return "CUBLAS_COMPUTE_32F_FAST_TF32";
            case CUBLAS_COMPUTE_64F: return "CUBLAS_COMPUTE_64F";
            case CUBLAS_COMPUTE_64F_PEDANTIC: return "CUBLAS_COMPUTE_64F_PEDANTIC";
            case CUBLAS_COMPUTE_32I: return "CUBLAS_COMPUTE_32I";
            case CUBLAS_COMPUTE_32I_PEDANTIC: return "CUBLAS_COMPUTE_32I_PEDANTIC";
        }
        return "INVALID cublasComputeType: "+n;
    }
}

