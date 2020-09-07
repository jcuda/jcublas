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

/**Enum for default math mode/tensor operation*/
public class cublasMath
{
    public static final int CUBLAS_DEFAULT_MATH = 0;
    /** deprecated, same effect as using CUBLAS_COMPUTE_32F_FAST_16F, will be removed in a future release */
    @Deprecated
    public static final int CUBLAS_TENSOR_OP_MATH = 1;
    /** same as using matching _PEDANTIC compute type when using cublas-T-routine calls or cublasEx() calls with
         cudaDataType as compute type */
    public static final int CUBLAS_PEDANTIC_MATH = 2;
    /** allow accelerating single precision routines using TF32 tensor cores */
    public static final int CUBLAS_TF32_TENSOR_OP_MATH = 3;
    /** flag to force any reductions to use the accumulator type and not output type in case of mixed precision routines
         with lower size output type */
    public static final int CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16;

    /**
     * Private constructor to prevent instantiation
     */
    private cublasMath()
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
            case CUBLAS_DEFAULT_MATH: return "CUBLAS_DEFAULT_MATH";
            case CUBLAS_TENSOR_OP_MATH: return "CUBLAS_TENSOR_OP_MATH";
            case CUBLAS_PEDANTIC_MATH: return "CUBLAS_PEDANTIC_MATH";
            case CUBLAS_TF32_TENSOR_OP_MATH: return "CUBLAS_TF32_TENSOR_OP_MATH";
            case CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION: return "CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION";
        }
        return "INVALID cublasMath: "+n;
    }
}

