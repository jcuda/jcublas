/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda
 *
 * Copyright (c) 2010-2015 Marco Hutter - http://www.jcuda.org
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

import jcuda.CudaException;
import jcuda.JCudaVersion;
import jcuda.LibUtils;
import jcuda.LibUtilsCuda;
import jcuda.LogLevel;
import jcuda.Pointer;
import jcuda.cudaDataType;
import jcuda.runtime.cudaStream_t;

/**
 * Java bindings for CUBLAS, the NVIDIA CUDA BLAS library.
 * <br />
 * This class contains the new CUBLAS API that was introduced
 * with CUDA 4.0, defined in the C header "cublas_v2.h".<br />
 * <br />
 * Most comments are taken from the CUBLAS header file.
 * <br />
 */
public class JCublas2
{
    /**
     * The flag that indicates whether the native library has been
     * loaded
     */
    private static boolean initialized = false;

    /**
     * Whether a CudaException should be thrown if a method is about
     * to set a result code that is not cublasStatus.CUBLAS_STATUS_SUCCESS
     */
    private static boolean exceptionsEnabled = false;

    /* Private constructor to prevent instantiation */
    private JCublas2()
    {
    }

    // Initialize the native library.
    static
    {
        initialize();
    }

    /**
     * Initializes the native library. Note that this method
     * does not have to be called explicitly, since it will
     * be called automatically when this class is loaded.
     */
    public static void initialize()
    {
        if (!initialized)
        {
            String libraryBaseName = "JCublas2-" + JCudaVersion.get();
            String libraryName = 
                LibUtils.createPlatformLibraryName(libraryBaseName);
            LibUtilsCuda.loadLibrary(libraryName);
            initialized = true;
        }
    }



    /**
     * Set the specified log level for the JCublas library.<br />
     * <br />
     * Currently supported log levels:
     * <br />
     * LOG_QUIET: Never print anything <br />
     * LOG_ERROR: Print error messages <br />
     * LOG_TRACE: Print a trace of all native function calls <br />
     *
     * @param logLevel The log level to use.
     */
    public static void setLogLevel(LogLevel logLevel)
    {
        setLogLevelNative(logLevel.ordinal());
    }

    private static native void setLogLevelNative(int logLevel);


    /**
     * Enables or disables exceptions. By default, the methods of this class
     * only return the {@link cublasStatus} from the native method.
     * If exceptions are enabled, a CudaException with a detailed error
     * message will be thrown if a method is about to return a result code
     * that is not cublasStatus.CUBLAS_STATUS_SUCCESS
     *
     * @param enabled Whether exceptions are enabled
     */
    public static void setExceptionsEnabled(boolean enabled)
    {
        exceptionsEnabled = enabled;
    }

    /**
     * If the given result is different to cublasStatus.CUBLAS_STATUS_SUCCESS
     * and exceptions have been enabled, this method will throw a
     * CudaException with an error message that corresponds to the
     * given result code. Otherwise, the given result is simply
     * returned.
     *
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException If exceptions have been enabled and
     * the given result code is not cublasStatus.CUBLAS_STATUS_SUCCESS
     */
    private static int checkResult(int result)
    {
        if (exceptionsEnabled && result != cublasStatus.CUBLAS_STATUS_SUCCESS)
        {
            throw new CudaException(cublasStatus.stringFor(result));
        }
        return result;
    }
    
    //=========================================================================
    // Memory management functions
    
    /**
     * <pre>
     * cublasStatus_t
     * cublasSetVector (int n, int elemSize, const void *x, int incx,
     *                  void *y, int incy)
     *
     * copies n elements from a vector x in CPU memory space to a vector y
     * in GPU memory space. Elements in both vectors are assumed to have a
     * size of elemSize bytes. Storage spacing between consecutive elements
     * is incx for the source vector x and incy for the destination vector
     * y. In general, y points to an object, or part of an object, allocated
     * via cublasAlloc(). Column major format for two-dimensional matrices
     * is assumed throughout CUBLAS. Therefore, if the increment for a vector
     * is equal to 1, this access a column vector while using an increment
     * equal to the leading dimension of the respective matrix accesses a
     * row vector.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasSetVector(
        int n,
        int elemSize,
        Pointer x,
        int incx,
        Pointer devicePtr,
        int incy)
    {
        return checkResult(cublasSetVectorNative(n, elemSize, x, incx, devicePtr, incy));
    }
    private static native int cublasSetVectorNative(
        int n,
        int elemSize,
        Pointer x,
        int incx,
        Pointer devicePtr,
        int incy);


    /**
     * <pre>
     * cublasStatus_t
     * cublasGetVector (int n, int elemSize, const void *x, int incx,
     *                  void *y, int incy)
     *
     * copies n elements from a vector x in GPU memory space to a vector y
     * in CPU memory space. Elements in both vectors are assumed to have a
     * size of elemSize bytes. Storage spacing between consecutive elements
     * is incx for the source vector x and incy for the destination vector
     * y. In general, x points to an object, or part of an object, allocated
     * via cublasAlloc(). Column major format for two-dimensional matrices
     * is assumed throughout CUBLAS. Therefore, if the increment for a vector
     * is equal to 1, this access a column vector while using an increment
     * equal to the leading dimension of the respective matrix accesses a
     * row vector.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasGetVector(
        int n,
        int elemSize,
        Pointer x,
        int incx,
        Pointer y,
        int incy)
    {
        return checkResult(cublasGetVectorNative(n, elemSize, x, incx, y, incy));
    }
    private static native int cublasGetVectorNative(
        int n,
        int elemSize,
        Pointer x,
        int incx,
        Pointer y,
        int incy);


    /**
     * <pre>
     * cublasStatus_t
     * cublasSetMatrix (int rows, int cols, int elemSize, const void *A,
     *                  int lda, void *B, int ldb)
     *
     * copies a tile of rows x cols elements from a matrix A in CPU memory
     * space to a matrix B in GPU memory space. Each element requires storage
     * of elemSize bytes. Both matrices are assumed to be stored in column
     * major format, with the leading dimension (i.e. number of rows) of
     * source matrix A provided in lda, and the leading dimension of matrix B
     * provided in ldb. In general, B points to an object, or part of an
     * object, that was allocated via cublasAlloc().
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or
     *                                ldb <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasSetMatrix(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb)
    {
        return checkResult(cublasSetMatrixNative(rows, cols, elemSize, A, lda, B, ldb));
    }
    private static native int cublasSetMatrixNative(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb);


    /**
     * <pre>
     * cublasStatus_t
     * cublasGetMatrix (int rows, int cols, int elemSize, const void *A,
     *                  int lda, void *B, int ldb)
     *
     * copies a tile of rows x cols elements from a matrix A in GPU memory
     * space to a matrix B in CPU memory space. Each element requires storage
     * of elemSize bytes. Both matrices are assumed to be stored in column
     * major format, with the leading dimension (i.e. number of rows) of
     * source matrix A provided in lda, and the leading dimension of matrix B
     * provided in ldb. In general, A points to an object, or part of an
     * object, that was allocated via cublasAlloc().
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasGetMatrix(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb)
    {
        return checkResult(cublasGetMatrixNative(rows, cols, elemSize, A, lda, B, ldb));
    }
    private static native int cublasGetMatrixNative(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb);


    /**
     * <pre>
     * cublasStatus
     * cublasSetVectorAsync ( int n, int elemSize, const void *x, int incx,
     *                       void *y, int incy, cudaStream_t stream );
     *
     * cublasSetVectorAsync has the same functionnality as cublasSetVector
     * but the transfer is done asynchronously within the CUDA stream passed
     * in parameter.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasSetVectorAsync(
        int n,
        int elemSize,
        Pointer hostPtr,
        int incx,
        Pointer devicePtr,
        int incy,
        cudaStream_t stream)
    {
        return checkResult(cublasSetVectorAsyncNative(n, elemSize, hostPtr, incx, devicePtr, incy, stream));
    }
    private static native int cublasSetVectorAsyncNative(
        int n,
        int elemSize,
        Pointer hostPtr,
        int incx,
        Pointer devicePtr,
        int incy,
        cudaStream_t stream);


    /**
     * <pre>
     * cublasStatus
     * cublasGetVectorAsync( int n, int elemSize, const void *x, int incx,
     *                       void *y, int incy, cudaStream_t stream)
     *
     * cublasGetVectorAsync has the same functionnality as cublasGetVector
     * but the transfer is done asynchronously within the CUDA stream passed
     * in parameter.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasGetVectorAsync(
        int n,
        int elemSize,
        Pointer devicePtr,
        int incx,
        Pointer hostPtr,
        int incy,
        cudaStream_t stream)
    {
        return checkResult(cublasGetVectorAsyncNative(n, elemSize, devicePtr, incx, hostPtr, incy, stream));
    }
    private static native int cublasGetVectorAsyncNative(
        int n,
        int elemSize,
        Pointer devicePtr,
        int incx,
        Pointer hostPtr,
        int incy,
        cudaStream_t stream);


    /**
     * <pre>
     * cublasStatus_t
     * cublasSetMatrixAsync (int rows, int cols, int elemSize, const void *A,
     *                       int lda, void *B, int ldb, cudaStream_t stream)
     *
     * cublasSetMatrixAsync has the same functionnality as cublasSetMatrix
     * but the transfer is done asynchronously within the CUDA stream passed
     * in parameter.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or
     *                                ldb <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasSetMatrixAsync(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        cudaStream_t stream)
    {
        return checkResult(cublasSetMatrixAsyncNative(rows, cols, elemSize, A, lda, B, ldb, stream));
    }
    private static native int cublasSetMatrixAsyncNative(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        cudaStream_t stream);


    /**
     * <pre>
     * cublasStatus_t
     * cublasGetMatrixAsync (int rows, int cols, int elemSize, const void *A,
     *                       int lda, void *B, int ldb, cudaStream_t stream)
     *
     * cublasGetMatrixAsync has the same functionnality as cublasGetMatrix
     * but the transfer is done asynchronously within the CUDA stream passed
     * in parameter.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasGetMatrixAsync(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        cudaStream_t stream)
    {
        return checkResult(cublasGetMatrixAsyncNative(rows, cols, elemSize, A, lda, B, ldb, stream));
    }
    private static native int cublasGetMatrixAsyncNative(
        int rows,
        int cols,
        int elemSize,
        Pointer A,
        int lda,
        Pointer B,
        int ldb,
        cudaStream_t stream);
    

    
    //=========================================================================
    // Memory management functions, 64 bits
    
    /**
     * <pre>
     * cublasStatus_t
     * cublasSetVector (int n, int elemSize, const void *x, int incx,
     *                  void *y, int incy)
     *
     * copies n elements from a vector x in CPU memory space to a vector y
     * in GPU memory space. Elements in both vectors are assumed to have a
     * size of elemSize bytes. Storage spacing between consecutive elements
     * is incx for the source vector x and incy for the destination vector
     * y. In general, y points to an object, or part of an object, allocated
     * via cublasAlloc(). Column major format for two-dimensional matrices
     * is assumed throughout CUBLAS. Therefore, if the increment for a vector
     * is equal to 1, this access a column vector while using an increment
     * equal to the leading dimension of the respective matrix accesses a
     * row vector.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasSetVector_64(
        long n,
        long elemSize,
        Pointer x,
        long incx,
        Pointer devicePtr,
        long incy)
    {
        return checkResult(cublasSetVector_64Native(n, elemSize, x, incx, devicePtr, incy));
    }
    private static native int cublasSetVector_64Native(
        long n,
        long elemSize,
        Pointer x,
        long incx,
        Pointer devicePtr,
        long incy);


    /**
     * <pre>
     * cublasStatus_t
     * cublasGetVector (int n, int elemSize, const void *x, int incx,
     *                  void *y, int incy)
     *
     * copies n elements from a vector x in GPU memory space to a vector y
     * in CPU memory space. Elements in both vectors are assumed to have a
     * size of elemSize bytes. Storage spacing between consecutive elements
     * is incx for the source vector x and incy for the destination vector
     * y. In general, x points to an object, or part of an object, allocated
     * via cublasAlloc(). Column major format for two-dimensional matrices
     * is assumed throughout CUBLAS. Therefore, if the increment for a vector
     * is equal to 1, this access a column vector while using an increment
     * equal to the leading dimension of the respective matrix accesses a
     * row vector.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasGetVector_64(
        long n,
        long elemSize,
        Pointer x,
        long incx,
        Pointer y,
        long incy)
    {
        return checkResult(cublasGetVector_64Native(n, elemSize, x, incx, y, incy));
    }
    private static native int cublasGetVector_64Native(
        long n,
        long elemSize,
        Pointer x,
        long incx,
        Pointer y,
        long incy);


    /**
     * <pre>
     * cublasStatus_t
     * cublasSetMatrix (int rows, int cols, int elemSize, const void *A,
     *                  int lda, void *B, int ldb)
     *
     * copies a tile of rows x cols elements from a matrix A in CPU memory
     * space to a matrix B in GPU memory space. Each element requires storage
     * of elemSize bytes. Both matrices are assumed to be stored in column
     * major format, with the leading dimension (i.e. number of rows) of
     * source matrix A provided in lda, and the leading dimension of matrix B
     * provided in ldb. In general, B points to an object, or part of an
     * object, that was allocated via cublasAlloc().
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or
     *                                ldb <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasSetMatrix_64(
        long rows,
        long cols,
        long elemSize,
        Pointer A,
        long lda,
        Pointer B,
        long ldb)
    {
        return checkResult(cublasSetMatrix_64Native(rows, cols, elemSize, A, lda, B, ldb));
    }
    private static native int cublasSetMatrix_64Native(
        long rows,
        long cols,
        long elemSize,
        Pointer A,
        long lda,
        Pointer B,
        long ldb);


    /**
     * <pre>
     * cublasStatus_t
     * cublasGetMatrix (int rows, int cols, int elemSize, const void *A,
     *                  int lda, void *B, int ldb)
     *
     * copies a tile of rows x cols elements from a matrix A in GPU memory
     * space to a matrix B in CPU memory space. Each element requires storage
     * of elemSize bytes. Both matrices are assumed to be stored in column
     * major format, with the leading dimension (i.e. number of rows) of
     * source matrix A provided in lda, and the leading dimension of matrix B
     * provided in ldb. In general, A points to an object, or part of an
     * object, that was allocated via cublasAlloc().
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasGetMatrix_64(
        long rows,
        long cols,
        long elemSize,
        Pointer A,
        long lda,
        Pointer B,
        long ldb)
    {
        return checkResult(cublasGetMatrix_64Native(rows, cols, elemSize, A, lda, B, ldb));
    }
    private static native int cublasGetMatrix_64Native(
        long rows,
        long cols,
        long elemSize,
        Pointer A,
        long lda,
        Pointer B,
        long ldb);


    /**
     * <pre>
     * cublasStatus
     * cublasSetVectorAsync ( int n, int elemSize, const void *x, int incx,
     *                       void *y, int incy, cudaStream_t stream );
     *
     * cublasSetVectorAsync has the same functionnality as cublasSetVector
     * but the transfer is done asynchronously within the CUDA stream passed
     * in parameter.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasSetVectorAsync_64(
        long n,
        long elemSize,
        Pointer hostPtr,
        long incx,
        Pointer devicePtr,
        long incy,
        cudaStream_t stream)
    {
        return checkResult(cublasSetVectorAsync_64Native(n, elemSize, hostPtr, incx, devicePtr, incy, stream));
    }
    private static native int cublasSetVectorAsync_64Native(
        long n,
        long elemSize,
        Pointer hostPtr,
        long incx,
        Pointer devicePtr,
        long incy,
        cudaStream_t stream);


    /**
     * <pre>
     * cublasStatus
     * cublasGetVectorAsync( int n, int elemSize, const void *x, int incx,
     *                       void *y, int incy, cudaStream_t stream)
     *
     * cublasGetVectorAsync has the same functionnality as cublasGetVector
     * but the transfer is done asynchronously within the CUDA stream passed
     * in parameter.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasGetVectorAsync_64(
        int n,
        int elemSize,
        Pointer devicePtr,
        int incx,
        Pointer hostPtr,
        int incy,
        cudaStream_t stream)
    {
        return checkResult(cublasGetVectorAsync_64Native(n, elemSize, devicePtr, incx, hostPtr, incy, stream));
    }
    private static native int cublasGetVectorAsync_64Native(
        long n,
        long elemSize,
        Pointer devicePtr,
        long incx,
        Pointer hostPtr,
        long incy,
        cudaStream_t stream);


    /**
     * <pre>
     * cublasStatus_t
     * cublasSetMatrixAsync (int rows, int cols, int elemSize, const void *A,
     *                       int lda, void *B, int ldb, cudaStream_t stream)
     *
     * cublasSetMatrixAsync has the same functionnality as cublasSetMatrix
     * but the transfer is done asynchronously within the CUDA stream passed
     * in parameter.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or
     *                                ldb <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasSetMatrixAsync_64(
        long rows,
        long cols,
        long elemSize,
        Pointer A,
        long lda,
        Pointer B,
        long ldb,
        cudaStream_t stream)
    {
        return checkResult(cublasSetMatrixAsync_64Native(rows, cols, elemSize, A, lda, B, ldb, stream));
    }
    private static native int cublasSetMatrixAsync_64Native(
        long rows,
        long cols,
        long elemSize,
        Pointer A,
        long lda,
        Pointer B,
        long ldb,
        cudaStream_t stream);


    /**
     * <pre>
     * cublasStatus_t
     * cublasGetMatrixAsync (int rows, int cols, int elemSize, const void *A,
     *                       int lda, void *B, int ldb, cudaStream_t stream)
     *
     * cublasGetMatrixAsync has the same functionnality as cublasGetMatrix
     * but the transfer is done asynchronously within the CUDA stream passed
     * in parameter.
     *
     * Return Values
     * -------------
     * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
     * CUBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
     * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
     * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
     * </pre>
     */
    public static int cublasGetMatrixAsync_64(
        long rows,
        long cols,
        long elemSize,
        Pointer A,
        long lda,
        Pointer B,
        long ldb,
        cudaStream_t stream)
    {
        return checkResult(cublasGetMatrixAsync_64Native(rows, cols, elemSize, A, lda, B, ldb, stream));
    }
    private static native int cublasGetMatrixAsync_64Native(
        long rows,
        long cols,
        long elemSize,
        Pointer A,
        long lda,
        Pointer B,
        long ldb,
        cudaStream_t stream);
    
    
    
    
    
    
    private static int cublasMigrateComputeType(cublasHandle handle,
        int dataType, int computeType[])
    {
        int mathMode[] =
        { cublasMath.CUBLAS_DEFAULT_MATH };
        int status = cublasStatus.CUBLAS_STATUS_SUCCESS;

        status = cublasGetMathMode(handle, mathMode);
        if (status != cublasStatus.CUBLAS_STATUS_SUCCESS)
        {
            return status;
        }

        boolean isPedantic =
            ((mathMode[0] & 0xf) == cublasMath.CUBLAS_PEDANTIC_MATH);

        switch (dataType)
        {
            case cudaDataType.CUDA_R_32F:
            case cudaDataType.CUDA_C_32F:
                computeType[0] =
                    isPedantic ? cublasComputeType.CUBLAS_COMPUTE_32F_PEDANTIC
                        : cublasComputeType.CUBLAS_COMPUTE_32F;
                return cublasStatus.CUBLAS_STATUS_SUCCESS;
            case cudaDataType.CUDA_R_64F:
            case cudaDataType.CUDA_C_64F:
                computeType[0] =
                    isPedantic ? cublasComputeType.CUBLAS_COMPUTE_64F_PEDANTIC
                        : cublasComputeType.CUBLAS_COMPUTE_64F;
                return cublasStatus.CUBLAS_STATUS_SUCCESS;
            case cudaDataType.CUDA_R_16F:
                computeType[0] =
                    isPedantic ? cublasComputeType.CUBLAS_COMPUTE_16F_PEDANTIC
                        : cublasComputeType.CUBLAS_COMPUTE_16F;
                return cublasStatus.CUBLAS_STATUS_SUCCESS;
            case cudaDataType.CUDA_R_32I:
                computeType[0] =
                    isPedantic ? cublasComputeType.CUBLAS_COMPUTE_32I_PEDANTIC
                        : cublasComputeType.CUBLAS_COMPUTE_32I;
                return cublasStatus.CUBLAS_STATUS_SUCCESS;
            default:
                return cublasStatus.CUBLAS_STATUS_NOT_SUPPORTED;
        }
    }
    
    /** wrappers to accept old code with cudaDataType computeType when referenced from c++ code */
    public static int cublasGemmEx(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, /** host or device pointer */
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer B, 
        int Btype, 
        int ldb, 
        Pointer beta, /** host or device pointer */
        Pointer C, 
        int Ctype, 
        int ldc, 
        int computeType, 
        int algo)
    {
        int cublasComputeType[] = { 0 };
        int status = cublasMigrateComputeType(handle, computeType, cublasComputeType);
        if (status != cublasStatus.CUBLAS_STATUS_SUCCESS) 
        {
            return status;
    }
        return cublasGemmEx_new(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, cublasComputeType[0], algo);
    }

    public static int cublasGemmBatchedEx(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, /** host or device pointer */
        Pointer Aarray, 
        int Atype, 
        int lda, 
        Pointer Barray, 
        int Btype, 
        int ldb, 
        Pointer beta, /** host or device pointer */
        Pointer Carray, 
        int Ctype, 
        int ldc, 
        int batchCount, 
        int computeType, 
        int algo)
    {
        int cublasComputeType[] = { 0 };
        int status = cublasMigrateComputeType(handle, computeType, cublasComputeType);
        if (status != cublasStatus.CUBLAS_STATUS_SUCCESS) 
        {
            return status;
    }
        return cublasGemmBatchedEx_new(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, cublasComputeType[0], algo);
    }

    public static int cublasGemmStridedBatchedEx(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, /** host or device pointer */
        Pointer A, 
        int Atype, 
        int lda, 
        long strideA, /** purposely signed */
        Pointer B, 
        int Btype, 
        int ldb, 
        long strideB, 
        Pointer beta, /** host or device pointer */
        Pointer C, 
        int Ctype, 
        int ldc, 
        long strideC, 
        int batchCount, 
        int computeType, 
        int algo)
    {
        int cublasComputeType[] = { 0 };
        int status = cublasMigrateComputeType(handle, computeType, cublasComputeType);
        if (status != cublasStatus.CUBLAS_STATUS_SUCCESS) 
        {
            return status;
    }
        return cublasGemmStridedBatchedEx_new(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, cublasComputeType[0], algo);
    }

    
    
    
    //=== Auto-generated part ================================================

    public static int cublasCreate(
        cublasHandle handle)
    {
        return checkResult(cublasCreateNative(handle));
    }
    private static native int cublasCreateNative(
        cublasHandle handle);


    public static int cublasDestroy(
        cublasHandle handle)
    {
        return checkResult(cublasDestroyNative(handle));
    }
    private static native int cublasDestroyNative(
        cublasHandle handle);


    public static int cublasGetVersion(
        cublasHandle handle, 
        int[] version)
    {
        return checkResult(cublasGetVersionNative(handle, version));
    }
    private static native int cublasGetVersionNative(
        cublasHandle handle, 
        int[] version);


    public static int cublasGetProperty(
        int type, 
        int[] value)
    {
        return checkResult(cublasGetPropertyNative(type, value));
    }
    private static native int cublasGetPropertyNative(
        int type, 
        int[] value);


    public static long cublasGetCudartVersion()
    {
        return cublasGetCudartVersionNative();
    }
    private static native long cublasGetCudartVersionNative();


    public static int cublasSetWorkspace(
        cublasHandle handle, 
        Pointer workspace, 
        long workspaceSizeInBytes)
    {
        return checkResult(cublasSetWorkspaceNative(handle, workspace, workspaceSizeInBytes));
    }
    private static native int cublasSetWorkspaceNative(
        cublasHandle handle, 
        Pointer workspace, 
        long workspaceSizeInBytes);


    public static int cublasSetStream(
        cublasHandle handle, 
        cudaStream_t streamId)
    {
        return checkResult(cublasSetStreamNative(handle, streamId));
    }
    private static native int cublasSetStreamNative(
        cublasHandle handle, 
        cudaStream_t streamId);


    public static int cublasGetStream(
        cublasHandle handle, 
        cudaStream_t streamId)
    {
        return checkResult(cublasGetStreamNative(handle, streamId));
    }
    private static native int cublasGetStreamNative(
        cublasHandle handle, 
        cudaStream_t streamId);


    public static int cublasGetPointerMode(
        cublasHandle handle, 
        int[] mode)
    {
        return checkResult(cublasGetPointerModeNative(handle, mode));
    }
    private static native int cublasGetPointerModeNative(
        cublasHandle handle, 
        int[] mode);


    public static int cublasSetPointerMode(
        cublasHandle handle, 
        int mode)
    {
        return checkResult(cublasSetPointerModeNative(handle, mode));
    }
    private static native int cublasSetPointerModeNative(
        cublasHandle handle, 
        int mode);


    public static int cublasGetAtomicsMode(
        cublasHandle handle, 
        int[] mode)
    {
        return checkResult(cublasGetAtomicsModeNative(handle, mode));
    }
    private static native int cublasGetAtomicsModeNative(
        cublasHandle handle, 
        int[] mode);


    public static int cublasSetAtomicsMode(
        cublasHandle handle, 
        int mode)
    {
        return checkResult(cublasSetAtomicsModeNative(handle, mode));
    }
    private static native int cublasSetAtomicsModeNative(
        cublasHandle handle, 
        int mode);


    public static int cublasGetMathMode(
        cublasHandle handle, 
        int[] mode)
    {
        return checkResult(cublasGetMathModeNative(handle, mode));
    }
    private static native int cublasGetMathModeNative(
        cublasHandle handle, 
        int[] mode);


    public static int cublasSetMathMode(
        cublasHandle handle, 
        int mode)
    {
        return checkResult(cublasSetMathModeNative(handle, mode));
    }
    private static native int cublasSetMathModeNative(
        cublasHandle handle, 
        int mode);


    public static int cublasGetSmCountTarget(
        cublasHandle handle, 
        Pointer smCountTarget)
    {
        return checkResult(cublasGetSmCountTargetNative(handle, smCountTarget));
    }
    private static native int cublasGetSmCountTargetNative(
        cublasHandle handle, 
        Pointer smCountTarget);


    public static int cublasSetSmCountTarget(
        cublasHandle handle, 
        int smCountTarget)
    {
        return checkResult(cublasSetSmCountTargetNative(handle, smCountTarget));
    }
    private static native int cublasSetSmCountTargetNative(
        cublasHandle handle, 
        int smCountTarget);


    public static String cublasGetStatusName(
        int status)
    {
        return cublasGetStatusNameNative(status);
    }
    private static native String cublasGetStatusNameNative(
        int status);


    public static String cublasGetStatusString(
        int status)
    {
        return cublasGetStatusStringNative(status);
    }
    private static native String cublasGetStatusStringNative(
        int status);


    public static int cublasLoggerConfigure(
        int logIsOn, 
        int logToStdOut, 
        int logToStdErr, 
        String logFileName)
    {
        return checkResult(cublasLoggerConfigureNative(logIsOn, logToStdOut, logToStdErr, logFileName));
    }
    private static native int cublasLoggerConfigureNative(
        int logIsOn, 
        int logToStdOut, 
        int logToStdErr, 
        String logFileName);


    public static int cublasSetLoggerCallback(
        cublasLogCallback userCallback)
    {
        return checkResult(cublasSetLoggerCallbackNative(userCallback));
    }
    private static native int cublasSetLoggerCallbackNative(
        cublasLogCallback userCallback);


    public static int cublasGetLoggerCallback(
        cublasLogCallback[] userCallback)
    {
        return checkResult(cublasGetLoggerCallbackNative(userCallback));
    }
    private static native int cublasGetLoggerCallbackNative(
        cublasLogCallback[] userCallback);


    /** --------------- CUBLAS BLAS1 Functions  ---------------- */
    public static int cublasNrm2Ex(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer result, 
        int resultType, 
        int executionType)
    {
        return checkResult(cublasNrm2ExNative(handle, n, x, xType, incx, result, resultType, executionType));
    }
    private static native int cublasNrm2ExNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer result, 
        int resultType, 
        int executionType);


    public static int cublasNrm2Ex_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer result, 
        int resultType, 
        int executionType)
    {
        return checkResult(cublasNrm2Ex_64Native(handle, n, x, xType, incx, result, resultType, executionType));
    }
    private static native int cublasNrm2Ex_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer result, 
        int resultType, 
        int executionType);


    public static int cublasSnrm2(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasSnrm2Native(handle, n, x, incx, result));
    }
    private static native int cublasSnrm2Native(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasSnrm2_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result)
    {
        return checkResult(cublasSnrm2_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasSnrm2_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result);


    public static int cublasDnrm2(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasDnrm2Native(handle, n, x, incx, result));
    }
    private static native int cublasDnrm2Native(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasDnrm2_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result)
    {
        return checkResult(cublasDnrm2_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasDnrm2_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result);


    public static int cublasScnrm2(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasScnrm2Native(handle, n, x, incx, result));
    }
    private static native int cublasScnrm2Native(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasScnrm2_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result)
    {
        return checkResult(cublasScnrm2_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasScnrm2_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result);


    public static int cublasDznrm2(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasDznrm2Native(handle, n, x, incx, result));
    }
    private static native int cublasDznrm2Native(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasDznrm2_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result)
    {
        return checkResult(cublasDznrm2_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasDznrm2_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result);


    public static int cublasDotEx(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer y, 
        int yType, 
        int incy, 
        Pointer result, 
        int resultType, 
        int executionType)
    {
        return checkResult(cublasDotExNative(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType));
    }
    private static native int cublasDotExNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer y, 
        int yType, 
        int incy, 
        Pointer result, 
        int resultType, 
        int executionType);


    public static int cublasDotEx_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer y, 
        int yType, 
        long incy, 
        Pointer result, 
        int resultType, 
        int executionType)
    {
        return checkResult(cublasDotEx_64Native(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType));
    }
    private static native int cublasDotEx_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer y, 
        int yType, 
        long incy, 
        Pointer result, 
        int resultType, 
        int executionType);


    public static int cublasDotcEx(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer y, 
        int yType, 
        int incy, 
        Pointer result, 
        int resultType, 
        int executionType)
    {
        return checkResult(cublasDotcExNative(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType));
    }
    private static native int cublasDotcExNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer y, 
        int yType, 
        int incy, 
        Pointer result, 
        int resultType, 
        int executionType);


    public static int cublasDotcEx_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer y, 
        int yType, 
        long incy, 
        Pointer result, 
        int resultType, 
        int executionType)
    {
        return checkResult(cublasDotcEx_64Native(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType));
    }
    private static native int cublasDotcEx_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer y, 
        int yType, 
        long incy, 
        Pointer result, 
        int resultType, 
        int executionType);


    public static int cublasSdot(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer result)
    {
        return checkResult(cublasSdotNative(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasSdotNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer result);


    public static int cublasSdot_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer result)
    {
        return checkResult(cublasSdot_v2_64Native(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasSdot_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer result);


    public static int cublasDdot(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer result)
    {
        return checkResult(cublasDdotNative(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasDdotNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer result);


    public static int cublasDdot_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer result)
    {
        return checkResult(cublasDdot_v2_64Native(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasDdot_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer result);


    public static int cublasCdotu(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer result)
    {
        return checkResult(cublasCdotuNative(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasCdotuNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer result);


    public static int cublasCdotu_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer result)
    {
        return checkResult(cublasCdotu_v2_64Native(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasCdotu_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer result);


    public static int cublasCdotc(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer result)
    {
        return checkResult(cublasCdotcNative(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasCdotcNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer result);


    public static int cublasCdotc_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer result)
    {
        return checkResult(cublasCdotc_v2_64Native(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasCdotc_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer result);


    public static int cublasZdotu(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer result)
    {
        return checkResult(cublasZdotuNative(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasZdotuNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer result);


    public static int cublasZdotu_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer result)
    {
        return checkResult(cublasZdotu_v2_64Native(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasZdotu_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer result);


    public static int cublasZdotc(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer result)
    {
        return checkResult(cublasZdotcNative(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasZdotcNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer result);


    public static int cublasZdotc_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer result)
    {
        return checkResult(cublasZdotc_v2_64Native(handle, n, x, incx, y, incy, result));
    }
    private static native int cublasZdotc_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer result);


    public static int cublasScalEx(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        int alphaType, 
        Pointer x, 
        int xType, 
        int incx, 
        int executionType)
    {
        return checkResult(cublasScalExNative(handle, n, alpha, alphaType, x, xType, incx, executionType));
    }
    private static native int cublasScalExNative(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        int alphaType, 
        Pointer x, 
        int xType, 
        int incx, 
        int executionType);


    public static int cublasScalEx_64(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        int alphaType, 
        Pointer x, 
        int xType, 
        long incx, 
        int executionType)
    {
        return checkResult(cublasScalEx_64Native(handle, n, alpha, alphaType, x, xType, incx, executionType));
    }
    private static native int cublasScalEx_64Native(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        int alphaType, 
        Pointer x, 
        int xType, 
        long incx, 
        int executionType);


    public static int cublasSscal(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasSscalNative(handle, n, alpha, x, incx));
    }
    private static native int cublasSscalNative(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx);


    public static int cublasSscal_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasSscal_v2_64Native(handle, n, alpha, x, incx));
    }
    private static native int cublasSscal_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx);


    public static int cublasDscal(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasDscalNative(handle, n, alpha, x, incx));
    }
    private static native int cublasDscalNative(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx);


    public static int cublasDscal_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasDscal_v2_64Native(handle, n, alpha, x, incx));
    }
    private static native int cublasDscal_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx);


    public static int cublasCscal(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasCscalNative(handle, n, alpha, x, incx));
    }
    private static native int cublasCscalNative(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx);


    public static int cublasCscal_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasCscal_v2_64Native(handle, n, alpha, x, incx));
    }
    private static native int cublasCscal_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx);


    public static int cublasCsscal(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasCsscalNative(handle, n, alpha, x, incx));
    }
    private static native int cublasCsscalNative(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx);


    public static int cublasCsscal_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasCsscal_v2_64Native(handle, n, alpha, x, incx));
    }
    private static native int cublasCsscal_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx);


    public static int cublasZscal(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasZscalNative(handle, n, alpha, x, incx));
    }
    private static native int cublasZscalNative(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx);


    public static int cublasZscal_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasZscal_v2_64Native(handle, n, alpha, x, incx));
    }
    private static native int cublasZscal_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx);


    public static int cublasZdscal(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasZdscalNative(handle, n, alpha, x, incx));
    }
    private static native int cublasZdscalNative(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx);


    public static int cublasZdscal_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasZdscal_v2_64Native(handle, n, alpha, x, incx));
    }
    private static native int cublasZdscal_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx);


    public static int cublasAxpyEx(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        int alphaType, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer y, 
        int yType, 
        int incy, 
        int executiontype)
    {
        return checkResult(cublasAxpyExNative(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype));
    }
    private static native int cublasAxpyExNative(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        int alphaType, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer y, 
        int yType, 
        int incy, 
        int executiontype);


    public static int cublasAxpyEx_64(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        int alphaType, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer y, 
        int yType, 
        long incy, 
        int executiontype)
    {
        return checkResult(cublasAxpyEx_64Native(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype));
    }
    private static native int cublasAxpyEx_64Native(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        int alphaType, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer y, 
        int yType, 
        long incy, 
        int executiontype);


    public static int cublasSaxpy(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasSaxpyNative(handle, n, alpha, x, incx, y, incy));
    }
    private static native int cublasSaxpyNative(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy);


    public static int cublasSaxpy_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasSaxpy_v2_64Native(handle, n, alpha, x, incx, y, incy));
    }
    private static native int cublasSaxpy_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy);


    public static int cublasDaxpy(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasDaxpyNative(handle, n, alpha, x, incx, y, incy));
    }
    private static native int cublasDaxpyNative(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy);


    public static int cublasDaxpy_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasDaxpy_v2_64Native(handle, n, alpha, x, incx, y, incy));
    }
    private static native int cublasDaxpy_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy);


    public static int cublasCaxpy(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasCaxpyNative(handle, n, alpha, x, incx, y, incy));
    }
    private static native int cublasCaxpyNative(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy);


    public static int cublasCaxpy_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasCaxpy_v2_64Native(handle, n, alpha, x, incx, y, incy));
    }
    private static native int cublasCaxpy_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy);


    public static int cublasZaxpy(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasZaxpyNative(handle, n, alpha, x, incx, y, incy));
    }
    private static native int cublasZaxpyNative(
        cublasHandle handle, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy);


    public static int cublasZaxpy_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasZaxpy_v2_64Native(handle, n, alpha, x, incx, y, incy));
    }
    private static native int cublasZaxpy_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy);


    public static int cublasCopyEx(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer y, 
        int yType, 
        int incy)
    {
        return checkResult(cublasCopyExNative(handle, n, x, xType, incx, y, yType, incy));
    }
    private static native int cublasCopyExNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer y, 
        int yType, 
        int incy);


    public static int cublasCopyEx_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer y, 
        int yType, 
        long incy)
    {
        return checkResult(cublasCopyEx_64Native(handle, n, x, xType, incx, y, yType, incy));
    }
    private static native int cublasCopyEx_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer y, 
        int yType, 
        long incy);


    public static int cublasScopy(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasScopyNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasScopyNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy);


    public static int cublasScopy_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasScopy_v2_64Native(handle, n, x, incx, y, incy));
    }
    private static native int cublasScopy_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy);


    public static int cublasDcopy(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasDcopyNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasDcopyNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy);


    public static int cublasDcopy_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasDcopy_v2_64Native(handle, n, x, incx, y, incy));
    }
    private static native int cublasDcopy_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy);


    public static int cublasCcopy(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasCcopyNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasCcopyNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy);


    public static int cublasCcopy_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasCcopy_v2_64Native(handle, n, x, incx, y, incy));
    }
    private static native int cublasCcopy_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy);


    public static int cublasZcopy(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasZcopyNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasZcopyNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy);


    public static int cublasZcopy_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasZcopy_v2_64Native(handle, n, x, incx, y, incy));
    }
    private static native int cublasZcopy_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy);


    public static int cublasSswap(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasSswapNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasSswapNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy);


    public static int cublasSswap_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasSswap_v2_64Native(handle, n, x, incx, y, incy));
    }
    private static native int cublasSswap_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy);


    public static int cublasDswap(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasDswapNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasDswapNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy);


    public static int cublasDswap_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasDswap_v2_64Native(handle, n, x, incx, y, incy));
    }
    private static native int cublasDswap_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy);


    public static int cublasCswap(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasCswapNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasCswapNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy);


    public static int cublasCswap_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasCswap_v2_64Native(handle, n, x, incx, y, incy));
    }
    private static native int cublasCswap_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy);


    public static int cublasZswap(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasZswapNative(handle, n, x, incx, y, incy));
    }
    private static native int cublasZswapNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy);


    public static int cublasZswap_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasZswap_v2_64Native(handle, n, x, incx, y, incy));
    }
    private static native int cublasZswap_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy);


    public static int cublasSwapEx(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer y, 
        int yType, 
        int incy)
    {
        return checkResult(cublasSwapExNative(handle, n, x, xType, incx, y, yType, incy));
    }
    private static native int cublasSwapExNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer y, 
        int yType, 
        int incy);


    public static int cublasSwapEx_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer y, 
        int yType, 
        long incy)
    {
        return checkResult(cublasSwapEx_64Native(handle, n, x, xType, incx, y, yType, incy));
    }
    private static native int cublasSwapEx_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer y, 
        int yType, 
        long incy);


    public static int cublasIsamax(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasIsamaxNative(handle, n, x, incx, result));
    }
    private static native int cublasIsamaxNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasIsamax_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result)
    {
        return checkResult(cublasIsamax_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasIsamax_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result);


    public static int cublasIdamax(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasIdamaxNative(handle, n, x, incx, result));
    }
    private static native int cublasIdamaxNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasIdamax_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result)
    {
        return checkResult(cublasIdamax_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasIdamax_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result);


    public static int cublasIcamax(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasIcamaxNative(handle, n, x, incx, result));
    }
    private static native int cublasIcamaxNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasIcamax_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result)
    {
        return checkResult(cublasIcamax_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasIcamax_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result);


    public static int cublasIzamax(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasIzamaxNative(handle, n, x, incx, result));
    }
    private static native int cublasIzamaxNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasIzamax_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result)
    {
        return checkResult(cublasIzamax_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasIzamax_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result);


    public static int cublasIamaxEx(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasIamaxExNative(handle, n, x, xType, incx, result));
    }
    private static native int cublasIamaxExNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer result);


    public static int cublasIamaxEx_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        long[] result)
    {
        return checkResult(cublasIamaxEx_64Native(handle, n, x, xType, incx, result));
    }
    private static native int cublasIamaxEx_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        long[] result);


    public static int cublasIsamin(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasIsaminNative(handle, n, x, incx, result));
    }
    private static native int cublasIsaminNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasIsamin_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result)
    {
        return checkResult(cublasIsamin_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasIsamin_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result);


    public static int cublasIdamin(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasIdaminNative(handle, n, x, incx, result));
    }
    private static native int cublasIdaminNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasIdamin_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result)
    {
        return checkResult(cublasIdamin_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasIdamin_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result);


    public static int cublasIcamin(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasIcaminNative(handle, n, x, incx, result));
    }
    private static native int cublasIcaminNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasIcamin_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result)
    {
        return checkResult(cublasIcamin_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasIcamin_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result);


    public static int cublasIzamin(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasIzaminNative(handle, n, x, incx, result));
    }
    private static native int cublasIzaminNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasIzamin_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result)
    {
        return checkResult(cublasIzamin_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasIzamin_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        long[] result);


    public static int cublasIaminEx(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasIaminExNative(handle, n, x, xType, incx, result));
    }
    private static native int cublasIaminExNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer result);


    public static int cublasIaminEx_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        long[] result)
    {
        return checkResult(cublasIaminEx_64Native(handle, n, x, xType, incx, result));
    }
    private static native int cublasIaminEx_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        long[] result);


    public static int cublasAsumEx(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer result, 
        int resultType, 
        int executiontype)
    {
        return checkResult(cublasAsumExNative(handle, n, x, xType, incx, result, resultType, executiontype));
    }
    private static native int cublasAsumExNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer result, 
        int resultType, 
        int executiontype);


    public static int cublasAsumEx_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer result, 
        int resultType, 
        int executiontype)
    {
        return checkResult(cublasAsumEx_64Native(handle, n, x, xType, incx, result, resultType, executiontype));
    }
    private static native int cublasAsumEx_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer result, 
        int resultType, 
        int executiontype);


    public static int cublasSasum(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasSasumNative(handle, n, x, incx, result));
    }
    private static native int cublasSasumNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasSasum_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result)
    {
        return checkResult(cublasSasum_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasSasum_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result);


    public static int cublasDasum(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasDasumNative(handle, n, x, incx, result));
    }
    private static native int cublasDasumNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasDasum_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result)
    {
        return checkResult(cublasDasum_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasDasum_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result);


    public static int cublasScasum(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasScasumNative(handle, n, x, incx, result));
    }
    private static native int cublasScasumNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasScasum_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result)
    {
        return checkResult(cublasScasum_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasScasum_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result);


    public static int cublasDzasum(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result)
    {
        return checkResult(cublasDzasumNative(handle, n, x, incx, result));
    }
    private static native int cublasDzasumNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer result);


    public static int cublasDzasum_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result)
    {
        return checkResult(cublasDzasum_v2_64Native(handle, n, x, incx, result));
    }
    private static native int cublasDzasum_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer result);


    public static int cublasSrot(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasSrotNative(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasSrotNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer c, 
        Pointer s);


    public static int cublasSrot_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasSrot_v2_64Native(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasSrot_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer c, 
        Pointer s);


    public static int cublasDrot(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasDrotNative(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasDrotNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer c, 
        Pointer s);


    public static int cublasDrot_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasDrot_v2_64Native(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasDrot_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer c, 
        Pointer s);


    public static int cublasCrot(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasCrotNative(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasCrotNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer c, 
        Pointer s);


    public static int cublasCrot_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasCrot_v2_64Native(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasCrot_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer c, 
        Pointer s);


    public static int cublasCsrot(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasCsrotNative(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasCsrotNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer c, 
        Pointer s);


    public static int cublasCsrot_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasCsrot_v2_64Native(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasCsrot_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer c, 
        Pointer s);


    public static int cublasZrot(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasZrotNative(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasZrotNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer c, 
        Pointer s);


    public static int cublasZrot_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasZrot_v2_64Native(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasZrot_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer c, 
        Pointer s);


    public static int cublasZdrot(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasZdrotNative(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasZdrotNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer c, 
        Pointer s);


    public static int cublasZdrot_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasZdrot_v2_64Native(handle, n, x, incx, y, incy, c, s));
    }
    private static native int cublasZdrot_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer c, 
        Pointer s);


    public static int cublasRotEx(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer y, 
        int yType, 
        int incy, 
        Pointer c, 
        Pointer s, 
        int csType, 
        int executiontype)
    {
        return checkResult(cublasRotExNative(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype));
    }
    private static native int cublasRotExNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer y, 
        int yType, 
        int incy, 
        Pointer c, 
        Pointer s, 
        int csType, 
        int executiontype);


    public static int cublasRotEx_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer y, 
        int yType, 
        long incy, 
        Pointer c, 
        Pointer s, 
        int csType, 
        int executiontype)
    {
        return checkResult(cublasRotEx_64Native(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype));
    }
    private static native int cublasRotEx_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer y, 
        int yType, 
        long incy, 
        Pointer c, 
        Pointer s, 
        int csType, 
        int executiontype);


    public static int cublasSrotg(
        cublasHandle handle, 
        Pointer a, 
        Pointer b, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasSrotgNative(handle, a, b, c, s));
    }
    private static native int cublasSrotgNative(
        cublasHandle handle, 
        Pointer a, 
        Pointer b, 
        Pointer c, 
        Pointer s);


    public static int cublasDrotg(
        cublasHandle handle, 
        Pointer a, 
        Pointer b, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasDrotgNative(handle, a, b, c, s));
    }
    private static native int cublasDrotgNative(
        cublasHandle handle, 
        Pointer a, 
        Pointer b, 
        Pointer c, 
        Pointer s);


    public static int cublasCrotg(
        cublasHandle handle, 
        Pointer a, 
        Pointer b, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasCrotgNative(handle, a, b, c, s));
    }
    private static native int cublasCrotgNative(
        cublasHandle handle, 
        Pointer a, 
        Pointer b, 
        Pointer c, 
        Pointer s);


    public static int cublasZrotg(
        cublasHandle handle, 
        Pointer a, 
        Pointer b, 
        Pointer c, 
        Pointer s)
    {
        return checkResult(cublasZrotgNative(handle, a, b, c, s));
    }
    private static native int cublasZrotgNative(
        cublasHandle handle, 
        Pointer a, 
        Pointer b, 
        Pointer c, 
        Pointer s);


    public static int cublasRotgEx(
        cublasHandle handle, 
        Pointer a, 
        Pointer b, 
        int abType, 
        Pointer c, 
        Pointer s, 
        int csType, 
        int executiontype)
    {
        return checkResult(cublasRotgExNative(handle, a, b, abType, c, s, csType, executiontype));
    }
    private static native int cublasRotgExNative(
        cublasHandle handle, 
        Pointer a, 
        Pointer b, 
        int abType, 
        Pointer c, 
        Pointer s, 
        int csType, 
        int executiontype);


    public static int cublasSrotm(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer param)
    {
        return checkResult(cublasSrotmNative(handle, n, x, incx, y, incy, param));
    }
    private static native int cublasSrotmNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer param);


    public static int cublasSrotm_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer param)
    {
        return checkResult(cublasSrotm_v2_64Native(handle, n, x, incx, y, incy, param));
    }
    private static native int cublasSrotm_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer param);


    public static int cublasDrotm(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer param)
    {
        return checkResult(cublasDrotmNative(handle, n, x, incx, y, incy, param));
    }
    private static native int cublasDrotmNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer param);


    public static int cublasDrotm_v2_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer param)
    {
        return checkResult(cublasDrotm_v2_64Native(handle, n, x, incx, y, incy, param));
    }
    private static native int cublasDrotm_v2_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer param);


    public static int cublasRotmEx(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer y, 
        int yType, 
        int incy, 
        Pointer param, 
        int paramType, 
        int executiontype)
    {
        return checkResult(cublasRotmExNative(handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype));
    }
    private static native int cublasRotmExNative(
        cublasHandle handle, 
        int n, 
        Pointer x, 
        int xType, 
        int incx, 
        Pointer y, 
        int yType, 
        int incy, 
        Pointer param, 
        int paramType, 
        int executiontype);


    public static int cublasRotmEx_64(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer y, 
        int yType, 
        long incy, 
        Pointer param, 
        int paramType, 
        int executiontype)
    {
        return checkResult(cublasRotmEx_64Native(handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype));
    }
    private static native int cublasRotmEx_64Native(
        cublasHandle handle, 
        long n, 
        Pointer x, 
        int xType, 
        long incx, 
        Pointer y, 
        int yType, 
        long incy, 
        Pointer param, 
        int paramType, 
        int executiontype);


    public static int cublasSrotmg(
        cublasHandle handle, 
        Pointer d1, 
        Pointer d2, 
        Pointer x1, 
        Pointer y1, 
        Pointer param)
    {
        return checkResult(cublasSrotmgNative(handle, d1, d2, x1, y1, param));
    }
    private static native int cublasSrotmgNative(
        cublasHandle handle, 
        Pointer d1, 
        Pointer d2, 
        Pointer x1, 
        Pointer y1, 
        Pointer param);


    public static int cublasDrotmg(
        cublasHandle handle, 
        Pointer d1, 
        Pointer d2, 
        Pointer x1, 
        Pointer y1, 
        Pointer param)
    {
        return checkResult(cublasDrotmgNative(handle, d1, d2, x1, y1, param));
    }
    private static native int cublasDrotmgNative(
        cublasHandle handle, 
        Pointer d1, 
        Pointer d2, 
        Pointer x1, 
        Pointer y1, 
        Pointer param);


    public static int cublasRotmgEx(
        cublasHandle handle, 
        Pointer d1, 
        int d1Type, 
        Pointer d2, 
        int d2Type, 
        Pointer x1, 
        int x1Type, 
        Pointer y1, 
        int y1Type, 
        Pointer param, 
        int paramType, 
        int executiontype)
    {
        return checkResult(cublasRotmgExNative(handle, d1, d1Type, d2, d2Type, x1, x1Type, y1, y1Type, param, paramType, executiontype));
    }
    private static native int cublasRotmgExNative(
        cublasHandle handle, 
        Pointer d1, 
        int d1Type, 
        Pointer d2, 
        int d2Type, 
        Pointer x1, 
        int x1Type, 
        Pointer y1, 
        int y1Type, 
        Pointer param, 
        int paramType, 
        int executiontype);


    /** --------------- CUBLAS BLAS2 Functions  ---------------- */
    /** GEMV */
    public static int cublasSgemv(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasSgemvNative(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasSgemvNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasSgemv_v2_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasSgemv_v2_64Native(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasSgemv_v2_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasDgemv(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasDgemvNative(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasDgemvNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasDgemv_v2_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasDgemv_v2_64Native(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasDgemv_v2_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasCgemv(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasCgemvNative(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasCgemvNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasCgemv_v2_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasCgemv_v2_64Native(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasCgemv_v2_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasZgemv(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasZgemvNative(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasZgemvNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasZgemv_v2_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasZgemv_v2_64Native(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasZgemv_v2_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    /** GBMV */
    public static int cublasSgbmv(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int kl, 
        int ku, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasSgbmvNative(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasSgbmvNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int kl, 
        int ku, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasSgbmv_v2_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        long kl, 
        long ku, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasSgbmv_v2_64Native(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasSgbmv_v2_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        long kl, 
        long ku, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasDgbmv(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int kl, 
        int ku, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasDgbmvNative(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasDgbmvNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int kl, 
        int ku, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasDgbmv_v2_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        long kl, 
        long ku, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasDgbmv_v2_64Native(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasDgbmv_v2_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        long kl, 
        long ku, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasCgbmv(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int kl, 
        int ku, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasCgbmvNative(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasCgbmvNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int kl, 
        int ku, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasCgbmv_v2_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        long kl, 
        long ku, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasCgbmv_v2_64Native(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasCgbmv_v2_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        long kl, 
        long ku, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasZgbmv(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int kl, 
        int ku, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasZgbmvNative(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasZgbmvNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int kl, 
        int ku, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasZgbmv_v2_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        long kl, 
        long ku, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasZgbmv_v2_64Native(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasZgbmv_v2_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        long kl, 
        long ku, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    /** TRMV */
    public static int cublasStrmv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasStrmvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasStrmvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasStrmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasStrmv_v2_64Native(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasStrmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    public static int cublasDtrmv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasDtrmvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasDtrmvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasDtrmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasDtrmv_v2_64Native(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasDtrmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    public static int cublasCtrmv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasCtrmvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasCtrmvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasCtrmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasCtrmv_v2_64Native(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasCtrmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    public static int cublasZtrmv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasZtrmvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasZtrmvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasZtrmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasZtrmv_v2_64Native(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasZtrmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    /** TBMV */
    public static int cublasStbmv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasStbmvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasStbmvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasStbmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasStbmv_v2_64Native(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasStbmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    public static int cublasDtbmv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasDtbmvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasDtbmvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasDtbmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasDtbmv_v2_64Native(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasDtbmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    public static int cublasCtbmv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasCtbmvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasCtbmvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasCtbmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasCtbmv_v2_64Native(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasCtbmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    public static int cublasZtbmv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasZtbmvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasZtbmvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasZtbmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasZtbmv_v2_64Native(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasZtbmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    /** TPMV */
    public static int cublasStpmv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasStpmvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasStpmvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx);


    public static int cublasStpmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasStpmv_v2_64Native(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasStpmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx);


    public static int cublasDtpmv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasDtpmvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasDtpmvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx);


    public static int cublasDtpmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasDtpmv_v2_64Native(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasDtpmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx);


    public static int cublasCtpmv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasCtpmvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasCtpmvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx);


    public static int cublasCtpmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasCtpmv_v2_64Native(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasCtpmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx);


    public static int cublasZtpmv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasZtpmvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasZtpmvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx);


    public static int cublasZtpmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasZtpmv_v2_64Native(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasZtpmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx);


    /** TRSV */
    public static int cublasStrsv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasStrsvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasStrsvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasStrsv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasStrsv_v2_64Native(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasStrsv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    public static int cublasDtrsv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasDtrsvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasDtrsvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasDtrsv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasDtrsv_v2_64Native(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasDtrsv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    public static int cublasCtrsv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasCtrsvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasCtrsvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasCtrsv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasCtrsv_v2_64Native(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasCtrsv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    public static int cublasZtrsv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasZtrsvNative(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasZtrsvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasZtrsv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasZtrsv_v2_64Native(handle, uplo, trans, diag, n, A, lda, x, incx));
    }
    private static native int cublasZtrsv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    /** TPSV */
    public static int cublasStpsv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasStpsvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasStpsvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx);


    public static int cublasStpsv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasStpsv_v2_64Native(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasStpsv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx);


    public static int cublasDtpsv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasDtpsvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasDtpsvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx);


    public static int cublasDtpsv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasDtpsv_v2_64Native(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasDtpsv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx);


    public static int cublasCtpsv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasCtpsvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasCtpsvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx);


    public static int cublasCtpsv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasCtpsv_v2_64Native(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasCtpsv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx);


    public static int cublasZtpsv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasZtpsvNative(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasZtpsvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        Pointer AP, 
        Pointer x, 
        int incx);


    public static int cublasZtpsv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasZtpsv_v2_64Native(handle, uplo, trans, diag, n, AP, x, incx));
    }
    private static native int cublasZtpsv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        Pointer AP, 
        Pointer x, 
        long incx);


    /** TBSV */
    public static int cublasStbsv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasStbsvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasStbsvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasStbsv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasStbsv_v2_64Native(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasStbsv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    public static int cublasDtbsv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasDtbsvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasDtbsvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasDtbsv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasDtbsv_v2_64Native(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasDtbsv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    public static int cublasCtbsv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasCtbsvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasCtbsvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasCtbsv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasCtbsv_v2_64Native(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasCtbsv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    public static int cublasZtbsv(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx)
    {
        return checkResult(cublasZtbsvNative(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasZtbsvNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        int n, 
        int k, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx);


    public static int cublasZtbsv_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx)
    {
        return checkResult(cublasZtbsv_v2_64Native(handle, uplo, trans, diag, n, k, A, lda, x, incx));
    }
    private static native int cublasZtbsv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int diag, 
        long n, 
        long k, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx);


    /** SYMV/HEMV */
    public static int cublasSsymv(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasSsymvNative(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasSsymvNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasSsymv_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasSsymv_v2_64Native(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasSsymv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasDsymv(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasDsymvNative(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasDsymvNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasDsymv_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasDsymv_v2_64Native(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasDsymv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasCsymv(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasCsymvNative(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasCsymvNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasCsymv_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasCsymv_v2_64Native(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasCsymv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasZsymv(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasZsymvNative(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasZsymvNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasZsymv_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasZsymv_v2_64Native(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasZsymv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasChemv(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasChemvNative(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasChemvNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasChemv_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasChemv_v2_64Native(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasChemv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasZhemv(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasZhemvNative(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasZhemvNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasZhemv_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasZhemv_v2_64Native(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasZhemv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    /** SBMV/HBMV */
    public static int cublasSsbmv(
        cublasHandle handle, 
        int uplo, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasSsbmvNative(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasSsbmvNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasSsbmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasSsbmv_v2_64Native(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasSsbmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasDsbmv(
        cublasHandle handle, 
        int uplo, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasDsbmvNative(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasDsbmvNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasDsbmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasDsbmv_v2_64Native(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasDsbmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasChbmv(
        cublasHandle handle, 
        int uplo, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasChbmvNative(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasChbmvNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasChbmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasChbmv_v2_64Native(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasChbmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasZhbmv(
        cublasHandle handle, 
        int uplo, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasZhbmvNative(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasZhbmvNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasZhbmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasZhbmv_v2_64Native(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
    }
    private static native int cublasZhbmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    /** SPMV/HPMV */
    public static int cublasSspmv(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasSspmvNative(handle, uplo, n, alpha, AP, x, incx, beta, y, incy));
    }
    private static native int cublasSspmvNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasSspmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasSspmv_v2_64Native(handle, uplo, n, alpha, AP, x, incx, beta, y, incy));
    }
    private static native int cublasSspmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasDspmv(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasDspmvNative(handle, uplo, n, alpha, AP, x, incx, beta, y, incy));
    }
    private static native int cublasDspmvNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasDspmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasDspmv_v2_64Native(handle, uplo, n, alpha, AP, x, incx, beta, y, incy));
    }
    private static native int cublasDspmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasChpmv(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasChpmvNative(handle, uplo, n, alpha, AP, x, incx, beta, y, incy));
    }
    private static native int cublasChpmvNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasChpmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasChpmv_v2_64Native(handle, uplo, n, alpha, AP, x, incx, beta, y, incy));
    }
    private static native int cublasChpmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    public static int cublasZhpmv(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy)
    {
        return checkResult(cublasZhpmvNative(handle, uplo, n, alpha, AP, x, incx, beta, y, incy));
    }
    private static native int cublasZhpmvNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        int incx, 
        Pointer beta, 
        Pointer y, 
        int incy);


    public static int cublasZhpmv_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy)
    {
        return checkResult(cublasZhpmv_v2_64Native(handle, uplo, n, alpha, AP, x, incx, beta, y, incy));
    }
    private static native int cublasZhpmv_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer AP, 
        Pointer x, 
        long incx, 
        Pointer beta, 
        Pointer y, 
        long incy);


    /** GER */
    public static int cublasSger(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasSgerNative(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasSgerNative(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda);


    public static int cublasSger_v2_64(
        cublasHandle handle, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasSger_v2_64Native(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasSger_v2_64Native(
        cublasHandle handle, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda);


    public static int cublasDger(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasDgerNative(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasDgerNative(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda);


    public static int cublasDger_v2_64(
        cublasHandle handle, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasDger_v2_64Native(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasDger_v2_64Native(
        cublasHandle handle, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda);


    public static int cublasCgeru(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasCgeruNative(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasCgeruNative(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda);


    public static int cublasCgeru_v2_64(
        cublasHandle handle, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasCgeru_v2_64Native(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasCgeru_v2_64Native(
        cublasHandle handle, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda);


    public static int cublasCgerc(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasCgercNative(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasCgercNative(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda);


    public static int cublasCgerc_v2_64(
        cublasHandle handle, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasCgerc_v2_64Native(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasCgerc_v2_64Native(
        cublasHandle handle, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda);


    public static int cublasZgeru(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasZgeruNative(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasZgeruNative(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda);


    public static int cublasZgeru_v2_64(
        cublasHandle handle, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasZgeru_v2_64Native(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasZgeru_v2_64Native(
        cublasHandle handle, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda);


    public static int cublasZgerc(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasZgercNative(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasZgercNative(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda);


    public static int cublasZgerc_v2_64(
        cublasHandle handle, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasZgerc_v2_64Native(handle, m, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasZgerc_v2_64Native(
        cublasHandle handle, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda);


    /** SYR/HER */
    public static int cublasSsyr(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasSsyrNative(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasSsyrNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer A, 
        int lda);


    public static int cublasSsyr_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasSsyr_v2_64Native(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasSsyr_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer A, 
        long lda);


    public static int cublasDsyr(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasDsyrNative(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasDsyrNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer A, 
        int lda);


    public static int cublasDsyr_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasDsyr_v2_64Native(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasDsyr_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer A, 
        long lda);


    public static int cublasCsyr(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasCsyrNative(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasCsyrNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer A, 
        int lda);


    public static int cublasCsyr_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasCsyr_v2_64Native(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasCsyr_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer A, 
        long lda);


    public static int cublasZsyr(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasZsyrNative(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasZsyrNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer A, 
        int lda);


    public static int cublasZsyr_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasZsyr_v2_64Native(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasZsyr_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer A, 
        long lda);


    public static int cublasCher(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasCherNative(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasCherNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer A, 
        int lda);


    public static int cublasCher_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasCher_v2_64Native(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasCher_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer A, 
        long lda);


    public static int cublasZher(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasZherNative(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasZherNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer A, 
        int lda);


    public static int cublasZher_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasZher_v2_64Native(handle, uplo, n, alpha, x, incx, A, lda));
    }
    private static native int cublasZher_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer A, 
        long lda);


    /** SPR/HPR */
    public static int cublasSspr(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer AP)
    {
        return checkResult(cublasSsprNative(handle, uplo, n, alpha, x, incx, AP));
    }
    private static native int cublasSsprNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer AP);


    public static int cublasSspr_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer AP)
    {
        return checkResult(cublasSspr_v2_64Native(handle, uplo, n, alpha, x, incx, AP));
    }
    private static native int cublasSspr_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer AP);


    public static int cublasDspr(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer AP)
    {
        return checkResult(cublasDsprNative(handle, uplo, n, alpha, x, incx, AP));
    }
    private static native int cublasDsprNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer AP);


    public static int cublasDspr_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer AP)
    {
        return checkResult(cublasDspr_v2_64Native(handle, uplo, n, alpha, x, incx, AP));
    }
    private static native int cublasDspr_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer AP);


    public static int cublasChpr(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer AP)
    {
        return checkResult(cublasChprNative(handle, uplo, n, alpha, x, incx, AP));
    }
    private static native int cublasChprNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer AP);


    public static int cublasChpr_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer AP)
    {
        return checkResult(cublasChpr_v2_64Native(handle, uplo, n, alpha, x, incx, AP));
    }
    private static native int cublasChpr_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer AP);


    public static int cublasZhpr(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer AP)
    {
        return checkResult(cublasZhprNative(handle, uplo, n, alpha, x, incx, AP));
    }
    private static native int cublasZhprNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer AP);


    public static int cublasZhpr_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer AP)
    {
        return checkResult(cublasZhpr_v2_64Native(handle, uplo, n, alpha, x, incx, AP));
    }
    private static native int cublasZhpr_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer AP);


    /** SYR2/HER2 */
    public static int cublasSsyr2(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasSsyr2Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasSsyr2Native(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda);


    public static int cublasSsyr2_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasSsyr2_v2_64Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasSsyr2_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda);


    public static int cublasDsyr2(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasDsyr2Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasDsyr2Native(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda);


    public static int cublasDsyr2_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasDsyr2_v2_64Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasDsyr2_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda);


    public static int cublasCsyr2(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasCsyr2Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasCsyr2Native(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda);


    public static int cublasCsyr2_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasCsyr2_v2_64Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasCsyr2_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda);


    public static int cublasZsyr2(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasZsyr2Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasZsyr2Native(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda);


    public static int cublasZsyr2_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasZsyr2_v2_64Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasZsyr2_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda);


    public static int cublasCher2(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasCher2Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasCher2Native(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda);


    public static int cublasCher2_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasCher2_v2_64Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasCher2_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda);


    public static int cublasZher2(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasZher2Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasZher2Native(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer A, 
        int lda);


    public static int cublasZher2_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda)
    {
        return checkResult(cublasZher2_v2_64Native(handle, uplo, n, alpha, x, incx, y, incy, A, lda));
    }
    private static native int cublasZher2_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer A, 
        long lda);


    /** SPR2/HPR2 */
    public static int cublasSspr2(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer AP)
    {
        return checkResult(cublasSspr2Native(handle, uplo, n, alpha, x, incx, y, incy, AP));
    }
    private static native int cublasSspr2Native(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer AP);


    public static int cublasSspr2_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer AP)
    {
        return checkResult(cublasSspr2_v2_64Native(handle, uplo, n, alpha, x, incx, y, incy, AP));
    }
    private static native int cublasSspr2_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer AP);


    public static int cublasDspr2(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer AP)
    {
        return checkResult(cublasDspr2Native(handle, uplo, n, alpha, x, incx, y, incy, AP));
    }
    private static native int cublasDspr2Native(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer AP);


    public static int cublasDspr2_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer AP)
    {
        return checkResult(cublasDspr2_v2_64Native(handle, uplo, n, alpha, x, incx, y, incy, AP));
    }
    private static native int cublasDspr2_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer AP);


    public static int cublasChpr2(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer AP)
    {
        return checkResult(cublasChpr2Native(handle, uplo, n, alpha, x, incx, y, incy, AP));
    }
    private static native int cublasChpr2Native(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer AP);


    public static int cublasChpr2_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer AP)
    {
        return checkResult(cublasChpr2_v2_64Native(handle, uplo, n, alpha, x, incx, y, incy, AP));
    }
    private static native int cublasChpr2_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer AP);


    public static int cublasZhpr2(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer AP)
    {
        return checkResult(cublasZhpr2Native(handle, uplo, n, alpha, x, incx, y, incy, AP));
    }
    private static native int cublasZhpr2Native(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer alpha, 
        Pointer x, 
        int incx, 
        Pointer y, 
        int incy, 
        Pointer AP);


    public static int cublasZhpr2_v2_64(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer AP)
    {
        return checkResult(cublasZhpr2_v2_64Native(handle, uplo, n, alpha, x, incx, y, incy, AP));
    }
    private static native int cublasZhpr2_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        long n, 
        Pointer alpha, 
        Pointer x, 
        long incx, 
        Pointer y, 
        long incy, 
        Pointer AP);


    /** BATCH GEMV */
    public static int cublasSgemvBatched(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer xarray, 
        int incx, 
        Pointer beta, 
        Pointer yarray, 
        int incy, 
        int batchCount)
    {
        return checkResult(cublasSgemvBatchedNative(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
    }
    private static native int cublasSgemvBatchedNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer xarray, 
        int incx, 
        Pointer beta, 
        Pointer yarray, 
        int incy, 
        int batchCount);


    public static int cublasSgemvBatched_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer xarray, 
        long incx, 
        Pointer beta, 
        Pointer yarray, 
        long incy, 
        long batchCount)
    {
        return checkResult(cublasSgemvBatched_64Native(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
    }
    private static native int cublasSgemvBatched_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer xarray, 
        long incx, 
        Pointer beta, 
        Pointer yarray, 
        long incy, 
        long batchCount);


    public static int cublasDgemvBatched(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer xarray, 
        int incx, 
        Pointer beta, 
        Pointer yarray, 
        int incy, 
        int batchCount)
    {
        return checkResult(cublasDgemvBatchedNative(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
    }
    private static native int cublasDgemvBatchedNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer xarray, 
        int incx, 
        Pointer beta, 
        Pointer yarray, 
        int incy, 
        int batchCount);


    public static int cublasDgemvBatched_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer xarray, 
        long incx, 
        Pointer beta, 
        Pointer yarray, 
        long incy, 
        long batchCount)
    {
        return checkResult(cublasDgemvBatched_64Native(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
    }
    private static native int cublasDgemvBatched_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer xarray, 
        long incx, 
        Pointer beta, 
        Pointer yarray, 
        long incy, 
        long batchCount);


    public static int cublasCgemvBatched(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer xarray, 
        int incx, 
        Pointer beta, 
        Pointer yarray, 
        int incy, 
        int batchCount)
    {
        return checkResult(cublasCgemvBatchedNative(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
    }
    private static native int cublasCgemvBatchedNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer xarray, 
        int incx, 
        Pointer beta, 
        Pointer yarray, 
        int incy, 
        int batchCount);


    public static int cublasCgemvBatched_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer xarray, 
        long incx, 
        Pointer beta, 
        Pointer yarray, 
        long incy, 
        long batchCount)
    {
        return checkResult(cublasCgemvBatched_64Native(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
    }
    private static native int cublasCgemvBatched_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer xarray, 
        long incx, 
        Pointer beta, 
        Pointer yarray, 
        long incy, 
        long batchCount);


    public static int cublasZgemvBatched(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer xarray, 
        int incx, 
        Pointer beta, 
        Pointer yarray, 
        int incy, 
        int batchCount)
    {
        return checkResult(cublasZgemvBatchedNative(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
    }
    private static native int cublasZgemvBatchedNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer xarray, 
        int incx, 
        Pointer beta, 
        Pointer yarray, 
        int incy, 
        int batchCount);


    public static int cublasZgemvBatched_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer xarray, 
        long incx, 
        Pointer beta, 
        Pointer yarray, 
        long incy, 
        long batchCount)
    {
        return checkResult(cublasZgemvBatched_64Native(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
    }
    private static native int cublasZgemvBatched_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer xarray, 
        long incx, 
        Pointer beta, 
        Pointer yarray, 
        long incy, 
        long batchCount);


    public static int cublasSgemvStridedBatched(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer x, 
        int incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        int incy, 
        long stridey, 
        int batchCount)
    {
        return checkResult(cublasSgemvStridedBatchedNative(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
    }
    private static native int cublasSgemvStridedBatchedNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer x, 
        int incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        int incy, 
        long stridey, 
        int batchCount);


    public static int cublasSgemvStridedBatched_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer x, 
        long incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        long incy, 
        long stridey, 
        long batchCount)
    {
        return checkResult(cublasSgemvStridedBatched_64Native(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
    }
    private static native int cublasSgemvStridedBatched_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer x, 
        long incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        long incy, 
        long stridey, 
        long batchCount);


    public static int cublasDgemvStridedBatched(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer x, 
        int incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        int incy, 
        long stridey, 
        int batchCount)
    {
        return checkResult(cublasDgemvStridedBatchedNative(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
    }
    private static native int cublasDgemvStridedBatchedNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer x, 
        int incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        int incy, 
        long stridey, 
        int batchCount);


    public static int cublasDgemvStridedBatched_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer x, 
        long incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        long incy, 
        long stridey, 
        long batchCount)
    {
        return checkResult(cublasDgemvStridedBatched_64Native(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
    }
    private static native int cublasDgemvStridedBatched_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer x, 
        long incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        long incy, 
        long stridey, 
        long batchCount);


    public static int cublasCgemvStridedBatched(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer x, 
        int incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        int incy, 
        long stridey, 
        int batchCount)
    {
        return checkResult(cublasCgemvStridedBatchedNative(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
    }
    private static native int cublasCgemvStridedBatchedNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer x, 
        int incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        int incy, 
        long stridey, 
        int batchCount);


    public static int cublasCgemvStridedBatched_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer x, 
        long incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        long incy, 
        long stridey, 
        long batchCount)
    {
        return checkResult(cublasCgemvStridedBatched_64Native(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
    }
    private static native int cublasCgemvStridedBatched_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer x, 
        long incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        long incy, 
        long stridey, 
        long batchCount);


    public static int cublasZgemvStridedBatched(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer x, 
        int incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        int incy, 
        long stridey, 
        int batchCount)
    {
        return checkResult(cublasZgemvStridedBatchedNative(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
    }
    private static native int cublasZgemvStridedBatchedNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer x, 
        int incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        int incy, 
        long stridey, 
        int batchCount);


    public static int cublasZgemvStridedBatched_64(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer x, 
        long incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        long incy, 
        long stridey, 
        long batchCount)
    {
        return checkResult(cublasZgemvStridedBatched_64Native(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
    }
    private static native int cublasZgemvStridedBatched_64Native(
        cublasHandle handle, 
        int trans, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer x, 
        long incx, 
        long stridex, 
        Pointer beta, 
        Pointer y, 
        long incy, 
        long stridey, 
        long batchCount);


    /** ---------------- CUBLAS BLAS3 Functions ---------------- */
    /** GEMM */
    public static int cublasSgemm(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasSgemmNative(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasSgemmNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasSgemm_v2_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasSgemm_v2_64Native(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasSgemm_v2_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasDgemm(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasDgemmNative(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasDgemmNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasDgemm_v2_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasDgemm_v2_64Native(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasDgemm_v2_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasCgemm(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasCgemmNative(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCgemmNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasCgemm_v2_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasCgemm_v2_64Native(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCgemm_v2_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasCgemm3m(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasCgemm3mNative(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCgemm3mNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasCgemm3m_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasCgemm3m_64Native(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCgemm3m_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasCgemm3mEx(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer B, 
        int Btype, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc)
    {
        return checkResult(cublasCgemm3mExNative(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc));
    }
    private static native int cublasCgemm3mExNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer B, 
        int Btype, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc);


    public static int cublasCgemm3mEx_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer B, 
        int Btype, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc)
    {
        return checkResult(cublasCgemm3mEx_64Native(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc));
    }
    private static native int cublasCgemm3mEx_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer B, 
        int Btype, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc);


    public static int cublasZgemm(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasZgemmNative(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZgemmNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasZgemm_v2_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasZgemm_v2_64Native(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZgemm_v2_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasZgemm3m(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasZgemm3mNative(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZgemm3mNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasZgemm3m_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasZgemm3m_64Native(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZgemm3m_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasSgemmEx(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer B, 
        int Btype, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc)
    {
        return checkResult(cublasSgemmExNative(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc));
    }
    private static native int cublasSgemmExNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer B, 
        int Btype, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc);


    public static int cublasSgemmEx_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer B, 
        int Btype, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc)
    {
        return checkResult(cublasSgemmEx_64Native(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc));
    }
    private static native int cublasSgemmEx_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer B, 
        int Btype, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc);


    public static int cublasGemmEx_new(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer B, 
        int Btype, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc, 
        int computeType, 
        int algo)
    {
        return checkResult(cublasGemmEx_newNative(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo));
    }
    private static native int cublasGemmEx_newNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer B, 
        int Btype, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc, 
        int computeType, 
        int algo);


    public static int cublasGemmEx_64_new(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer B, 
        int Btype, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc, 
        int computeType, 
        int algo)
    {
        return checkResult(cublasGemmEx_64_newNative(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo));
    }
    private static native int cublasGemmEx_64_newNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer B, 
        int Btype, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc, 
        int computeType, 
        int algo);


    public static int cublasCgemmEx(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer B, 
        int Btype, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc)
    {
        return checkResult(cublasCgemmExNative(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc));
    }
    private static native int cublasCgemmExNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer B, 
        int Btype, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc);


    public static int cublasCgemmEx_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer B, 
        int Btype, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc)
    {
        return checkResult(cublasCgemmEx_64Native(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc));
    }
    private static native int cublasCgemmEx_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer B, 
        int Btype, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc);


    /** SYRK */
    public static int cublasSsyrk(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasSsyrkNative(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasSsyrkNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasSsyrk_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasSsyrk_v2_64Native(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasSsyrk_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasDsyrk(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasDsyrkNative(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasDsyrkNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasDsyrk_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasDsyrk_v2_64Native(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasDsyrk_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasCsyrk(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasCsyrkNative(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasCsyrkNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasCsyrk_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasCsyrk_v2_64Native(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasCsyrk_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasZsyrk(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasZsyrkNative(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasZsyrkNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasZsyrk_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasZsyrk_v2_64Native(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasZsyrk_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasCsyrkEx(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc)
    {
        return checkResult(cublasCsyrkExNative(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc));
    }
    private static native int cublasCsyrkExNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc);


    public static int cublasCsyrkEx_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc)
    {
        return checkResult(cublasCsyrkEx_64Native(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc));
    }
    private static native int cublasCsyrkEx_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc);


    public static int cublasCsyrk3mEx(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc)
    {
        return checkResult(cublasCsyrk3mExNative(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc));
    }
    private static native int cublasCsyrk3mExNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc);


    public static int cublasCsyrk3mEx_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc)
    {
        return checkResult(cublasCsyrk3mEx_64Native(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc));
    }
    private static native int cublasCsyrk3mEx_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc);


    /** HERK */
    public static int cublasCherk(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasCherkNative(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasCherkNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasCherk_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasCherk_v2_64Native(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasCherk_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasZherk(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasZherkNative(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasZherkNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasZherk_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasZherk_v2_64Native(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    }
    private static native int cublasZherk_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasCherkEx(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc)
    {
        return checkResult(cublasCherkExNative(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc));
    }
    private static native int cublasCherkExNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc);


    public static int cublasCherkEx_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc)
    {
        return checkResult(cublasCherkEx_64Native(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc));
    }
    private static native int cublasCherkEx_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc);


    public static int cublasCherk3mEx(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc)
    {
        return checkResult(cublasCherk3mExNative(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc));
    }
    private static native int cublasCherk3mExNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc);


    public static int cublasCherk3mEx_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc)
    {
        return checkResult(cublasCherk3mEx_64Native(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc));
    }
    private static native int cublasCherk3mEx_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc);


    /** SYR2K / HER2K */
    public static int cublasSsyr2k(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasSsyr2kNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasSsyr2kNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasSsyr2k_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasSsyr2k_v2_64Native(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasSsyr2k_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasDsyr2k(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasDsyr2kNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasDsyr2kNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasDsyr2k_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasDsyr2k_v2_64Native(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasDsyr2k_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasCsyr2k(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasCsyr2kNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCsyr2kNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasCsyr2k_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasCsyr2k_v2_64Native(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCsyr2k_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasZsyr2k(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasZsyr2kNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZsyr2kNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasZsyr2k_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasZsyr2k_v2_64Native(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZsyr2k_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasCher2k(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasCher2kNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCher2kNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasCher2k_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasCher2k_v2_64Native(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCher2k_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasZher2k(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasZher2kNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZher2kNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasZher2k_v2_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasZher2k_v2_64Native(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZher2k_v2_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    /** SYRKX / HERKX */
    public static int cublasSsyrkx(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasSsyrkxNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasSsyrkxNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasSsyrkx_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasSsyrkx_64Native(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasSsyrkx_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasDsyrkx(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasDsyrkxNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasDsyrkxNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasDsyrkx_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasDsyrkx_64Native(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasDsyrkx_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasCsyrkx(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasCsyrkxNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCsyrkxNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasCsyrkx_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasCsyrkx_64Native(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCsyrkx_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasZsyrkx(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasZsyrkxNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZsyrkxNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasZsyrkx_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasZsyrkx_64Native(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZsyrkx_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasCherkx(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasCherkxNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCherkxNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasCherkx_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasCherkx_64Native(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCherkx_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasZherkx(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasZherkxNative(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZherkxNative(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasZherkx_64(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasZherkx_64Native(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZherkx_64Native(
        cublasHandle handle, 
        int uplo, 
        int trans, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    /** SYMM */
    public static int cublasSsymm(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasSsymmNative(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasSsymmNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasSsymm_v2_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasSsymm_v2_64Native(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasSsymm_v2_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasDsymm(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasDsymmNative(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasDsymmNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasDsymm_v2_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasDsymm_v2_64Native(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasDsymm_v2_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasCsymm(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasCsymmNative(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCsymmNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasCsymm_v2_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasCsymm_v2_64Native(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasCsymm_v2_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasZsymm(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasZsymmNative(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZsymmNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasZsymm_v2_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasZsymm_v2_64Native(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZsymm_v2_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    /** HEMM */
    public static int cublasChemm(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasChemmNative(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasChemmNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasChemm_v2_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasChemm_v2_64Native(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasChemm_v2_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    public static int cublasZhemm(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasZhemmNative(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZhemmNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cublasZhemm_v2_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasZhemm_v2_64Native(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    }
    private static native int cublasZhemm_v2_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer beta, 
        Pointer C, 
        long ldc);


    /** TRSM */
    public static int cublasStrsm(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb)
    {
        return checkResult(cublasStrsmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    }
    private static native int cublasStrsmNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb);


    public static int cublasStrsm_v2_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb)
    {
        return checkResult(cublasStrsm_v2_64Native(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    }
    private static native int cublasStrsm_v2_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb);


    public static int cublasDtrsm(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb)
    {
        return checkResult(cublasDtrsmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    }
    private static native int cublasDtrsmNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb);


    public static int cublasDtrsm_v2_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb)
    {
        return checkResult(cublasDtrsm_v2_64Native(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    }
    private static native int cublasDtrsm_v2_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb);


    public static int cublasCtrsm(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb)
    {
        return checkResult(cublasCtrsmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    }
    private static native int cublasCtrsmNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb);


    public static int cublasCtrsm_v2_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb)
    {
        return checkResult(cublasCtrsm_v2_64Native(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    }
    private static native int cublasCtrsm_v2_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb);


    public static int cublasZtrsm(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb)
    {
        return checkResult(cublasZtrsmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    }
    private static native int cublasZtrsmNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb);


    public static int cublasZtrsm_v2_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb)
    {
        return checkResult(cublasZtrsm_v2_64Native(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    }
    private static native int cublasZtrsm_v2_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb);


    /** TRMM */
    public static int cublasStrmm(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasStrmmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc));
    }
    private static native int cublasStrmmNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc);


    public static int cublasStrmm_v2_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasStrmm_v2_64Native(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc));
    }
    private static native int cublasStrmm_v2_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc);


    public static int cublasDtrmm(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasDtrmmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc));
    }
    private static native int cublasDtrmmNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc);


    public static int cublasDtrmm_v2_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasDtrmm_v2_64Native(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc));
    }
    private static native int cublasDtrmm_v2_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc);


    public static int cublasCtrmm(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasCtrmmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc));
    }
    private static native int cublasCtrmmNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc);


    public static int cublasCtrmm_v2_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasCtrmm_v2_64Native(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc));
    }
    private static native int cublasCtrmm_v2_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc);


    public static int cublasZtrmm(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasZtrmmNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc));
    }
    private static native int cublasZtrmmNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc);


    public static int cublasZtrmm_v2_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasZtrmm_v2_64Native(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc));
    }
    private static native int cublasZtrmm_v2_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc);


    public static int cublasSgemmBatched(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer Barray, 
        int ldb, 
        Pointer beta, 
        Pointer Carray, 
        int ldc, 
        int batchCount)
    {
        return checkResult(cublasSgemmBatchedNative(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount));
    }
    private static native int cublasSgemmBatchedNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer Barray, 
        int ldb, 
        Pointer beta, 
        Pointer Carray, 
        int ldc, 
        int batchCount);


    public static int cublasSgemmBatched_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer Barray, 
        long ldb, 
        Pointer beta, 
        Pointer Carray, 
        long ldc, 
        long batchCount)
    {
        return checkResult(cublasSgemmBatched_64Native(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount));
    }
    private static native int cublasSgemmBatched_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer Barray, 
        long ldb, 
        Pointer beta, 
        Pointer Carray, 
        long ldc, 
        long batchCount);


    public static int cublasDgemmBatched(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer Barray, 
        int ldb, 
        Pointer beta, 
        Pointer Carray, 
        int ldc, 
        int batchCount)
    {
        return checkResult(cublasDgemmBatchedNative(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount));
    }
    private static native int cublasDgemmBatchedNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer Barray, 
        int ldb, 
        Pointer beta, 
        Pointer Carray, 
        int ldc, 
        int batchCount);


    public static int cublasDgemmBatched_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer Barray, 
        long ldb, 
        Pointer beta, 
        Pointer Carray, 
        long ldc, 
        long batchCount)
    {
        return checkResult(cublasDgemmBatched_64Native(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount));
    }
    private static native int cublasDgemmBatched_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer Barray, 
        long ldb, 
        Pointer beta, 
        Pointer Carray, 
        long ldc, 
        long batchCount);


    public static int cublasCgemmBatched(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer Barray, 
        int ldb, 
        Pointer beta, 
        Pointer Carray, 
        int ldc, 
        int batchCount)
    {
        return checkResult(cublasCgemmBatchedNative(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount));
    }
    private static native int cublasCgemmBatchedNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer Barray, 
        int ldb, 
        Pointer beta, 
        Pointer Carray, 
        int ldc, 
        int batchCount);


    public static int cublasCgemmBatched_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer Barray, 
        long ldb, 
        Pointer beta, 
        Pointer Carray, 
        long ldc, 
        long batchCount)
    {
        return checkResult(cublasCgemmBatched_64Native(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount));
    }
    private static native int cublasCgemmBatched_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer Barray, 
        long ldb, 
        Pointer beta, 
        Pointer Carray, 
        long ldc, 
        long batchCount);


    public static int cublasCgemm3mBatched(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer Barray, 
        int ldb, 
        Pointer beta, 
        Pointer Carray, 
        int ldc, 
        int batchCount)
    {
        return checkResult(cublasCgemm3mBatchedNative(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount));
    }
    private static native int cublasCgemm3mBatchedNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer Barray, 
        int ldb, 
        Pointer beta, 
        Pointer Carray, 
        int ldc, 
        int batchCount);


    public static int cublasCgemm3mBatched_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer Barray, 
        long ldb, 
        Pointer beta, 
        Pointer Carray, 
        long ldc, 
        long batchCount)
    {
        return checkResult(cublasCgemm3mBatched_64Native(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount));
    }
    private static native int cublasCgemm3mBatched_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer Barray, 
        long ldb, 
        Pointer beta, 
        Pointer Carray, 
        long ldc, 
        long batchCount);


    public static int cublasZgemmBatched(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer Barray, 
        int ldb, 
        Pointer beta, 
        Pointer Carray, 
        int ldc, 
        int batchCount)
    {
        return checkResult(cublasZgemmBatchedNative(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount));
    }
    private static native int cublasZgemmBatchedNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer Aarray, 
        int lda, 
        Pointer Barray, 
        int ldb, 
        Pointer beta, 
        Pointer Carray, 
        int ldc, 
        int batchCount);


    public static int cublasZgemmBatched_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer Barray, 
        long ldb, 
        Pointer beta, 
        Pointer Carray, 
        long ldc, 
        long batchCount)
    {
        return checkResult(cublasZgemmBatched_64Native(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount));
    }
    private static native int cublasZgemmBatched_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer Aarray, 
        long lda, 
        Pointer Barray, 
        long ldb, 
        Pointer beta, 
        Pointer Carray, 
        long ldc, 
        long batchCount);


    public static int cublasSgemmStridedBatched(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer B, 
        int ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        int ldc, 
        long strideC, 
        int batchCount)
    {
        return checkResult(cublasSgemmStridedBatchedNative(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount));
    }
    private static native int cublasSgemmStridedBatchedNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer B, 
        int ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        int ldc, 
        long strideC, 
        int batchCount);


    public static int cublasSgemmStridedBatched_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer B, 
        long ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        long ldc, 
        long strideC, 
        long batchCount)
    {
        return checkResult(cublasSgemmStridedBatched_64Native(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount));
    }
    private static native int cublasSgemmStridedBatched_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer B, 
        long ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        long ldc, 
        long strideC, 
        long batchCount);


    public static int cublasDgemmStridedBatched(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer B, 
        int ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        int ldc, 
        long strideC, 
        int batchCount)
    {
        return checkResult(cublasDgemmStridedBatchedNative(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount));
    }
    private static native int cublasDgemmStridedBatchedNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer B, 
        int ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        int ldc, 
        long strideC, 
        int batchCount);


    public static int cublasDgemmStridedBatched_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer B, 
        long ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        long ldc, 
        long strideC, 
        long batchCount)
    {
        return checkResult(cublasDgemmStridedBatched_64Native(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount));
    }
    private static native int cublasDgemmStridedBatched_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer B, 
        long ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        long ldc, 
        long strideC, 
        long batchCount);


    public static int cublasCgemmStridedBatched(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer B, 
        int ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        int ldc, 
        long strideC, 
        int batchCount)
    {
        return checkResult(cublasCgemmStridedBatchedNative(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount));
    }
    private static native int cublasCgemmStridedBatchedNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer B, 
        int ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        int ldc, 
        long strideC, 
        int batchCount);


    public static int cublasCgemmStridedBatched_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer B, 
        long ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        long ldc, 
        long strideC, 
        long batchCount)
    {
        return checkResult(cublasCgemmStridedBatched_64Native(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount));
    }
    private static native int cublasCgemmStridedBatched_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer B, 
        long ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        long ldc, 
        long strideC, 
        long batchCount);


    public static int cublasCgemm3mStridedBatched(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer B, 
        int ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        int ldc, 
        long strideC, 
        int batchCount)
    {
        return checkResult(cublasCgemm3mStridedBatchedNative(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount));
    }
    private static native int cublasCgemm3mStridedBatchedNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer B, 
        int ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        int ldc, 
        long strideC, 
        int batchCount);


    public static int cublasCgemm3mStridedBatched_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer B, 
        long ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        long ldc, 
        long strideC, 
        long batchCount)
    {
        return checkResult(cublasCgemm3mStridedBatched_64Native(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount));
    }
    private static native int cublasCgemm3mStridedBatched_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer B, 
        long ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        long ldc, 
        long strideC, 
        long batchCount);


    public static int cublasZgemmStridedBatched(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer B, 
        int ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        int ldc, 
        long strideC, 
        int batchCount)
    {
        return checkResult(cublasZgemmStridedBatchedNative(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount));
    }
    private static native int cublasZgemmStridedBatchedNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        long strideA, 
        Pointer B, 
        int ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        int ldc, 
        long strideC, 
        int batchCount);


    public static int cublasZgemmStridedBatched_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer B, 
        long ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        long ldc, 
        long strideC, 
        long batchCount)
    {
        return checkResult(cublasZgemmStridedBatched_64Native(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount));
    }
    private static native int cublasZgemmStridedBatched_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        long strideA, 
        Pointer B, 
        long ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        long ldc, 
        long strideC, 
        long batchCount);


    public static int cublasGemmBatchedEx_new(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer Aarray, 
        int Atype, 
        int lda, 
        Pointer Barray, 
        int Btype, 
        int ldb, 
        Pointer beta, 
        Pointer Carray, 
        int Ctype, 
        int ldc, 
        int batchCount, 
        int computeType, 
        int algo)
    {
        return checkResult(cublasGemmBatchedEx_newNative(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo));
    }
    private static native int cublasGemmBatchedEx_newNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer Aarray, 
        int Atype, 
        int lda, 
        Pointer Barray, 
        int Btype, 
        int ldb, 
        Pointer beta, 
        Pointer Carray, 
        int Ctype, 
        int ldc, 
        int batchCount, 
        int computeType, 
        int algo);


    public static int cublasGemmBatchedEx_64_new(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer Aarray, 
        int Atype, 
        long lda, 
        Pointer Barray, 
        int Btype, 
        long ldb, 
        Pointer beta, 
        Pointer Carray, 
        int Ctype, 
        long ldc, 
        long batchCount, 
        int computeType, 
        int algo)
    {
        return checkResult(cublasGemmBatchedEx_64_newNative(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo));
    }
    private static native int cublasGemmBatchedEx_64_newNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer Aarray, 
        int Atype, 
        long lda, 
        Pointer Barray, 
        int Btype, 
        long ldb, 
        Pointer beta, 
        Pointer Carray, 
        int Ctype, 
        long ldc, 
        long batchCount, 
        int computeType, 
        int algo);


    public static int cublasGemmStridedBatchedEx_new(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        long strideA, 
        Pointer B, 
        int Btype, 
        int ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc, 
        long strideC, 
        int batchCount, 
        int computeType, 
        int algo)
    {
        return checkResult(cublasGemmStridedBatchedEx_newNative(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo));
    }
    private static native int cublasGemmStridedBatchedEx_newNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        int lda, 
        long strideA, 
        Pointer B, 
        int Btype, 
        int ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        int ldc, 
        long strideC, 
        int batchCount, 
        int computeType, 
        int algo);


    public static int cublasGemmStridedBatchedEx_64_new(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        long strideA, 
        Pointer B, 
        int Btype, 
        long ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc, 
        long strideC, 
        long batchCount, 
        int computeType, 
        int algo)
    {
        return checkResult(cublasGemmStridedBatchedEx_64_newNative(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo));
    }
    private static native int cublasGemmStridedBatchedEx_64_newNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        long k, 
        Pointer alpha, 
        Pointer A, 
        int Atype, 
        long lda, 
        long strideA, 
        Pointer B, 
        int Btype, 
        long ldb, 
        long strideB, 
        Pointer beta, 
        Pointer C, 
        int Ctype, 
        long ldc, 
        long strideC, 
        long batchCount, 
        int computeType, 
        int algo);


    /** ---------------- CUBLAS BLAS-like Extension ---------------- */
    /** GEAM */
    public static int cublasSgeam(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasSgeamNative(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
    }
    private static native int cublasSgeamNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc);


    public static int cublasSgeam_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasSgeam_64Native(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
    }
    private static native int cublasSgeam_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc);


    public static int cublasDgeam(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasDgeamNative(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
    }
    private static native int cublasDgeamNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc);


    public static int cublasDgeam_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasDgeam_64Native(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
    }
    private static native int cublasDgeam_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc);


    public static int cublasCgeam(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasCgeamNative(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
    }
    private static native int cublasCgeamNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc);


    public static int cublasCgeam_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasCgeam_64Native(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
    }
    private static native int cublasCgeam_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc);


    public static int cublasZgeam(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasZgeamNative(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
    }
    private static native int cublasZgeamNative(
        cublasHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer beta, 
        Pointer B, 
        int ldb, 
        Pointer C, 
        int ldc);


    public static int cublasZgeam_64(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasZgeam_64Native(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
    }
    private static native int cublasZgeam_64Native(
        cublasHandle handle, 
        int transa, 
        int transb, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer beta, 
        Pointer B, 
        long ldb, 
        Pointer C, 
        long ldc);


    /** TRSM - Batched Triangular Solver */
    public static int cublasStrsmBatched(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        int batchCount)
    {
        return checkResult(cublasStrsmBatchedNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount));
    }
    private static native int cublasStrsmBatchedNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        int batchCount);


    public static int cublasStrsmBatched_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        long batchCount)
    {
        return checkResult(cublasStrsmBatched_64Native(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount));
    }
    private static native int cublasStrsmBatched_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        long batchCount);


    public static int cublasDtrsmBatched(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        int batchCount)
    {
        return checkResult(cublasDtrsmBatchedNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount));
    }
    private static native int cublasDtrsmBatchedNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        int batchCount);


    public static int cublasDtrsmBatched_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        long batchCount)
    {
        return checkResult(cublasDtrsmBatched_64Native(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount));
    }
    private static native int cublasDtrsmBatched_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        long batchCount);


    public static int cublasCtrsmBatched(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        int batchCount)
    {
        return checkResult(cublasCtrsmBatchedNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount));
    }
    private static native int cublasCtrsmBatchedNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        int batchCount);


    public static int cublasCtrsmBatched_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        long batchCount)
    {
        return checkResult(cublasCtrsmBatched_64Native(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount));
    }
    private static native int cublasCtrsmBatched_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        long batchCount);


    public static int cublasZtrsmBatched(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        int batchCount)
    {
        return checkResult(cublasZtrsmBatchedNative(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount));
    }
    private static native int cublasZtrsmBatchedNative(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer B, 
        int ldb, 
        int batchCount);


    public static int cublasZtrsmBatched_64(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        long batchCount)
    {
        return checkResult(cublasZtrsmBatched_64Native(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount));
    }
    private static native int cublasZtrsmBatched_64Native(
        cublasHandle handle, 
        int side, 
        int uplo, 
        int trans, 
        int diag, 
        long m, 
        long n, 
        Pointer alpha, 
        Pointer A, 
        long lda, 
        Pointer B, 
        long ldb, 
        long batchCount);


    /** DGMM */
    public static int cublasSdgmm(
        cublasHandle handle, 
        int mode, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasSdgmmNative(handle, mode, m, n, A, lda, x, incx, C, ldc));
    }
    private static native int cublasSdgmmNative(
        cublasHandle handle, 
        int mode, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer C, 
        int ldc);


    public static int cublasSdgmm_64(
        cublasHandle handle, 
        int mode, 
        long m, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasSdgmm_64Native(handle, mode, m, n, A, lda, x, incx, C, ldc));
    }
    private static native int cublasSdgmm_64Native(
        cublasHandle handle, 
        int mode, 
        long m, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer C, 
        long ldc);


    public static int cublasDdgmm(
        cublasHandle handle, 
        int mode, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasDdgmmNative(handle, mode, m, n, A, lda, x, incx, C, ldc));
    }
    private static native int cublasDdgmmNative(
        cublasHandle handle, 
        int mode, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer C, 
        int ldc);


    public static int cublasDdgmm_64(
        cublasHandle handle, 
        int mode, 
        long m, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasDdgmm_64Native(handle, mode, m, n, A, lda, x, incx, C, ldc));
    }
    private static native int cublasDdgmm_64Native(
        cublasHandle handle, 
        int mode, 
        long m, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer C, 
        long ldc);


    public static int cublasCdgmm(
        cublasHandle handle, 
        int mode, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasCdgmmNative(handle, mode, m, n, A, lda, x, incx, C, ldc));
    }
    private static native int cublasCdgmmNative(
        cublasHandle handle, 
        int mode, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer C, 
        int ldc);


    public static int cublasCdgmm_64(
        cublasHandle handle, 
        int mode, 
        long m, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasCdgmm_64Native(handle, mode, m, n, A, lda, x, incx, C, ldc));
    }
    private static native int cublasCdgmm_64Native(
        cublasHandle handle, 
        int mode, 
        long m, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer C, 
        long ldc);


    public static int cublasZdgmm(
        cublasHandle handle, 
        int mode, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cublasZdgmmNative(handle, mode, m, n, A, lda, x, incx, C, ldc));
    }
    private static native int cublasZdgmmNative(
        cublasHandle handle, 
        int mode, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer x, 
        int incx, 
        Pointer C, 
        int ldc);


    public static int cublasZdgmm_64(
        cublasHandle handle, 
        int mode, 
        long m, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer C, 
        long ldc)
    {
        return checkResult(cublasZdgmm_64Native(handle, mode, m, n, A, lda, x, incx, C, ldc));
    }
    private static native int cublasZdgmm_64Native(
        cublasHandle handle, 
        int mode, 
        long m, 
        long n, 
        Pointer A, 
        long lda, 
        Pointer x, 
        long incx, 
        Pointer C, 
        long ldc);


    /** Batched - MATINV*/
    public static int cublasSmatinvBatched(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Ainv, 
        int lda_inv, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasSmatinvBatchedNative(handle, n, A, lda, Ainv, lda_inv, info, batchSize));
    }
    private static native int cublasSmatinvBatchedNative(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Ainv, 
        int lda_inv, 
        Pointer info, 
        int batchSize);


    public static int cublasDmatinvBatched(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Ainv, 
        int lda_inv, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasDmatinvBatchedNative(handle, n, A, lda, Ainv, lda_inv, info, batchSize));
    }
    private static native int cublasDmatinvBatchedNative(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Ainv, 
        int lda_inv, 
        Pointer info, 
        int batchSize);


    public static int cublasCmatinvBatched(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Ainv, 
        int lda_inv, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasCmatinvBatchedNative(handle, n, A, lda, Ainv, lda_inv, info, batchSize));
    }
    private static native int cublasCmatinvBatchedNative(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Ainv, 
        int lda_inv, 
        Pointer info, 
        int batchSize);


    public static int cublasZmatinvBatched(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Ainv, 
        int lda_inv, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasZmatinvBatchedNative(handle, n, A, lda, Ainv, lda_inv, info, batchSize));
    }
    private static native int cublasZmatinvBatchedNative(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer Ainv, 
        int lda_inv, 
        Pointer info, 
        int batchSize);


    /** Batch QR Factorization */
    public static int cublasSgeqrfBatched(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer Aarray, 
        int lda, 
        Pointer TauArray, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasSgeqrfBatchedNative(handle, m, n, Aarray, lda, TauArray, info, batchSize));
    }
    private static native int cublasSgeqrfBatchedNative(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer Aarray, 
        int lda, 
        Pointer TauArray, 
        Pointer info, 
        int batchSize);


    public static int cublasDgeqrfBatched(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer Aarray, 
        int lda, 
        Pointer TauArray, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasDgeqrfBatchedNative(handle, m, n, Aarray, lda, TauArray, info, batchSize));
    }
    private static native int cublasDgeqrfBatchedNative(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer Aarray, 
        int lda, 
        Pointer TauArray, 
        Pointer info, 
        int batchSize);


    public static int cublasCgeqrfBatched(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer Aarray, 
        int lda, 
        Pointer TauArray, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasCgeqrfBatchedNative(handle, m, n, Aarray, lda, TauArray, info, batchSize));
    }
    private static native int cublasCgeqrfBatchedNative(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer Aarray, 
        int lda, 
        Pointer TauArray, 
        Pointer info, 
        int batchSize);


    public static int cublasZgeqrfBatched(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer Aarray, 
        int lda, 
        Pointer TauArray, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasZgeqrfBatchedNative(handle, m, n, Aarray, lda, TauArray, info, batchSize));
    }
    private static native int cublasZgeqrfBatchedNative(
        cublasHandle handle, 
        int m, 
        int n, 
        Pointer Aarray, 
        int lda, 
        Pointer TauArray, 
        Pointer info, 
        int batchSize);


    /** Least Square Min only m >= n and Non-transpose supported */
    public static int cublasSgelsBatched(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer Carray, 
        int ldc, 
        Pointer info, 
        Pointer devInfoArray, 
        int batchSize)
    {
        return checkResult(cublasSgelsBatchedNative(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize));
    }
    private static native int cublasSgelsBatchedNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer Carray, 
        int ldc, 
        Pointer info, 
        Pointer devInfoArray, 
        int batchSize);


    public static int cublasDgelsBatched(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer Carray, 
        int ldc, 
        Pointer info, 
        Pointer devInfoArray, 
        int batchSize)
    {
        return checkResult(cublasDgelsBatchedNative(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize));
    }
    private static native int cublasDgelsBatchedNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer Carray, 
        int ldc, 
        Pointer info, 
        Pointer devInfoArray, 
        int batchSize);


    public static int cublasCgelsBatched(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer Carray, 
        int ldc, 
        Pointer info, 
        Pointer devInfoArray, 
        int batchSize)
    {
        return checkResult(cublasCgelsBatchedNative(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize));
    }
    private static native int cublasCgelsBatchedNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer Carray, 
        int ldc, 
        Pointer info, 
        Pointer devInfoArray, 
        int batchSize);


    public static int cublasZgelsBatched(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer Carray, 
        int ldc, 
        Pointer info, 
        Pointer devInfoArray, 
        int batchSize)
    {
        return checkResult(cublasZgelsBatchedNative(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize));
    }
    private static native int cublasZgelsBatchedNative(
        cublasHandle handle, 
        int trans, 
        int m, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer Carray, 
        int ldc, 
        Pointer info, 
        Pointer devInfoArray, 
        int batchSize);


    /** TPTTR : Triangular Pack format to Triangular format */
    public static int cublasStpttr(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer AP, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasStpttrNative(handle, uplo, n, AP, A, lda));
    }
    private static native int cublasStpttrNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer AP, 
        Pointer A, 
        int lda);


    public static int cublasDtpttr(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer AP, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasDtpttrNative(handle, uplo, n, AP, A, lda));
    }
    private static native int cublasDtpttrNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer AP, 
        Pointer A, 
        int lda);


    public static int cublasCtpttr(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer AP, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasCtpttrNative(handle, uplo, n, AP, A, lda));
    }
    private static native int cublasCtpttrNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer AP, 
        Pointer A, 
        int lda);


    public static int cublasZtpttr(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer AP, 
        Pointer A, 
        int lda)
    {
        return checkResult(cublasZtpttrNative(handle, uplo, n, AP, A, lda));
    }
    private static native int cublasZtpttrNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer AP, 
        Pointer A, 
        int lda);


    /** TRTTP : Triangular format to Triangular Pack format */
    public static int cublasStrttp(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer AP)
    {
        return checkResult(cublasStrttpNative(handle, uplo, n, A, lda, AP));
    }
    private static native int cublasStrttpNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer AP);


    public static int cublasDtrttp(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer AP)
    {
        return checkResult(cublasDtrttpNative(handle, uplo, n, A, lda, AP));
    }
    private static native int cublasDtrttpNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer AP);


    public static int cublasCtrttp(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer AP)
    {
        return checkResult(cublasCtrttpNative(handle, uplo, n, A, lda, AP));
    }
    private static native int cublasCtrttpNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer AP);


    public static int cublasZtrttp(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer AP)
    {
        return checkResult(cublasZtrttpNative(handle, uplo, n, A, lda, AP));
    }
    private static native int cublasZtrttpNative(
        cublasHandle handle, 
        int uplo, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer AP);


    /** Batched LU - GETRF*/
    public static int cublasSgetrfBatched(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasSgetrfBatchedNative(handle, n, A, lda, P, info, batchSize));
    }
    private static native int cublasSgetrfBatchedNative(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer info, 
        int batchSize);


    public static int cublasDgetrfBatched(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasDgetrfBatchedNative(handle, n, A, lda, P, info, batchSize));
    }
    private static native int cublasDgetrfBatchedNative(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer info, 
        int batchSize);


    public static int cublasCgetrfBatched(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasCgetrfBatchedNative(handle, n, A, lda, P, info, batchSize));
    }
    private static native int cublasCgetrfBatchedNative(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer info, 
        int batchSize);


    public static int cublasZgetrfBatched(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasZgetrfBatchedNative(handle, n, A, lda, P, info, batchSize));
    }
    private static native int cublasZgetrfBatchedNative(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer info, 
        int batchSize);


    /** Batched inversion based on LU factorization from getrf */
    public static int cublasSgetriBatched(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer C, 
        int ldc, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasSgetriBatchedNative(handle, n, A, lda, P, C, ldc, info, batchSize));
    }
    private static native int cublasSgetriBatchedNative(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer C, 
        int ldc, 
        Pointer info, 
        int batchSize);


    public static int cublasDgetriBatched(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer C, 
        int ldc, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasDgetriBatchedNative(handle, n, A, lda, P, C, ldc, info, batchSize));
    }
    private static native int cublasDgetriBatchedNative(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer C, 
        int ldc, 
        Pointer info, 
        int batchSize);


    public static int cublasCgetriBatched(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer C, 
        int ldc, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasCgetriBatchedNative(handle, n, A, lda, P, C, ldc, info, batchSize));
    }
    private static native int cublasCgetriBatchedNative(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer C, 
        int ldc, 
        Pointer info, 
        int batchSize);


    public static int cublasZgetriBatched(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer C, 
        int ldc, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasZgetriBatchedNative(handle, n, A, lda, P, C, ldc, info, batchSize));
    }
    private static native int cublasZgetriBatchedNative(
        cublasHandle handle, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer P, 
        Pointer C, 
        int ldc, 
        Pointer info, 
        int batchSize);


    /** Batched solver based on LU factorization from getrf */
    public static int cublasSgetrsBatched(
        cublasHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer devIpiv, 
        Pointer Barray, 
        int ldb, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasSgetrsBatchedNative(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
    }
    private static native int cublasSgetrsBatchedNative(
        cublasHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer devIpiv, 
        Pointer Barray, 
        int ldb, 
        Pointer info, 
        int batchSize);


    public static int cublasDgetrsBatched(
        cublasHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer devIpiv, 
        Pointer Barray, 
        int ldb, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasDgetrsBatchedNative(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
    }
    private static native int cublasDgetrsBatchedNative(
        cublasHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer devIpiv, 
        Pointer Barray, 
        int ldb, 
        Pointer info, 
        int batchSize);


    public static int cublasCgetrsBatched(
        cublasHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer devIpiv, 
        Pointer Barray, 
        int ldb, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasCgetrsBatchedNative(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
    }
    private static native int cublasCgetrsBatchedNative(
        cublasHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer devIpiv, 
        Pointer Barray, 
        int ldb, 
        Pointer info, 
        int batchSize);


    public static int cublasZgetrsBatched(
        cublasHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer devIpiv, 
        Pointer Barray, 
        int ldb, 
        Pointer info, 
        int batchSize)
    {
        return checkResult(cublasZgetrsBatchedNative(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
    }
    private static native int cublasZgetrsBatchedNative(
        cublasHandle handle, 
        int trans, 
        int n, 
        int nrhs, 
        Pointer Aarray, 
        int lda, 
        Pointer devIpiv, 
        Pointer Barray, 
        int ldb, 
        Pointer info, 
        int batchSize);


    public static int cublasMigrateComputeType_new(
        cublasHandle handle, 
        int dataType, 
        int[] computeType)
    {
        return checkResult(cublasMigrateComputeType_newNative(handle, dataType, computeType));
    }
    private static native int cublasMigrateComputeType_newNative(
        cublasHandle handle, 
        int dataType, 
        int[] computeType);

}

