/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda
 *
 * Copyright (c) 2008-2015 Marco Hutter - http://www.jcuda.org
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


#include "JCublas.hpp"
#include "JCublas_common.hpp"
#include <iostream>
#include <string>
#include <map>

int jCublasStatus;

jfieldID cuComplex_x; // float
jfieldID cuComplex_y; // float

jfieldID cuDoubleComplex_x; // double
jfieldID cuDoubleComplex_y; // double

// Class and method ID for cuComplex and its constructor
jclass cuComplex_Class;
jmethodID cuComplex_Constructor;

// Class and method ID for cuDoubleComplex and its constructor
jclass cuDoubleComplex_Class;
jmethodID cuDoubleComplex_Constructor;

/**
 * Creates a global reference to the class with the given name and
 * stores it in the given jclass argument, and stores the no-args
 * constructor ID for this class in the given jmethodID.
 * Returns whether this initialization succeeded.
 */
bool init(JNIEnv *env, const char *className, jclass &globalCls, jmethodID &constructor)
{
    jclass cls = NULL;
    if (!init(env, cls, className)) return false;
    if (!init(env, cls, constructor, "<init>", "()V")) return false;

    globalCls = (jclass)env->NewGlobalRef(cls);
    if (globalCls == NULL)
    {
        Logger::log(LOG_ERROR, "Failed to create reference to class %s\n", className);
        return false;
    }
    return true;
}

/**
 * Called when the library is loaded. Will initialize all
 * required field and method IDs
 */
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved)
{
    JNIEnv *env = NULL;
    if (jvm->GetEnv((void **)&env, JNI_VERSION_1_4))
    {
        return JNI_ERR;
    }

    Logger::log(LOG_TRACE, "Initializing JCublas\n");

    jclass cls = NULL;

    // Initialize the JNIUtils and PointerUtils
    if (initJNIUtils(env) == JNI_ERR) return JNI_ERR;
    if (initPointerUtils(env) == JNI_ERR) return JNI_ERR;


    // Obtain the fieldIDs for cuComplex#x and cuComplex#y
    if (!init(env, cls, "jcuda/cuComplex")) return JNI_ERR;
    if (!init(env, cls, cuComplex_x, "x", "F")) return JNI_ERR;
    if (!init(env, cls, cuComplex_y, "y", "F")) return JNI_ERR;


    // Obtain the fieldIDs for cuDoubleComplex#x and cuDoubleComplex#y
    if (!init(env, cls, "jcuda/cuDoubleComplex")) return JNI_ERR;
    if (!init(env, cls, cuDoubleComplex_x, "x", "D")) return JNI_ERR;
    if (!init(env, cls, cuDoubleComplex_y, "y", "D")) return JNI_ERR;


    // Obtain the constructors
    if (!init(env, "jcuda/cuDoubleComplex", cuDoubleComplex_Class, cuDoubleComplex_Constructor)) return JNI_ERR;
    if (!init(env, "jcuda/cuComplex",       cuComplex_Class,       cuComplex_Constructor)) return JNI_ERR;


    return JNI_VERSION_1_4;

}









/**
 * Caches the class and field definitions of jcuda.jcublas.JCuComplex
 * and passes the call to Cublas
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    cublasInitNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasInitNative(JNIEnv *env, jclass cla)
{
    Logger::log(LOG_TRACE, "Initializing cublas\n");
    return cublasInit();
}


/**
 * Passes the call to Cublas
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    cublasShutdownNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasShutdownNative
  (JNIEnv *env, jclass cla)
{
    Logger::log(LOG_TRACE, "Shutting down cublas\n");
    return cublasShutdown();
}


/**
 * Passes the call to Cublas
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    cublasGetErrorNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasGetErrorNative
  (JNIEnv *env, jclass cla)
{
    int cublasError = cublasGetError();
    int result = cublasError;
    if (cublasError == CUBLAS_STATUS_SUCCESS)
    {
        result = jCublasStatus;
    }
    jCublasStatus = CUBLAS_STATUS_SUCCESS;
    return result;
}


/**
 * Passes the call to Cublas
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    cublasAllocNative
 * Signature: (IILjcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasAllocNative
  (JNIEnv *env, jclass cla, jint n, jint elemSize, jobject devicePtr)
{
    if (devicePtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devicePtr' is null for cublasAlloc");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Allocating %d elements of size %d for '%s'\n", n, elemSize, "devicePtr");

    void *nativeDevicePtr = 0;
    jCublasStatus = cublasAlloc(n, elemSize, &nativeDevicePtr);
    if (jCublasStatus == CUBLAS_STATUS_SUCCESS)
    {
        setPointer(env, devicePtr, (jlong)nativeDevicePtr);
    }
    return jCublasStatus;
}


/**
 * Passes the call to Cublas
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    cublasFreeNative
 * Signature: (Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasFreeNative
  (JNIEnv *env, jclass cla, jobject devicePtr)
{
    if (devicePtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devicePtr' is null for cublasFree");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Freeing device memory '%s'\n", "devicePtr");

    void *nativeDevicePtr = getPointer(env, devicePtr);
    jCublasStatus = cublasFree(nativeDevicePtr);
    return jCublasStatus;
}




//============================================================================
// Memory management functions


/*
 * Passes the call to Cublas
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    cublasSetVectorNative
 * Signature: (IILjcuda/Pointer;IILjcuda/Pointer;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasSetVectorNative
  (JNIEnv *env, jclass cla, jint n, jint elemSize, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSetVector");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasSetVector");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    void *deviceMemory = NULL;
    void *hostMemory = NULL;

    PointerData *xPointerData = initPointerData(env, x);
    if (xPointerData == NULL)
    {
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    deviceMemory = getPointer(env, y);

    Logger::log(LOG_TRACE, "Setting %d elements of size %d from java with inc %d to '%s' with inc %d\n",
        n, elemSize, incx, "y", incy);

    cublasStatus result = cublasSetVector(n, elemSize, (void*)xPointerData->getPointer(env), incx, deviceMemory, incy);

    if (!releasePointerData(env, xPointerData, JNI_ABORT)) return JCUBLAS_STATUS_INTERNAL_ERROR;
    return result;
}




/*
 * Passes the call to Cublas
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    cublasGetVectorNative
 * Signature: (IILjcuda/Pointer;IILjcuda/Pointer;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasGetVectorNative
  (JNIEnv *env, jclass cla, jint n, jint elemSize, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasGetVector");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasGetVector");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    void *deviceMemory = NULL;
    void *hostMemory = NULL;

    deviceMemory = getPointer(env, x);
    PointerData *yPointerData = initPointerData(env, y);
    if (yPointerData == NULL)
    {
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Getting %d elements of size %d from '%s' with inc %d to java with inc %d\n",
        n, elemSize, "x", incx, incy);

    cublasStatus result = cublasGetVector(n, elemSize, deviceMemory, incx, (void*)yPointerData->getPointer(env), incy);

    if (!releasePointerData(env, yPointerData)) return JCUBLAS_STATUS_INTERNAL_ERROR;
    return result;
}





/*
 * Passes the call to Cublas
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    cublasSetMatrixNative
 * Signature: (IIILjcuda/Pointer;IILjcuda/Pointer;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasSetMatrixNative
  (JNIEnv *env, jclass cla, jint rows, jint cols, jint elemSize, jobject A, jint lda, jobject B, jint ldb)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasSetMatrix");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasSetMatrix");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    void *deviceMemory = NULL;
    void *hostMemory = NULL;

    PointerData *APointerData = initPointerData(env, A);
    if (APointerData == NULL)
    {
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    deviceMemory = getPointer(env, B);

    Logger::log(LOG_TRACE, "Setting %dx%d elements of size %d from java with lda %d to '%s' with ldb %d\n",
        rows, cols, elemSize, lda, "B", ldb);

    cublasStatus result = cublasSetMatrix(rows, cols, elemSize, (void*)APointerData->getPointer(env), lda, deviceMemory, ldb);

    if (!releasePointerData(env, APointerData, JNI_ABORT)) return JCUBLAS_STATUS_INTERNAL_ERROR;
    return result;
}


/*
 * Passes the call to Cublas
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    cublasGetMatrixNative
 * Signature: (IIILjcuda/Pointer;IILjcuda/Pointer;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasGetMatrixNative
  (JNIEnv *env, jclass cla, jint rows, jint cols, jint elemSize, jobject A, jint lda, jobject B, jint ldb)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasGetMatrix");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasGetMatrix");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    void *deviceMemory = NULL;
    void *hostMemory = NULL;

    deviceMemory = getPointer(env, A);
    PointerData *BPointerData = initPointerData(env, B);
    if (BPointerData == NULL)
    {
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Getting %dx%d elements of size %d from '%s' with lda %d to java with ldb %d\n",
        rows, cols, elemSize, "A", lda, ldb);

    cublasStatus result = cublasGetMatrix(rows, cols, elemSize, deviceMemory, lda, (void*)BPointerData->getPointer(env), ldb);

    if (!releasePointerData(env, BPointerData)) return JCUBLAS_STATUS_INTERNAL_ERROR;
    return result;
}









//============================================================================
// Asynchronous Memory management functions


/*
 * Passes the call to Cublas
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    cublasSetVectorAsyncNative
 * Signature: (IILjcuda/Pointer;IILjcuda/Pointer;IILjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasSetVectorAsyncNative
  (JNIEnv *env, jclass cla, jint n, jint elemSize, jobject x, jint incx, jobject y, jint incy, jobject stream)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSetVectorAsync");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasSetVectorAsync");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    void *deviceMemory = NULL;
    void *hostMemory = NULL;
    cudaStream_t nativeStream = NULL;

    PointerData *xPointerData = initPointerData(env, x);
    if (xPointerData == NULL)
    {
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    deviceMemory = getPointer(env, y);

    nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    Logger::log(LOG_TRACE, "Setting %d elements of size %d from java with inc %d to '%s' with inc %d\n",
        n, elemSize, incx, "y", incy);

    cublasStatus result = cublasSetVectorAsync(n, elemSize, (void*)xPointerData->getPointer(env), incx, deviceMemory, incy, nativeStream);

    if (!releasePointerData(env, xPointerData, JNI_ABORT)) return JCUBLAS_STATUS_INTERNAL_ERROR;
    return result;
}




/*
 * Passes the call to Cublas
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    cublasGetVectorAsyncNative
 * Signature: (IILjcuda/Pointer;IILjcuda/Pointer;IILjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasGetVectorAsyncNative
  (JNIEnv *env, jclass cla, jint n, jint elemSize, jobject x, jint incx, jobject y, jint incy, jobject stream)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasGetVectorAsync");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasGetVectorAsync");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    void *deviceMemory = NULL;
    void *hostMemory = NULL;
    cudaStream_t nativeStream = NULL;

    deviceMemory = getPointer(env, x);
    PointerData *yPointerData = initPointerData(env, y);
    if (yPointerData == NULL)
    {
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    Logger::log(LOG_TRACE, "Getting %d elements of size %d from '%s' with inc %d to java with inc %d\n",
        n, elemSize, "x", incx, incy);

    cublasStatus result = cublasGetVectorAsync(n, elemSize, deviceMemory, incx, (void*)yPointerData->getPointer(env), incy, nativeStream);

    if (!releasePointerData(env, yPointerData)) return JCUBLAS_STATUS_INTERNAL_ERROR;
    return result;
}





/*
 * Passes the call to Cublas
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    cublasSetMatrixAsyncNative
 * Signature: (IIILjcuda/Pointer;IILjcuda/Pointer;IILjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasSetMatrixAsyncNative
  (JNIEnv *env, jclass cla, jint rows, jint cols, jint elemSize, jobject A, jint lda, jobject B, jint ldb, jobject stream)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasSetMatrixAsync");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasSetMatrixAsync");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    void *deviceMemory = NULL;
    void *hostMemory = NULL;
    cudaStream_t nativeStream = NULL;

    PointerData *APointerData = initPointerData(env, A);
    if (APointerData == NULL)
    {
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    deviceMemory = getPointer(env, B);

    nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    Logger::log(LOG_TRACE, "Setting %dx%d elements of size %d from java with lda %d to '%s' with ldb %d\n",
        rows, cols, elemSize, lda, "B", ldb);

    cublasStatus result = cublasSetMatrixAsync(rows, cols, elemSize, (void*)APointerData->getPointer(env), lda, deviceMemory, ldb, nativeStream);

    if (!releasePointerData(env, APointerData, JNI_ABORT)) return JCUBLAS_STATUS_INTERNAL_ERROR;
    return result;
}


/*
 * Passes the call to Cublas
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    cublasGetMatrixAsyncNative
 * Signature: (IIILjcuda/Pointer;IILjcuda/Pointer;IILjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasGetMatrixAsyncNative
  (JNIEnv *env, jclass cla, jint rows, jint cols, jint elemSize, jobject A, jint lda, jobject B, jint ldb, jobject stream)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasGetMatrixAsync");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasGetMatrixAsync");
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    void *deviceMemory = NULL;
    void *hostMemory = NULL;
    cudaStream_t nativeStream = NULL;

    deviceMemory = getPointer(env, A);
    PointerData *BPointerData = initPointerData(env, B);
    if (BPointerData == NULL)
    {
        return JCUBLAS_STATUS_INTERNAL_ERROR;
    }

    nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    Logger::log(LOG_TRACE, "Getting %dx%d elements of size %d from '%s' with lda %d to java with ldb %d\n",
        rows, cols, elemSize, "A", lda, ldb);

    cublasStatus result = cublasGetMatrixAsync(rows, cols, elemSize, deviceMemory, lda, (void*)BPointerData->getPointer(env), ldb, nativeStream);

    if (!releasePointerData(env, BPointerData)) return JCUBLAS_STATUS_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_jcublas_JCublas
 * Method:    cublasSetKernelStreamNative
 * Signature: (Ljcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasSetKernelStreamNative
  (JNIEnv *env, jclass cla, jobject stream)
{
    Logger::log(LOG_TRACE, "Setting the kernel stream\n");

    cudaStream_t nativeStream = NULL;
    nativeStream = (cudaStream_t)getNativePointerValue(env, stream);
    return cublasSetKernelStream(nativeStream);
}







/*
 * Set the log level
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    setLogLevelNative
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_setLogLevelNative
  (JNIEnv *env, jclass cla, jint logLevel)
{
    Logger::setLogLevel((LogLevel)logLevel);
}



/*
 * Prints the specified vector of single precision floating point elements
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    printVector
 * Signature: (ILjcuda/Pointer;)V
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_printVector
  (JNIEnv *env, jclass cla, jint n, jobject x)
{
    float *deviceMemory = (float*)getPointer(env, x);
    float *hostMemory = (float*)malloc(n * sizeof(float));
    cublasGetVector(n, 4, deviceMemory, 1, hostMemory, 1);
    LogLevel tempLogLevel = Logger::currentLogLevel;
    Logger::setLogLevel(LOG_INFO);
    for (int i=0; i<n; i++)
    {
        Logger::log(LOG_INFO, "%2.1f  ", hostMemory[i]);
    }
    Logger::log(LOG_INFO, "\n");
    Logger::setLogLevel(tempLogLevel);
    free(hostMemory);
}


/*
 * Prints the specified matrix of single precision floating point elements
 *
 * Class:     jcuda_jcublas_JCublas
 * Method:    printMatrix
 * Signature: (ILjcuda/Pointer;I)V
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_printMatrix
  (JNIEnv *env, jclass cla, jint cols, jobject A, jint lda)
{
    float *deviceMemory = (float*)getPointer(env, A);
    float *hostMemory = (float*)malloc(cols * lda * sizeof(float));
    cublasGetMatrix(lda, cols, 4, deviceMemory, lda, hostMemory, lda);
    LogLevel tempLogLevel = Logger::currentLogLevel;
    Logger::setLogLevel(LOG_INFO);
    for (int r=0; r<lda; r++)
    {
        for (int c=0; c<cols; c++)
        {
            Logger::log(LOG_INFO, "%2.1f  ", hostMemory[c * lda + r]);
        }
        Logger::log(LOG_INFO, "\n");
    }
    Logger::log(LOG_INFO, "\n");
    Logger::setLogLevel(tempLogLevel);
    free(hostMemory);
}


/**
 * Set the pointer to be the array elements that correspond to the given java
 * array. Returns true iff the array elements could be obtained, and the java
 * array had the expectedSize.
 *
 * Note that the array elements must later be released by calling
 * env->ReleaseFloatArrayElements(array, arrayElements, JNI_ABORT);
 */
bool getArrayElements(JNIEnv *env, jfloatArray array, float* &arrayElements, int expectedSize)
{
    int size = env->GetArrayLength(array);
    if (size != expectedSize)
    {
        Logger::log(LOG_ERROR, "Expected an array size of %d, but it has a size of %d\n", expectedSize, size);
        jCublasStatus = JCUBLAS_STATUS_INTERNAL_ERROR;
        return false;
    }
    arrayElements = env->GetFloatArrayElements(array, NULL);
    if (arrayElements == NULL)
    {
        Logger::log(LOG_ERROR, "Out of memory while obtaining array elements\n");
        jCublasStatus = JCUBLAS_STATUS_INTERNAL_ERROR;
        return false;
    }
    return true;
}

/**
 * Set the pointer to be the array elements that correspond to the given java
 * array. Returns true iff the array elements could be obtained, and the java
 * array had the expectedSize.
 *
 * Note that the array elements must later be released by calling
 * env->ReleaseDoubleArrayElements(array, arrayElements, JNI_ABORT);
 */
bool getDoubleArrayElements(JNIEnv *env, jdoubleArray array, double* &arrayElements, int expectedSize)
{
    int size = env->GetArrayLength(array);
    if (size != expectedSize)
    {
        Logger::log(LOG_ERROR, "Expected an array size of %d, but it has a size of %d\n", expectedSize, size);
        jCublasStatus = JCUBLAS_STATUS_INTERNAL_ERROR;
        return false;
    }
    arrayElements = env->GetDoubleArrayElements(array, NULL);
    if (arrayElements == NULL)
    {
        Logger::log(LOG_ERROR, "Out of memory while obtaining array elements\n");
        jCublasStatus = JCUBLAS_STATUS_INTERNAL_ERROR;
        return false;
    }
    return true;
}



//============================================================================
// Methods that are not handled by the code generator:



/**
 * <pre>
 * void
 * cublasSrotm (int n, float *x, int incx, float *y, int incy,
 *              const float* sparam)
 *
 * applies the modified Givens transformation, h, to the 2 x n matrix
 *
 *    ( transpose(x) )
 *    ( transpose(y) )
 *
 * The elements of x are in x[lx + i * incx], i = 0 to n-1, where lx = 1 if
 * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
 * incy. With sparam[0] = sflag, h has one of the following forms:
 *
 *        sflag = -1.0f   sflag = 0.0f    sflag = 1.0f    sflag = -2.0f
 *
 *        (sh00  sh01)    (1.0f  sh01)    (sh00  1.0f)    (1.0f  0.0f)
 *    h = (          )    (          )    (          )    (          )
 *        (sh10  sh11)    (sh10  1.0f)    (-1.0f sh11)    (0.0f  1.0f)
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 * y      single precision vector with n elements
 * incy   storage spacing between elements of y
 * sparam 5-element vector. sparam[0] is sflag described above. sparam[1]
 *        through sparam[4] contain the 2x2 rotation matrix h: sparam[1]
 *        contains sh00, sparam[2] contains sh10, sparam[3] contains sh01,
 *        and sprams[4] contains sh11.
 *
 * Output
 * ------
 * x     rotated vector x (unchanged if n <= 0)
 * y     rotated vector y (unchanged if n <= 0)
 *
 * Reference: http://www.netlib.org/blas/srotm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSrotmNative
  (JNIEnv *env, jclass cla, jint n, jobject x, jint incx, jobject y, jint incy, jfloatArray sparam)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSrotm");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasSrotm");
        return;
    }
    if (sparam == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sparam' is null for cublasSrotm");
        return;
    }

    void *deviceMemoryX = NULL;
    void *deviceMemoryY = NULL;
    float *sparamArrayElements = NULL;

    deviceMemoryX = getPointer(env, x);
    deviceMemoryY = getPointer(env, y);

    if (!getArrayElements(env, sparam, sparamArrayElements, 5)) return;

    Logger::log(LOG_TRACE, "Executing cublasSrotm(%d, '%s', %d, '%s', %d, [%f, %f, %f, %f, %f])\n",
        n, "x", incx,"y", incy, sparamArrayElements[0],
        sparamArrayElements[1], sparamArrayElements[2],
        sparamArrayElements[3], sparamArrayElements[4]);

    cublasSrotm(n, ((float*)deviceMemoryX), incx, ((float*)deviceMemoryY), incy, sparamArrayElements);

    env->ReleaseFloatArrayElements(sparam, sparamArrayElements, JNI_ABORT);
}

/**
 * <pre>
 * void
 * cublasSrotmg (float *psd1, float *psd2, float *psx1, const float *psy1,
 *                float *sparam)
 *
 * constructs the modified Givens transformation matrix h which zeros
 * the second component of the 2-vector transpose(sqrt(sd1)*sx1,sqrt(sd2)*sy1).
 * With sparam[0] = sflag, h has one of the following forms:
 *
 *        sflag = -1.0f   sflag = 0.0f    sflag = 1.0f    sflag = -2.0f
 *
 *        (sh00  sh01)    (1.0f  sh01)    (sh00  1.0f)    (1.0f  0.0f)
 *    h = (          )    (          )    (          )    (          )
 *        (sh10  sh11)    (sh10  1.0f)    (-1.0f sh11)    (0.0f  1.0f)
 *
 * sparam[1] through sparam[4] contain sh00, sh10, sh01, sh11,
 * respectively. Values of 1.0f, -1.0f, or 0.0f implied by the value
 * of sflag are not stored in sparam.
 *
 * Input
 * -----
 * sd1    single precision scalar
 * sd2    single precision scalar
 * sx1    single precision scalar
 * sy1    single precision scalar
 *
 * Output
 * ------
 * sd1    changed to represent the effect of the transformation
 * sd2    changed to represent the effect of the transformation
 * sx1    changed to represent the effect of the transformation
 * sparam 5-element vector. sparam[0] is sflag described above. sparam[1]
 *        through sparam[4] contain the 2x2 rotation matrix h: sparam[1]
 *        contains sh00, sparam[2] contains sh10, sparam[3] contains sh01,
 *        and sprams[4] contains sh11.
 *
 * Reference: http://www.netlib.org/blas/srotmg.f
 *
 * This functions does not set any error status.
 *</pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSrotmgNative
  (JNIEnv *env, jclass cla, jfloatArray sd1, jfloatArray sd2, jfloatArray sx1, jfloat sy1, jfloatArray sparam)
{
    if (sd1 == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sd1' is null for cublasSrotmg");
        return;
    }
    if (sd2 == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sd2' is null for cublasSrotmg");
        return;
    }
    if (sx1 == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sx1' is null for cublasSrotmg");
        return;
    }
    if (sparam == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sparam' is null for cublasSrotmg");
        return;
    }

    float *sd1ArrayElements = NULL;
    float *sd2ArrayElements = NULL;
    float *sx1ArrayElements = NULL;
    float *sparamArrayElements = NULL;

    if (!getArrayElements(env, sd1, sd1ArrayElements, 1)) return;
    if (!getArrayElements(env, sd2, sd2ArrayElements, 1)) return;
    if (!getArrayElements(env, sx1, sx1ArrayElements, 1)) return;
    if (!getArrayElements(env, sparam, sparamArrayElements, 5)) return;

    Logger::log(LOG_TRACE, "Executing cublasSrotmg(%f, %f, %f, %f, [%f, %f, %f, %f, %f])\n",
        sd1ArrayElements[0], sd2ArrayElements[0], sx1ArrayElements[0], sy1,
        sparamArrayElements[0], sparamArrayElements[1], sparamArrayElements[2],
        sparamArrayElements[3], sparamArrayElements[4]);

    cublasSrotmg(sd1ArrayElements, sd2ArrayElements,
        sx1ArrayElements, &sy1, sparamArrayElements);

    env->ReleaseFloatArrayElements(sparam, sparamArrayElements, 0);
    env->ReleaseFloatArrayElements(sd1, sd1ArrayElements, 0);
    env->ReleaseFloatArrayElements(sd2, sd2ArrayElements, 0);
    env->ReleaseFloatArrayElements(sx1, sx1ArrayElements, 0);
}





/**
 * </pre>
 * void
 * cublasDrotm (int n, double *x, int incx, double *y, int incy,
 *              const double* sparam)
 *
 * applies the modified Givens transformation, h, to the 2 x n matrix
 *
 *    ( transpose(x) )
 *    ( transpose(y) )
 *
 * The elements of x are in x[lx + i * incx], i = 0 to n-1, where lx = 1 if
 * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
 * incy. With sparam[0] = sflag, h has one of the following forms:
 *
 *        sflag = -1.0    sflag = 0.0     sflag = 1.0     sflag = -2.0
 *
 *        (sh00  sh01)    (1.0   sh01)    (sh00   1.0)    (1.0    0.0)
 *    h = (          )    (          )    (          )    (          )
 *        (sh10  sh11)    (sh10   1.0)    (-1.0  sh11)    (0.0    1.0)
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      double-precision vector with n elements
 * incx   storage spacing between elements of x
 * y      double-precision vector with n elements
 * incy   storage spacing between elements of y
 * sparam 5-element vector. sparam[0] is sflag described above. sparam[1]
 *        through sparam[4] contain the 2x2 rotation matrix h: sparam[1]
 *        contains sh00, sparam[2] contains sh10, sparam[3] contains sh01,
 *        and sprams[4] contains sh11.
 *
 * Output
 * ------
 * x     rotated vector x (unchanged if n <= 0)
 * y     rotated vector y (unchanged if n <= 0)
 *
 * Reference: http://www.netlib.org/blas/drotm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDrotmNative
  (JNIEnv *env, jclass cla, jint n, jobject x, jint incx, jobject y, jint incy, jdoubleArray sparam)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDrotm");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasDrotm");
        return;
    }
    if (sparam == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sparam' is null for cublasDrotm");
        return;
    }

    void *deviceMemoryX = NULL;
    void *deviceMemoryY = NULL;
    double *sparamArrayElements = NULL;

    deviceMemoryX = getPointer(env, x);
    deviceMemoryY = getPointer(env, y);

    if (!getDoubleArrayElements(env, sparam, sparamArrayElements, 5)) return;

    Logger::log(LOG_TRACE, "Executing cublasSrotm(%d, '%s', %d, '%s', %d, [%lf, %lf, %lf, %lf, %lf])\n",
        n, "x", incx, "y", incy, sparamArrayElements[0],
        sparamArrayElements[1], sparamArrayElements[2],
        sparamArrayElements[3], sparamArrayElements[4]);

    cublasDrotm(n, ((double*)deviceMemoryX), incx, ((double*)deviceMemoryY), incy, sparamArrayElements);

    env->ReleaseDoubleArrayElements(sparam, sparamArrayElements, JNI_ABORT);
}










/**
 * </pre>
 * void
 * cublasDrotmg (double *psd1, double *psd2, double *psx1, const double *psy1,
 *               double *sparam)
 *
 * constructs the modified Givens transformation matrix h which zeros
 * the second component of the 2-vector transpose(sqrt(sd1)*sx1,sqrt(sd2)*sy1).
 * With sparam[0] = sflag, h has one of the following forms:
 *
 *        sflag = -1.0    sflag = 0.0     sflag = 1.0     sflag = -2.0
 *
 *        (sh00  sh01)    (1.0   sh01)    (sh00   1.0)    (1.0    0.0)
 *    h = (          )    (          )    (          )    (          )
 *        (sh10  sh11)    (sh10   1.0)    (-1.0  sh11)    (0.0    1.0)
 *
 * sparam[1] through sparam[4] contain sh00, sh10, sh01, sh11,
 * respectively. Values of 1.0, -1.0, or 0.0 implied by the value
 * of sflag are not stored in sparam.
 *
 * Input
 * -----
 * sd1    single precision scalar
 * sd2    single precision scalar
 * sx1    single precision scalar
 * sy1    single precision scalar
 *
 * Output
 * ------
 * sd1    changed to represent the effect of the transformation
 * sd2    changed to represent the effect of the transformation
 * sx1    changed to represent the effect of the transformation
 * sparam 5-element vector. sparam[0] is sflag described above. sparam[1]
 *        through sparam[4] contain the 2x2 rotation matrix h: sparam[1]
 *        contains sh00, sparam[2] contains sh10, sparam[3] contains sh01,
 *        and sprams[4] contains sh11.
 *
 * Reference: http://www.netlib.org/blas/drotmg.f
 *
 * This functions does not set any error status.
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDrotmgNative
  (JNIEnv *env, jclass cla, jdoubleArray sd1, jdoubleArray sd2, jdoubleArray sx1, jdouble sy1, jdoubleArray sparam)
{
    if (sd1 == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sd1' is null for cublasDrotmg");
        return;
    }
    if (sd2 == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sd2' is null for cublasDrotmg");
        return;
    }
    if (sx1 == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sx1' is null for cublasDrotmg");
        return;
    }
    if (sparam == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sparam' is null for cublasDrotmg");
        return;
    }

    double *sd1ArrayElements = NULL;
    double *sd2ArrayElements = NULL;
    double *sx1ArrayElements = NULL;
    double *sparamArrayElements = NULL;

    if (!getDoubleArrayElements(env, sd1, sd1ArrayElements, 1)) return;
    if (!getDoubleArrayElements(env, sd2, sd2ArrayElements, 1)) return;
    if (!getDoubleArrayElements(env, sx1, sx1ArrayElements, 1)) return;
    if (!getDoubleArrayElements(env, sparam, sparamArrayElements, 5)) return;

    Logger::log(LOG_TRACE, "Executing cublasSrotmg(%lf, %lf, %lf, %lf, [%lf, %lf, %lf, %lf, %lf])\n",
        sd1ArrayElements[0], sd2ArrayElements[0], sx1ArrayElements[0], sy1,
        sparamArrayElements[0], sparamArrayElements[1], sparamArrayElements[2],
        sparamArrayElements[3], sparamArrayElements[4]);

    cublasDrotmg(sd1ArrayElements, sd2ArrayElements,
        sx1ArrayElements, &sy1, sparamArrayElements);

    env->ReleaseDoubleArrayElements(sparam, sparamArrayElements, 0);
    env->ReleaseDoubleArrayElements(sd1, sd1ArrayElements, 0);
    env->ReleaseDoubleArrayElements(sd2, sd2ArrayElements, 0);
    env->ReleaseDoubleArrayElements(sx1, sx1ArrayElements, 0);
}










//============================================================================
// Auto-generated part:

/**
 * <pre>
 * int
 * cublasIsamax (int n, const float *x, int incx)
 *
 * finds the smallest index of the maximum magnitude element of single
 * precision vector x; that is, the result is the first i, i = 0 to n - 1,
 * that maximizes abs(x[1 + i * incx])).
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns the smallest index (0 if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/isamax.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasIsamaxNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasIsamax");
        return 0;
    }
    float* nativeX;

    nativeX = (float*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasIsamax(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasIsamax(n, nativeX, incx);
}




/**
 * <pre>
 * int
 * cublasIsamin (int n, const float *x, int incx)
 *
 * finds the smallest index of the minimum magnitude element of single
 * precision vector x; that is, the result is the first i, i = 0 to n - 1,
 * that minimizes abs(x[1 + i * incx])).
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns the smallest index (0 if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/scilib/blass.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasIsaminNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasIsamin");
        return 0;
    }
    float* nativeX;

    nativeX = (float*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasIsamin(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasIsamin(n, nativeX, incx);
}




/**
 * <pre>
 * float
 * cublasSasum (int n, const float *x, int incx)
 *
 * computes the sum of the absolute values of the elements of single
 * precision vector x; that is, the result is the sum from i = 0 to n - 1 of
 * abs(x[1 + i * incx]).
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns the single precision sum of absolute values
 * (0 if n <= 0 or incx <= 0, or if an error occurs)
 *
 * Reference: http://www.netlib.org/blas/sasum.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jfloat JNICALL Java_jcuda_jcublas_JCublas_cublasSasumNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSasum");
        return 0.0;
    }
    float* nativeX;

    nativeX = (float*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasSasum(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasSasum(n, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasSaxpy (int n, float alpha, const float *x, int incx, float *y,
 *              int incy)
 *
 * multiplies single precision vector x by single precision scalar alpha
 * and adds the result to single precision vector y; that is, it overwrites
 * single precision y with single precision alpha * x + y. For i = 0 to n - 1,
 * it replaces y[ly + i * incy] with alpha * x[lx + i * incx] + y[ly + i *
 * incy], where lx = 1 if incx >= 0, else lx = 1 +(1 - n) * incx, and ly is
 * defined in a similar way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * alpha  single precision scalar multiplier
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 * y      single precision vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * y      single precision result (unchanged if n <= 0)
 *
 * Reference: http://www.netlib.org/blas/saxpy.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSaxpyNative
    (JNIEnv *env, jclass cls, jint n, jfloat alpha, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSaxpy");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasSaxpy");
        return;
    }
    float* nativeX;
    float* nativeY;

    nativeX = (float*)getPointer(env, x);
    nativeY = (float*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasSaxpy(%d, %f, '%s', %d, '%s', %d)\n",
        n, alpha, "x", incx, "y", incy);

    cublasSaxpy(n, alpha, nativeX, incx, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasScopy (int n, const float *x, int incx, float *y, int incy)
 *
 * copies the single precision vector x to the single precision vector y. For
 * i = 0 to n-1, copies x[lx + i * incx] to y[ly + i * incy], where lx = 1 if
 * incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a similar
 * way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 * y      single precision vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * y      contains single precision vector x
 *
 * Reference: http://www.netlib.org/blas/scopy.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasScopyNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasScopy");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasScopy");
        return;
    }
    float* nativeX;
    float* nativeY;

    nativeX = (float*)getPointer(env, x);
    nativeY = (float*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasScopy(%d, '%s', %d, '%s', %d)\n",
        n, "x", incx, "y", incy);

    cublasScopy(n, nativeX, incx, nativeY, incy);
}




/**
 * <pre>
 * float
 * cublasSdot (int n, const float *x, int incx, const float *y, int incy)
 *
 * computes the dot product of two single precision vectors. It returns the
 * dot product of the single precision vectors x and y if successful, and
 * 0.0f otherwise. It computes the sum for i = 0 to n - 1 of x[lx + i *
 * incx] * y[ly + i * incy], where lx = 1 if incx >= 0, else lx = 1 + (1 - n)
 * *incx, and ly is defined in a similar way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 * y      single precision vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * returns single precision dot product (zero if n <= 0)
 *
 * Reference: http://www.netlib.org/blas/sdot.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has nor been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to execute on GPU
 * </pre>
 */
JNIEXPORT jfloat JNICALL Java_jcuda_jcublas_JCublas_cublasSdotNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSdot");
        return 0.0;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasSdot");
        return 0.0;
    }
    float* nativeX;
    float* nativeY;

    nativeX = (float*)getPointer(env, x);
    nativeY = (float*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasSdot(%d, '%s', %d, '%s', %d)\n",
        n, "x", incx, "y", incy);

    return cublasSdot(n, nativeX, incx, nativeY, incy);
}




/**
 * <pre>
 * float
 * cublasSnrm2 (int n, const float *x, int incx)
 *
 * computes the Euclidean norm of the single precision n-vector x (with
 * storage increment incx). This code uses a multiphase model of
 * accumulation to avoid intermediate underflow and overflow.
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns Euclidian norm (0 if n <= 0 or incx <= 0, or if an error occurs)
 *
 * Reference: http://www.netlib.org/blas/snrm2.f
 * Reference: http://www.netlib.org/slatec/lin/snrm2.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jfloat JNICALL Java_jcuda_jcublas_JCublas_cublasSnrm2Native
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSnrm2");
        return 0.0;
    }
    float* nativeX;

    nativeX = (float*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasSnrm2(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasSnrm2(n, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasSrot (int n, float *x, int incx, float *y, int incy, float sc,
 *             float ss)
 *
 * multiplies a 2x2 matrix ( sc ss) with the 2xn matrix ( transpose(x) )
 *                         (-ss sc)                     ( transpose(y) )
 *
 * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if
 * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
 * incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 * y      single precision vector with n elements
 * incy   storage spacing between elements of y
 * sc     element of rotation matrix
 * ss     element of rotation matrix
 *
 * Output
 * ------
 * x      rotated vector x (unchanged if n <= 0)
 * y      rotated vector y (unchanged if n <= 0)
 *
 * Reference  http://www.netlib.org/blas/srot.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSrotNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy, jfloat sc, jfloat ss)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSrot");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasSrot");
        return;
    }
    float* nativeX;
    float* nativeY;

    nativeX = (float*)getPointer(env, x);
    nativeY = (float*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasSrot(%d, '%s', %d, '%s', %d, %f, %f)\n",
        n, "x", incx, "y", incy, sc, ss);

    cublasSrot(n, nativeX, incx, nativeY, incy, sc, ss);
}




/**
 * <pre>
 * void
 * cublasSrotg (float *host_sa, float *host_sb, float *host_sc, float *host_ss)
 *
 * constructs the Givens tranformation
 *
 *        ( sc  ss )
 *    G = (        ) ,  sc^2 + ss^2 = 1,
 *        (-ss  sc )
 *
 * which zeros the second entry of the 2-vector transpose(sa, sb).
 *
 * The quantity r = (+/-) sqrt (sa^2 + sb^2) overwrites sa in storage. The
 * value of sb is overwritten by a value z which allows sc and ss to be
 * recovered by the following algorithm:
 *
 *    if z=1          set sc = 0.0 and ss = 1.0
 *    if abs(z) < 1   set sc = sqrt(1-z^2) and ss = z
 *    if abs(z) > 1   set sc = 1/z and ss = sqrt(1-sc^2)
 *
 * The function srot (n, x, incx, y, incy, sc, ss) normally is called next
 * to apply the transformation to a 2 x n matrix.
 * Note that is function is provided for completeness and run exclusively
 * on the Host.
 *
 * Input
 * -----
 * sa     single precision scalar
 * sb     single precision scalar
 *
 * Output
 * ------
 * sa     single precision r
 * sb     single precision z
 * sc     single precision result
 * ss     single precision result
 *
 * Reference: http://www.netlib.org/blas/srotg.f
 *
 * This function does not set any error status.
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSrotgNative
    (JNIEnv *env, jclass cls, jobject host_sa, jobject host_sb, jobject host_sc, jobject host_ss)
{
    if (host_sa == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'host_sa' is null for cublasSrotg");
        return;
    }
    if (host_sb == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'host_sb' is null for cublasSrotg");
        return;
    }
    if (host_sc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'host_sc' is null for cublasSrotg");
        return;
    }
    if (host_ss == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'host_ss' is null for cublasSrotg");
        return;
    }
    float* nativeHOST_SA;
    float* nativeHOST_SB;
    float* nativeHOST_SC;
    float* nativeHOST_SS;

    nativeHOST_SA = (float*)getPointer(env, host_sa);
    nativeHOST_SB = (float*)getPointer(env, host_sb);
    nativeHOST_SC = (float*)getPointer(env, host_sc);
    nativeHOST_SS = (float*)getPointer(env, host_ss);

    Logger::log(LOG_TRACE, "Executing cublasSrotg('%s', '%s', '%s', '%s')\n",
        "host_sa", "host_sb", "host_sc", "host_ss");

    cublasSrotg(nativeHOST_SA, nativeHOST_SB, nativeHOST_SC, nativeHOST_SS);
}




/**
 * <pre>
 * void
 * sscal (int n, float alpha, float *x, int incx)
 *
 * replaces single precision vector x with single precision alpha * x. For i
 * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx],
 * where ix = 1 if incx >= 0, else ix = 1 + (1 - n) * incx.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * alpha  single precision scalar multiplier
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * x      single precision result (unchanged if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/sscal.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSscalNative
    (JNIEnv *env, jclass cls, jint n, jfloat alpha, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSscal");
        return;
    }
    float* nativeX;

    nativeX = (float*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasSscal(%d, %f, '%s', %d)\n",
        n, alpha, "x", incx);

    cublasSscal(n, alpha, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasSswap (int n, float *x, int incx, float *y, int incy)
 *
 * replaces single precision vector x with single precision alpha * x. For i
 * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx],
 * where ix = 1 if incx >= 0, else ix = 1 + (1 - n) * incx.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * alpha  single precision scalar multiplier
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * x      single precision result (unchanged if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/sscal.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSswapNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSswap");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasSswap");
        return;
    }
    float* nativeX;
    float* nativeY;

    nativeX = (float*)getPointer(env, x);
    nativeY = (float*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasSswap(%d, '%s', %d, '%s', %d)\n",
        n, "x", incx, "y", incy);

    cublasSswap(n, nativeX, incx, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasCaxpy (int n, cuComplex alpha, const cuComplex *x, int incx,
 *              cuComplex *y, int incy)
 *
 * multiplies single-complex vector x by single-complex scalar alpha and adds
 * the result to single-complex vector y; that is, it overwrites single-complex
 * y with single-complex alpha * x + y. For i = 0 to n - 1, it replaces
 * y[ly + i * incy] with alpha * x[lx + i * incx] + y[ly + i * incy], where
 * lx = 0 if incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a
 * similar way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * alpha  single-complex scalar multiplier
 * x      single-complex vector with n elements
 * incx   storage spacing between elements of x
 * y      single-complex vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * y      single-complex result (unchanged if n <= 0)
 *
 * Reference: http://www.netlib.org/blas/caxpy.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCaxpyNative
    (JNIEnv *env, jclass cls, jint n, jobject alpha, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCaxpy");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasCaxpy");
        return;
    }
    cuComplex* nativeX;
    cuComplex* nativeY;
    cuComplex complexAlpha;

    nativeX = (cuComplex*)getPointer(env, x);
    nativeY = (cuComplex*)getPointer(env, y);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCaxpy(%d, [%f,%f], '%s', %d, '%s', %d)\n",
        n, complexAlpha.x, complexAlpha.y, "x", incx, "y", incy);

    cublasCaxpy(n, complexAlpha, nativeX, incx, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasCcopy (int n, const cuComplex *x, int incx, cuComplex *y, int incy)
 *
 * copies the single-complex vector x to the single-complex vector y. For
 * i = 0 to n-1, copies x[lx + i * incx] to y[ly + i * incy], where lx = 1 if
 * incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a similar
 * way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      single-complex vector with n elements
 * incx   storage spacing between elements of x
 * y      single-complex vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * y      contains single complex vector x
 *
 * Reference: http://www.netlib.org/blas/ccopy.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCcopyNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCcopy");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasCcopy");
        return;
    }
    cuComplex* nativeX;
    cuComplex* nativeY;

    nativeX = (cuComplex*)getPointer(env, x);
    nativeY = (cuComplex*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasCcopy(%d, '%s', %d, '%s', %d)\n",
        n, "x", incx, "y", incy);

    cublasCcopy(n, nativeX, incx, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasZcopy (int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
 *
 * copies the double-complex vector x to the double-complex vector y. For
 * i = 0 to n-1, copies x[lx + i * incx] to y[ly + i * incy], where lx = 1 if
 * incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a similar
 * way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      double-complex vector with n elements
 * incx   storage spacing between elements of x
 * y      double-complex vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * y      contains double complex vector x
 *
 * Reference: http://www.netlib.org/blas/zcopy.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZcopyNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZcopy");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZcopy");
        return;
    }
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;

    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasZcopy(%d, '%s', %d, '%s', %d)\n",
        n, "x", incx, "y", incy);

    cublasZcopy(n, nativeX, incx, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasCscal (int n, cuComplex alpha, cuComplex *x, int incx)
 *
 * replaces single-complex vector x with single-complex alpha * x. For i
 * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx],
 * where ix = 1 if incx >= 0, else ix = 1 + (1 - n) * incx.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * alpha  single-complex scalar multiplier
 * x      single-complex vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * x      single-complex result (unchanged if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/cscal.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCscalNative
    (JNIEnv *env, jclass cls, jint n, jobject alpha, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCscal");
        return;
    }
    cuComplex* nativeX;
    cuComplex complexAlpha;

    nativeX = (cuComplex*)getPointer(env, x);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCscal(%d, [%f,%f], '%s', %d)\n",
        n, complexAlpha.x, complexAlpha.y, "x", incx);

    cublasCscal(n, complexAlpha, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasCrotg (cuComplex *host_ca, cuComplex cb, float *host_sc, cuComplex *host_cs)
 *
 * constructs the complex Givens tranformation
 *
 *        ( sc  cs )
 *    G = (        ) ,  sc^2 + cabs(cs)^2 = 1,
 *        (-cs  sc )
 *
 * which zeros the second entry of the complex 2-vector transpose(ca, cb).
 *
 * The quantity ca/cabs(ca)*norm(ca,cb) overwrites ca in storage. The
 * function crot (n, x, incx, y, incy, sc, cs) is normally called next
 * to apply the transformation to a 2 x n matrix.
 * Note that is function is provided for completeness and run exclusively
 * on the Host.
 *
 * Input
 * -----
 * ca     single-precision complex precision scalar
 * cb     single-precision complex scalar
 *
 * Output
 * ------
 * ca     single-precision complex ca/cabs(ca)*norm(ca,cb)
 * sc     single-precision cosine component of rotation matrix
 * cs     single-precision complex sine component of rotation matrix
 *
 * Reference: http://www.netlib.org/blas/crotg.f
 *
 * This function does not set any error status.
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCrotgNative
    (JNIEnv *env, jclass cls, jobject host_ca, jobject cb, jobject host_sc, jobject host_cs)
{
    if (host_ca == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'host_ca' is null for cublasCrotg");
        return;
    }
    if (host_sc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'host_sc' is null for cublasCrotg");
        return;
    }
    if (host_cs == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'host_cs' is null for cublasCrotg");
        return;
    }
    cuComplex* nativeHOST_CA;
    float* nativeHOST_SC;
    cuComplex* nativeHOST_CS;
    cuComplex complexCb;

    nativeHOST_CA = (cuComplex*)getPointer(env, host_ca);
    nativeHOST_SC = (float*)getPointer(env, host_sc);
    nativeHOST_CS = (cuComplex*)getPointer(env, host_cs);

    complexCb.x = env->GetFloatField(cb, cuComplex_x);
    complexCb.y = env->GetFloatField(cb, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCrotg('%s', [%f,%f], '%s', '%s')\n",
        "host_ca", complexCb.x, complexCb.y, "host_sc", "host_cs");

    cublasCrotg(nativeHOST_CA, complexCb, nativeHOST_SC, nativeHOST_CS);
}




/**
 * <pre>
 * void
 * cublasCrot (int n, cuComplex *x, int incx, cuComplex *y, int incy, float sc,
 *             cuComplex cs)
 *
 * multiplies a 2x2 matrix ( sc       cs) with the 2xn matrix ( transpose(x) )
 *                         (-conj(cs) sc)                     ( transpose(y) )
 *
 * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if
 * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
 * incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      single-precision complex vector with n elements
 * incx   storage spacing between elements of x
 * y      single-precision complex vector with n elements
 * incy   storage spacing between elements of y
 * sc     single-precision cosine component of rotation matrix
 * cs     single-precision complex sine component of rotation matrix
 *
 * Output
 * ------
 * x      rotated single-precision complex vector x (unchanged if n <= 0)
 * y      rotated single-precision complex vector y (unchanged if n <= 0)
 *
 * Reference: http://netlib.org/lapack/explore-html/crot.f.html
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCrotNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy, jfloat c, jobject s)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCrot");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasCrot");
        return;
    }
    cuComplex* nativeX;
    cuComplex* nativeY;
    cuComplex complexS;

    nativeX = (cuComplex*)getPointer(env, x);
    nativeY = (cuComplex*)getPointer(env, y);

    complexS.x = env->GetFloatField(s, cuComplex_x);
    complexS.y = env->GetFloatField(s, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCrot(%d, '%s', %d, '%s', %d, %f, [%f,%f])\n",
        n, "x", incx, "y", incy, c, complexS.x, complexS.y);

    cublasCrot(n, nativeX, incx, nativeY, incy, c, complexS);
}




/**
 * <pre>
 * void
 * csrot (int n, cuComplex *x, int incx, cuCumplex *y, int incy, float c,
 *        float s)
 *
 * multiplies a 2x2 rotation matrix ( c s) with a 2xn matrix ( transpose(x) )
 *                                  (-s c)                   ( transpose(y) )
 *
 * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if
 * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
 * incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      single-precision complex vector with n elements
 * incx   storage spacing between elements of x
 * y      single-precision complex vector with n elements
 * incy   storage spacing between elements of y
 * c      cosine component of rotation matrix
 * s      sine component of rotation matrix
 *
 * Output
 * ------
 * x      rotated vector x (unchanged if n <= 0)
 * y      rotated vector y (unchanged if n <= 0)
 *
 * Reference  http://www.netlib.org/blas/csrot.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCsrotNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy, jfloat c, jfloat s)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCsrot");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasCsrot");
        return;
    }
    cuComplex* nativeX;
    cuComplex* nativeY;

    nativeX = (cuComplex*)getPointer(env, x);
    nativeY = (cuComplex*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasCsrot(%d, '%s', %d, '%s', %d, %f, %f)\n",
        n, "x", incx, "y", incy, c, s);

    cublasCsrot(n, nativeX, incx, nativeY, incy, c, s);
}




/**
 * <pre>
 * void
 * cublasCsscal (int n, float alpha, cuComplex *x, int incx)
 *
 * replaces single-complex vector x with single-complex alpha * x. For i
 * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx],
 * where ix = 1 if incx >= 0, else ix = 1 + (1 - n) * incx.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * alpha  single precision scalar multiplier
 * x      single-complex vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * x      single-complex result (unchanged if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/csscal.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCsscalNative
    (JNIEnv *env, jclass cls, jint n, jfloat alpha, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCsscal");
        return;
    }
    cuComplex* nativeX;

    nativeX = (cuComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasCsscal(%d, %f, '%s', %d)\n",
        n, alpha, "x", incx);

    cublasCsscal(n, alpha, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasCswap (int n, const cuComplex *x, int incx, cuComplex *y, int incy)
 *
 * interchanges the single-complex vector x with the single-complex vector y.
 * For i = 0 to n-1, interchanges x[lx + i * incx] with y[ly + i * incy], where
 * lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a
 * similar way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      single-complex vector with n elements
 * incx   storage spacing between elements of x
 * y      single-complex vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * x      contains-single complex vector y
 * y      contains-single complex vector x
 *
 * Reference: http://www.netlib.org/blas/cswap.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCswapNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCswap");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasCswap");
        return;
    }
    cuComplex* nativeX;
    cuComplex* nativeY;

    nativeX = (cuComplex*)getPointer(env, x);
    nativeY = (cuComplex*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasCswap(%d, '%s', %d, '%s', %d)\n",
        n, "x", incx, "y", incy);

    cublasCswap(n, nativeX, incx, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasZswap (int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
 *
 * interchanges the double-complex vector x with the double-complex vector y.
 * For i = 0 to n-1, interchanges x[lx + i * incx] with y[ly + i * incy], where
 * lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a
 * similar way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      double-complex vector with n elements
 * incx   storage spacing between elements of x
 * y      double-complex vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * x      contains-double complex vector y
 * y      contains-double complex vector x
 *
 * Reference: http://www.netlib.org/blas/zswap.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZswapNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZswap");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZswap");
        return;
    }
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;

    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasZswap(%d, '%s', %d, '%s', %d)\n",
        n, "x", incx, "y", incy);

    cublasZswap(n, nativeX, incx, nativeY, incy);
}




/**
 * <pre>
 * cuComplex
 * cdotu (int n, const cuComplex *x, int incx, const cuComplex *y, int incy)
 *
 * computes the dot product of two single-complex vectors. It returns the
 * dot product of the single-complex vectors x and y if successful, and complex
 * zero otherwise. It computes the sum for i = 0 to n - 1 of x[lx + i * incx] *
 * y[ly + i * incy], where lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx;
 * ly is defined in a similar way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      single-complex vector with n elements
 * incx   storage spacing between elements of x
 * y      single-complex vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * returns single-complex dot product (zero if n <= 0)
 *
 * Reference: http://www.netlib.org/blas/cdotu.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has nor been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to execute on GPU
 * </pre>
 */
JNIEXPORT jobject JNICALL Java_jcuda_jcublas_JCublas_cublasCdotuNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCdotu");
        return NULL;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasCdotu");
        return NULL;
    }
    cuComplex* nativeX;
    cuComplex* nativeY;

    nativeX = (cuComplex*)getPointer(env, x);
    nativeY = (cuComplex*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasCdotu(%d, '%s', %d, '%s', %d)\n",
        n, "x", incx, "y", incy);

    cuComplex nativeResult = cublasCdotu(n, nativeX, incx, nativeY, incy);

    jobject result = env->NewObject(cuComplex_Class, cuComplex_Constructor);
    if (env->ExceptionCheck())
    {
        return NULL;
    }
    env->SetFloatField(result, cuComplex_x, nativeResult.x);
    env->SetFloatField(result, cuComplex_y, nativeResult.y);
    return result;
}




/**
 * <pre>
 * cuComplex
 * cublasCdotc (int n, const cuComplex *x, int incx, const cuComplex *y,
 *              int incy)
 *
 * computes the dot product of two single-complex vectors. It returns the
 * dot product of the single-complex vectors x and y if successful, and complex
 * zero otherwise. It computes the sum for i = 0 to n - 1 of x[lx + i * incx] *
 * y[ly + i * incy], where lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx;
 * ly is defined in a similar way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      single-complex vector with n elements
 * incx   storage spacing between elements of x
 * y      single-complex vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * returns single-complex dot product (zero if n <= 0)
 *
 * Reference: http://www.netlib.org/blas/cdotc.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has nor been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to execute on GPU
 * </pre>
 */
JNIEXPORT jobject JNICALL Java_jcuda_jcublas_JCublas_cublasCdotcNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCdotc");
        return NULL;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasCdotc");
        return NULL;
    }
    cuComplex* nativeX;
    cuComplex* nativeY;

    nativeX = (cuComplex*)getPointer(env, x);
    nativeY = (cuComplex*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasCdotc(%d, '%s', %d, '%s', %d)\n",
        n, "x", incx, "y", incy);

    cuComplex nativeResult = cublasCdotc(n, nativeX, incx, nativeY, incy);

    jobject result = env->NewObject(cuComplex_Class, cuComplex_Constructor);
    if (env->ExceptionCheck())
    {
        return NULL;
    }
    env->SetFloatField(result, cuComplex_x, nativeResult.x);
    env->SetFloatField(result, cuComplex_y, nativeResult.y);
    return result;

}




/**
 * <pre>
 * int
 * cublasIcamax (int n, const float *x, int incx)
 *
 * finds the smallest index of the element having maximum absolute value
 * in single-complex vector x; that is, the result is the first i, i = 0
 * to n - 1 that maximizes abs(real(x[1+i*incx]))+abs(imag(x[1 + i * incx])).
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      single-complex vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns the smallest index (0 if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/icamax.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasIcamaxNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasIcamax");
        return 0;
    }
    cuComplex* nativeX;

    nativeX = (cuComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasIcamax(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasIcamax(n, nativeX, incx);
}




/**
 * <pre>
 * int
 * cublasIcamin (int n, const float *x, int incx)
 *
 * finds the smallest index of the element having minimum absolute value
 * in single-complex vector x; that is, the result is the first i, i = 0
 * to n - 1 that minimizes abs(real(x[1+i*incx]))+abs(imag(x[1 + i * incx])).
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      single-complex vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns the smallest index (0 if n <= 0 or incx <= 0)
 *
 * Reference: see ICAMAX.
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasIcaminNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasIcamin");
        return 0;
    }
    cuComplex* nativeX;

    nativeX = (cuComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasIcamin(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasIcamin(n, nativeX, incx);
}




/**
 * <pre>
 * float
 * cublasScasum (int n, const cuDouble *x, int incx)
 *
 * takes the sum of the absolute values of a complex vector and returns a
 * single precision result. Note that this is not the L1 norm of the vector.
 * The result is the sum from 0 to n-1 of abs(real(x[ix+i*incx])) +
 * abs(imag(x(ix+i*incx))), where ix = 1 if incx <= 0, else ix = 1+(1-n)*incx.
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      single-complex vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns the single precision sum of absolute values of real and imaginary
 * parts (0 if n <= 0 or incx <= 0, or if an error occurs)
 *
 * Reference: http://www.netlib.org/blas/scasum.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jfloat JNICALL Java_jcuda_jcublas_JCublas_cublasScasumNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasScasum");
        return 0.0;
    }
    cuComplex* nativeX;

    nativeX = (cuComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasScasum(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasScasum(n, nativeX, incx);
}




/**
 * <pre>
 * float
 * cublasScnrm2 (int n, const cuComplex *x, int incx)
 *
 * computes the Euclidean norm of the single-complex n-vector x. This code
 * uses simple scaling to avoid intermediate underflow and overflow.
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      single-complex vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns Euclidian norm (0 if n <= 0 or incx <= 0, or if an error occurs)
 *
 * Reference: http://www.netlib.org/blas/scnrm2.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jfloat JNICALL Java_jcuda_jcublas_JCublas_cublasScnrm2Native
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasScnrm2");
        return 0.0;
    }
    cuComplex* nativeX;

    nativeX = (cuComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasScnrm2(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasScnrm2(n, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasZaxpy (int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx,
 *              cuDoubleComplex *y, int incy)
 *
 * multiplies double-complex vector x by double-complex scalar alpha and adds
 * the result to double-complex vector y; that is, it overwrites double-complex
 * y with double-complex alpha * x + y. For i = 0 to n - 1, it replaces
 * y[ly + i * incy] with alpha * x[lx + i * incx] + y[ly + i * incy], where
 * lx = 0 if incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a
 * similar way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * alpha  double-complex scalar multiplier
 * x      double-complex vector with n elements
 * incx   storage spacing between elements of x
 * y      double-complex vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * y      double-complex result (unchanged if n <= 0)
 *
 * Reference: http://www.netlib.org/blas/zaxpy.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZaxpyNative
    (JNIEnv *env, jclass cls, jint n, jobject alpha, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZaxpy");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZaxpy");
        return;
    }
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;
    cuDoubleComplex dobuleComplexAlpha;

    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZaxpy(%d, [%lf,%lf], '%s', %d, '%s', %d)\n",
        n, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "x", incx, "y", incy);

    cublasZaxpy(n, dobuleComplexAlpha, nativeX, incx, nativeY, incy);
}




/**
 * <pre>
 * cuDoubleComplex
 * zdotu (int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy)
 *
 * computes the dot product of two double-complex vectors. It returns the
 * dot product of the double-complex vectors x and y if successful, and double-complex
 * zero otherwise. It computes the sum for i = 0 to n - 1 of x[lx + i * incx] *
 * y[ly + i * incy], where lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx;
 * ly is defined in a similar way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      double-complex vector with n elements
 * incx   storage spacing between elements of x
 * y      double-complex vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * returns double-complex dot product (zero if n <= 0)
 *
 * Reference: http://www.netlib.org/blas/zdotu.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has nor been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to execute on GPU
 * </pre>
 */
JNIEXPORT jobject JNICALL Java_jcuda_jcublas_JCublas_cublasZdotuNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZdotu");
        return NULL;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZdotu");
        return NULL;
    }
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;

    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasZdotu(%d, '%s', %d, '%s', %d)\n",
        n, "x", incx, "y", incy);

    cuDoubleComplex nativeResult = cublasZdotu(n, nativeX, incx, nativeY, incy);

    jobject result = env->NewObject(cuDoubleComplex_Class, cuDoubleComplex_Constructor);
    if (env->ExceptionCheck())
    {
        return NULL;
    }
    env->SetDoubleField(result, cuDoubleComplex_x, nativeResult.x);
    env->SetDoubleField(result, cuDoubleComplex_y, nativeResult.y);
    return result;
}




/**
 * <pre>
 * cuDoubleComplex
 * cublasZdotc (int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy)
 *
 * computes the dot product of two double-precision complex vectors. It returns the
 * dot product of the double-precision complex vectors conjugate(x) and y if successful,
 * and double-precision complex zero otherwise. It computes the
 * sum for i = 0 to n - 1 of conjugate(x[lx + i * incx]) *  y[ly + i * incy],
 * where lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx;
 * ly is defined in a similar way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      double-precision complex vector with n elements
 * incx   storage spacing between elements of x
 * y      double-precision complex vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * returns double-complex dot product (zero if n <= 0)
 *
 * Reference: http://www.netlib.org/blas/zdotc.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has nor been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to execute on GPU
 * </pre>
 */
JNIEXPORT jobject JNICALL Java_jcuda_jcublas_JCublas_cublasZdotcNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZdotc");
        return NULL;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZdotc");
        return NULL;
    }
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;

    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasZdotc(%d, '%s', %d, '%s', %d)\n",
        n, "x", incx, "y", incy);

    cuDoubleComplex nativeResult = cublasZdotc(n, nativeX, incx, nativeY, incy);

    jobject result = env->NewObject(cuDoubleComplex_Class, cuDoubleComplex_Constructor);
    if (env->ExceptionCheck())
    {
        return NULL;
    }
    env->SetDoubleField(result, cuDoubleComplex_x, nativeResult.x);
    env->SetDoubleField(result, cuDoubleComplex_y, nativeResult.y);
    return result;
}




/**
 * <pre>
 * void
 * cublasZscal (int n, cuComplex alpha, cuComplex *x, int incx)
 *
 * replaces double-complex vector x with double-complex alpha * x. For i
 * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx],
 * where ix = 1 if incx >= 0, else ix = 1 + (1 - n) * incx.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * alpha  double-complex scalar multiplier
 * x      double-complex vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * x      double-complex result (unchanged if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/zscal.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZscalNative
    (JNIEnv *env, jclass cls, jint n, jobject alpha, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZscal");
        return;
    }
    cuDoubleComplex* nativeX;
    cuDoubleComplex dobuleComplexAlpha;

    nativeX = (cuDoubleComplex*)getPointer(env, x);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZscal(%d, [%lf,%lf], '%s', %d)\n",
        n, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "x", incx);

    cublasZscal(n, dobuleComplexAlpha, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasZdscal (int n, double alpha, cuDoubleComplex *x, int incx)
 *
 * replaces double-complex vector x with double-complex alpha * x. For i
 * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx],
 * where ix = 1 if incx >= 0, else ix = 1 + (1 - n) * incx.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * alpha  double precision scalar multiplier
 * x      double-complex vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * x      double-complex result (unchanged if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/zdscal.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZdscalNative
    (JNIEnv *env, jclass cls, jint n, jdouble alpha, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZdscal");
        return;
    }
    cuDoubleComplex* nativeX;

    nativeX = (cuDoubleComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasZdscal(%d, %lf, '%s', %d)\n",
        n, alpha, "x", incx);

    cublasZdscal(n, alpha, nativeX, incx);
}




/**
 * <pre>
 * double
 * cublasDznrm2 (int n, const cuDoubleComplex *x, int incx)
 *
 * computes the Euclidean norm of the double precision complex n-vector x. This code
 * uses simple scaling to avoid intermediate underflow and overflow.
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      double-complex vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns Euclidian norm (0 if n <= 0 or incx <= 0, or if an error occurs)
 *
 * Reference: http://www.netlib.org/blas/dznrm2.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jdouble JNICALL Java_jcuda_jcublas_JCublas_cublasDznrm2Native
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDznrm2");
        return 0.0;
    }
    cuDoubleComplex* nativeX;

    nativeX = (cuDoubleComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasDznrm2(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasDznrm2(n, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasZrotg (cuDoubleComplex *host_ca, cuDoubleComplex cb, double *host_sc, double *host_cs)
 *
 * constructs the complex Givens tranformation
 *
 *        ( sc  cs )
 *    G = (        ) ,  sc^2 + cabs(cs)^2 = 1,
 *        (-cs  sc )
 *
 * which zeros the second entry of the complex 2-vector transpose(ca, cb).
 *
 * The quantity ca/cabs(ca)*norm(ca,cb) overwrites ca in storage. The
 * function crot (n, x, incx, y, incy, sc, cs) is normally called next
 * to apply the transformation to a 2 x n matrix.
 * Note that is function is provided for completeness and run exclusively
 * on the Host.
 *
 * Input
 * -----
 * ca     double-precision complex precision scalar
 * cb     double-precision complex scalar
 *
 * Output
 * ------
 * ca     double-precision complex ca/cabs(ca)*norm(ca,cb)
 * sc     double-precision cosine component of rotation matrix
 * cs     double-precision complex sine component of rotation matrix
 *
 * Reference: http://www.netlib.org/blas/zrotg.f
 *
 * This function does not set any error status.
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZrotgNative
    (JNIEnv *env, jclass cls, jobject host_ca, jobject cb, jobject host_sc, jobject host_cs)
{
    if (host_ca == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'host_ca' is null for cublasZrotg");
        return;
    }
    if (host_sc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'host_sc' is null for cublasZrotg");
        return;
    }
    if (host_cs == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'host_cs' is null for cublasZrotg");
        return;
    }
    cuDoubleComplex* nativeHOST_CA;
    double* nativeHOST_SC;
    cuDoubleComplex* nativeHOST_CS;
    cuDoubleComplex dobuleComplexCb;

    nativeHOST_CA = (cuDoubleComplex*)getPointer(env, host_ca);
    nativeHOST_SC = (double*)getPointer(env, host_sc);
    nativeHOST_CS = (cuDoubleComplex*)getPointer(env, host_cs);

    dobuleComplexCb.x = env->GetDoubleField(cb, cuDoubleComplex_x);
    dobuleComplexCb.y = env->GetDoubleField(cb, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZrotg('%s', [%lf,%lf], '%s', '%s')\n",
        "host_ca", dobuleComplexCb.x, dobuleComplexCb.y, "host_sc", "host_cs");

    cublasZrotg(nativeHOST_CA, dobuleComplexCb, nativeHOST_SC, nativeHOST_CS);
}




/**
 * <pre>
 * cublasZrot (int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, double sc,
 *             cuDoubleComplex cs)
 *
 * multiplies a 2x2 matrix ( sc       cs) with the 2xn matrix ( transpose(x) )
 *                         (-conj(cs) sc)                     ( transpose(y) )
 *
 * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if
 * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
 * incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      double-precision complex vector with n elements
 * incx   storage spacing between elements of x
 * y      double-precision complex vector with n elements
 * incy   storage spacing between elements of y
 * sc     double-precision cosine component of rotation matrix
 * cs     double-precision complex sine component of rotation matrix
 *
 * Output
 * ------
 * x      rotated double-precision complex vector x (unchanged if n <= 0)
 * y      rotated double-precision complex vector y (unchanged if n <= 0)
 *
 * Reference: http://netlib.org/lapack/explore-html/zrot.f.html
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZrotNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy, jdouble sc, jobject cs)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZrot");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZrot");
        return;
    }
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;
    cuDoubleComplex dobuleComplexCs;

    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);

    dobuleComplexCs.x = env->GetDoubleField(cs, cuDoubleComplex_x);
    dobuleComplexCs.y = env->GetDoubleField(cs, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZrot(%d, '%s', %d, '%s', %d, %lf, [%lf,%lf])\n",
        n, "x", incx, "y", incy, sc, dobuleComplexCs.x, dobuleComplexCs.y);

    cublasZrot(n, nativeX, incx, nativeY, incy, sc, dobuleComplexCs);
}




/**
 * <pre>
 * void
 * zdrot (int n, cuDoubleComplex *x, int incx, cuCumplex *y, int incy, double c,
 *        double s)
 *
 * multiplies a 2x2 matrix ( c s) with the 2xn matrix ( transpose(x) )
 *                         (-s c)                     ( transpose(y) )
 *
 * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if
 * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
 * incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      double-precision complex vector with n elements
 * incx   storage spacing between elements of x
 * y      double-precision complex vector with n elements
 * incy   storage spacing between elements of y
 * c      cosine component of rotation matrix
 * s      sine component of rotation matrix
 *
 * Output
 * ------
 * x      rotated vector x (unchanged if n <= 0)
 * y      rotated vector y (unchanged if n <= 0)
 *
 * Reference  http://www.netlib.org/blas/zdrot.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZdrotNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy, jdouble c, jdouble s)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZdrot");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZdrot");
        return;
    }
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;

    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasZdrot(%d, '%s', %d, '%s', %d, %lf, %lf)\n",
        n, "x", incx, "y", incy, c, s);

    cublasZdrot(n, nativeX, incx, nativeY, incy, c, s);
}




/**
 * <pre>
 * int
 * cublasIzamax (int n, const double *x, int incx)
 *
 * finds the smallest index of the element having maximum absolute value
 * in double-complex vector x; that is, the result is the first i, i = 0
 * to n - 1 that maximizes abs(real(x[1+i*incx]))+abs(imag(x[1 + i * incx])).
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      double-complex vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns the smallest index (0 if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/izamax.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasIzamaxNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasIzamax");
        return 0;
    }
    cuDoubleComplex* nativeX;

    nativeX = (cuDoubleComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasIzamax(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasIzamax(n, nativeX, incx);
}




/**
 * <pre>
 * int
 * cublasIzamin (int n, const cuDoubleComplex *x, int incx)
 *
 * finds the smallest index of the element having minimum absolute value
 * in double-complex vector x; that is, the result is the first i, i = 0
 * to n - 1 that minimizes abs(real(x[1+i*incx]))+abs(imag(x[1 + i * incx])).
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      double-complex vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns the smallest index (0 if n <= 0 or incx <= 0)
 *
 * Reference: Analogous to IZAMAX, see there.
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasIzaminNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasIzamin");
        return 0;
    }
    cuDoubleComplex* nativeX;

    nativeX = (cuDoubleComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasIzamin(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasIzamin(n, nativeX, incx);
}




/**
 * <pre>
 * double
 * cublasDzasum (int n, const cuDoubleComplex *x, int incx)
 *
 * takes the sum of the absolute values of a complex vector and returns a
 * double precision result. Note that this is not the L1 norm of the vector.
 * The result is the sum from 0 to n-1 of abs(real(x[ix+i*incx])) +
 * abs(imag(x(ix+i*incx))), where ix = 1 if incx <= 0, else ix = 1+(1-n)*incx.
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      double-complex vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns the double precision sum of absolute values of real and imaginary
 * parts (0 if n <= 0 or incx <= 0, or if an error occurs)
 *
 * Reference: http://www.netlib.org/blas/dzasum.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jdouble JNICALL Java_jcuda_jcublas_JCublas_cublasDzasumNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDzasum");
        return 0.0;
    }
    cuDoubleComplex* nativeX;

    nativeX = (cuDoubleComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasDzasum(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasDzasum(n, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasSgbmv (char trans, int m, int n, int kl, int ku, float alpha,
 *              const float *A, int lda, const float *x, int incx, float beta,
 *              float *y, int incy)
 *
 * performs one of the matrix-vector operations
 *
 *    y = alpha*op(A)*x + beta*y,  op(A)=A or op(A) = transpose(A)
 *
 * alpha and beta are single precision scalars. x and y are single precision
 * vectors. A is an m by n band matrix consisting of single precision elements
 * with kl sub-diagonals and ku super-diagonals.
 *
 * Input
 * -----
 * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
 *        't', 'C', or 'c', op(A) = transpose(A)
 * m      specifies the number of rows of the matrix A. m must be at least
 *        zero.
 * n      specifies the number of columns of the matrix A. n must be at least
 *        zero.
 * kl     specifies the number of sub-diagonals of matrix A. It must be at
 *        least zero.
 * ku     specifies the number of super-diagonals of matrix A. It must be at
 *        least zero.
 * alpha  single precision scalar multiplier applied to op(A).
 * A      single precision array of dimensions (lda, n). The leading
 *        (kl + ku + 1) x n part of the array A must contain the band matrix A,
 *        supplied column by column, with the leading diagonal of the matrix
 *        in row (ku + 1) of the array, the first super-diagonal starting at
 *        position 2 in row ku, the first sub-diagonal starting at position 1
 *        in row (ku + 2), and so on. Elements in the array A that do not
 *        correspond to elements in the band matrix (such as the top left
 *        ku x ku triangle) are not referenced.
 * lda    leading dimension of A. lda must be at least (kl + ku + 1).
 * x      single precision array of length at least (1+(n-1)*abs(incx)) when
 *        trans == 'N' or 'n' and at least (1+(m-1)*abs(incx)) otherwise.
 * incx   storage spacing between elements of x. incx must not be zero.
 * beta   single precision scalar multiplier applied to vector y. If beta is
 *        zero, y is not read.
 * y      single precision array of length at least (1+(m-1)*abs(incy)) when
 *        trans == 'N' or 'n' and at least (1+(n-1)*abs(incy)) otherwise. If
 *        beta is zero, y is not read.
 * incy   storage spacing between elements of y. incy must not be zero.
 *
 * Output
 * ------
 * y      updated according to y = alpha*op(A)*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/sgbmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n, kl, or ku < 0; if incx or incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSgbmvNative
    (JNIEnv *env, jclass cls, jchar trans, jint m, jint n, jint kl, jint ku, jfloat alpha, jobject A, jint lda, jobject x, jint incx, jfloat beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasSgbmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSgbmv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasSgbmv");
        return;
    }
    float* nativeA;
    float* nativeX;
    float* nativeY;

    nativeA = (float*)getPointer(env, A);
    nativeX = (float*)getPointer(env, x);
    nativeY = (float*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasSgbmv(%c, %d, %d, %d, %d, %f, '%s', %d, '%s', %d, %f, '%s', %d)\n",
        trans, m, n, kl, ku, alpha, "A", lda, "x", incx, beta, "y", incy);

    cublasSgbmv((char)trans, m, n, kl, ku, alpha, nativeA, lda, nativeX, incx, beta, nativeY, incy);
}




/**
 * <pre>
 * cublasSgemv (char trans, int m, int n, float alpha, const float *A, int lda,
 *              const float *x, int incx, float beta, float *y, int incy)
 *
 * performs one of the matrix-vector operations
 *
 *    y = alpha * op(A) * x + beta * y,
 *
 * where op(A) is one of
 *
 *    op(A) = A   or   op(A) = transpose(A)
 *
 * where alpha and beta are single precision scalars, x and y are single
 * precision vectors, and A is an m x n matrix consisting of single precision
 * elements. Matrix A is stored in column major format, and lda is the leading
 * dimension of the two-dimensional array in which A is stored.
 *
 * Input
 * -----
 * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If trans =
 *        trans = 't', 'T', 'c', or 'C', op(A) = transpose(A)
 * m      specifies the number of rows of the matrix A. m must be at least
 *        zero.
 * n      specifies the number of columns of the matrix A. n must be at least
 *        zero.
 * alpha  single precision scalar multiplier applied to op(A).
 * A      single precision array of dimensions (lda, n) if trans = 'n' or
 *        'N'), and of dimensions (lda, m) otherwise. lda must be at least
 *        max(1, m) and at least max(1, n) otherwise.
 * lda    leading dimension of two-dimensional array used to store matrix A
 * x      single precision array of length at least (1 + (n - 1) * abs(incx))
 *        when trans = 'N' or 'n' and at least (1 + (m - 1) * abs(incx))
 *        otherwise.
 * incx   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 * beta   single precision scalar multiplier applied to vector y. If beta
 *        is zero, y is not read.
 * y      single precision array of length at least (1 + (m - 1) * abs(incy))
 *        when trans = 'N' or 'n' and at least (1 + (n - 1) * abs(incy))
 *        otherwise.
 * incy   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 *
 * Output
 * ------
 * y      updated according to alpha * op(A) * x + beta * y
 *
 * Reference: http://www.netlib.org/blas/sgemv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0, or if incx or incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSgemvNative
    (JNIEnv *env, jclass cls, jchar trans, jint m, jint n, jfloat alpha, jobject A, jint lda, jobject x, jint incx, jfloat beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasSgemv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSgemv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasSgemv");
        return;
    }
    float* nativeA;
    float* nativeX;
    float* nativeY;

    nativeA = (float*)getPointer(env, A);
    nativeX = (float*)getPointer(env, x);
    nativeY = (float*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasSgemv(%c, %d, %d, %f, '%s', %d, '%s', %d, %f, '%s', %d)\n",
        trans, m, n, alpha, "A", lda, "x", incx, beta, "y", incy);

    cublasSgemv((char)trans, m, n, alpha, nativeA, lda, nativeX, incx, beta, nativeY, incy);
}




/**
 * <pre>
 * cublasSger (int m, int n, float alpha, const float *x, int incx,
 *             const float *y, int incy, float *A, int lda)
 *
 * performs the symmetric rank 1 operation
 *
 *    A = alpha * x * transpose(y) + A,
 *
 * where alpha is a single precision scalar, x is an m element single
 * precision vector, y is an n element single precision vector, and A
 * is an m by n matrix consisting of single precision elements. Matrix A
 * is stored in column major format, and lda is the leading dimension of
 * the two-dimensional array used to store A.
 *
 * Input
 * -----
 * m      specifies the number of rows of the matrix A. It must be at least
 *        zero.
 * n      specifies the number of columns of the matrix A. It must be at
 *        least zero.
 * alpha  single precision scalar multiplier applied to x * transpose(y)
 * x      single precision array of length at least (1 + (m - 1) * abs(incx))
 * incx   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 * y      single precision array of length at least (1 + (n - 1) * abs(incy))
 * incy   specifies the storage spacing between elements of y. incy must not
 *        be zero.
 * A      single precision array of dimensions (lda, n).
 * lda    leading dimension of two-dimensional array used to store matrix A
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * transpose(y) + A
 *
 * Reference: http://www.netlib.org/blas/sger.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSgerNative
    (JNIEnv *env, jclass cls, jint m, jint n, jfloat alpha, jobject x, jint incx, jobject y, jint incy, jobject A, jint lda)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSger");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasSger");
        return;
    }
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasSger");
        return;
    }
    float* nativeX;
    float* nativeY;
    float* nativeA;

    nativeX = (float*)getPointer(env, x);
    nativeY = (float*)getPointer(env, y);
    nativeA = (float*)getPointer(env, A);

    Logger::log(LOG_TRACE, "Executing cublasSger(%d, %d, %f, '%s', %d, '%s', %d, '%s', %d)\n",
        m, n, alpha, "x", incx, "y", incy, "A", lda);

    cublasSger(m, n, alpha, nativeX, incx, nativeY, incy, nativeA, lda);
}




/**
 * <pre>
 * void
 * cublasSsbmv (char uplo, int n, int k, float alpha, const float *A, int lda,
 *              const float *x, int incx, float beta, float *y, int incy)
 *
 * performs the matrix-vector operation
 *
 *     y := alpha*A*x + beta*y
 *
 * alpha and beta are single precision scalars. x and y are single precision
 * vectors with n elements. A is an n x n symmetric band matrix consisting
 * of single precision elements, with k super-diagonals and the same number
 * of sub-diagonals.
 *
 * Input
 * -----
 * uplo   specifies whether the upper or lower triangular part of the symmetric
 *        band matrix A is being supplied. If uplo == 'U' or 'u', the upper
 *        triangular part is being supplied. If uplo == 'L' or 'l', the lower
 *        triangular part is being supplied.
 * n      specifies the number of rows and the number of columns of the
 *        symmetric matrix A. n must be at least zero.
 * k      specifies the number of super-diagonals of matrix A. Since the matrix
 *        is symmetric, this is also the number of sub-diagonals. k must be at
 *        least zero.
 * alpha  single precision scalar multiplier applied to A*x.
 * A      single precision array of dimensions (lda, n). When uplo == 'U' or
 *        'u', the leading (k + 1) x n part of array A must contain the upper
 *        triangular band of the symmetric matrix, supplied column by column,
 *        with the leading diagonal of the matrix in row (k+1) of the array,
 *        the first super-diagonal starting at position 2 in row k, and so on.
 *        The top left k x k triangle of the array A is not referenced. When
 *        uplo == 'L' or 'l', the leading (k + 1) x n part of the array A must
 *        contain the lower triangular band part of the symmetric matrix,
 *        supplied column by column, with the leading diagonal of the matrix in
 *        row 1 of the array, the first sub-diagonal starting at position 1 in
 *        row 2, and so on. The bottom right k x k triangle of the array A is
 *        not referenced.
 * lda    leading dimension of A. lda must be at least (k + 1).
 * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * beta   single precision scalar multiplier applied to vector y. If beta is
 *        zero, y is not read.
 * y      single precision array of length at least (1 + (n - 1) * abs(incy)).
 *        If beta is zero, y is not read.
 * incy   storage spacing between elements of y. incy must not be zero.
 *
 * Output
 * ------
 * y      updated according to alpha*A*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/ssbmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_INVALID_VALUE    if k or n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSsbmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jint k, jfloat alpha, jobject A, jint lda, jobject x, jint incx, jfloat beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasSsbmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSsbmv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasSsbmv");
        return;
    }
    float* nativeA;
    float* nativeX;
    float* nativeY;

    nativeA = (float*)getPointer(env, A);
    nativeX = (float*)getPointer(env, x);
    nativeY = (float*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasSsbmv(%c, %d, %d, %f, '%s', %d, '%s', %d, %f, '%s', %d)\n",
        uplo, n, k, alpha, "A", lda, "x", incx, beta, "y", incy);

    cublasSsbmv((char)uplo, n, k, alpha, nativeA, lda, nativeX, incx, beta, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasSspmv (char uplo, int n, float alpha, const float *AP, const float *x,
 *              int incx, float beta, float *y, int incy)
 *
 * performs the matrix-vector operation
 *
 *    y = alpha * A * x + beta * y
 *
 * Alpha and beta are single precision scalars, and x and y are single
 * precision vectors with n elements. A is a symmetric n x n matrix
 * consisting of single precision elements that is supplied in packed form.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array AP. If uplo == 'U' or 'u', then the upper
 *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then
 *        the lower triangular part of A is supplied in AP.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  single precision scalar multiplier applied to A*x.
 * AP     single precision array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * beta   single precision scalar multiplier applied to vector y;
 * y      single precision array of length at least (1 + (n - 1) * abs(incy)).
 *        If beta is zero, y is not read.
 * incy   storage spacing between elements of y. incy must not be zero.
 *
 * Output
 * ------
 * y      updated according to y = alpha*A*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/sspmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSspmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jfloat alpha, jobject AP, jobject x, jint incx, jfloat beta, jobject y, jint incy)
{
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasSspmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSspmv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasSspmv");
        return;
    }
    float* nativeAP;
    float* nativeX;
    float* nativeY;

    nativeAP = (float*)getPointer(env, AP);
    nativeX = (float*)getPointer(env, x);
    nativeY = (float*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasSspmv(%c, %d, %f, '%s', '%s', %d, %f, '%s', %d)\n",
        uplo, n, alpha, "AP", "x", incx, beta, "y", incy);

    cublasSspmv((char)uplo, n, alpha, nativeAP, nativeX, incx, beta, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasSspr (char uplo, int n, float alpha, const float *x, int incx,
 *             float *AP)
 *
 * performs the symmetric rank 1 operation
 *
 *    A = alpha * x * transpose(x) + A,
 *
 * where alpha is a single precision scalar and x is an n element single
 * precision vector. A is a symmetric n x n matrix consisting of single
 * precision elements that is supplied in packed form.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array AP. If uplo == 'U' or 'u', then the upper
 *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then
 *        the lower triangular part of A is supplied in AP.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  single precision scalar multiplier applied to x * transpose(x).
 * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * AP     single precision array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * transpose(x) + A
 *
 * Reference: http://www.netlib.org/blas/sspr.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSsprNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jfloat alpha, jobject x, jint incx, jobject AP)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSspr");
        return;
    }
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasSspr");
        return;
    }
    float* nativeX;
    float* nativeAP;

    nativeX = (float*)getPointer(env, x);
    nativeAP = (float*)getPointer(env, AP);

    Logger::log(LOG_TRACE, "Executing cublasSspr(%c, %d, %f, '%s', %d, '%s')\n",
        uplo, n, alpha, "x", incx, "AP");

    cublasSspr((char)uplo, n, alpha, nativeX, incx, nativeAP);
}




/**
 * <pre>
 * void
 * cublasSspr2 (char uplo, int n, float alpha, const float *x, int incx,
 *              const float *y, int incy, float *AP)
 *
 * performs the symmetric rank 2 operation
 *
 *    A = alpha*x*transpose(y) + alpha*y*transpose(x) + A,
 *
 * where alpha is a single precision scalar, and x and y are n element single
 * precision vectors. A is a symmetric n x n matrix consisting of single
 * precision elements that is supplied in packed form.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array A. If uplo == 'U' or 'u', then only the
 *        upper triangular part of A may be referenced and the lower triangular
 *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
 *        triangular part of A may be referenced and the upper triangular part
 *        of A is inferred.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  single precision scalar multiplier applied to x * transpose(y) +
 *        y * transpose(x).
 * x      single precision array of length at least (1 + (n - 1) * abs (incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * y      single precision array of length at least (1 + (n - 1) * abs (incy)).
 * incy   storage spacing between elements of y. incy must not be zero.
 * AP     single precision array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 *
 * Output
 * ------
 * A      updated according to A = alpha*x*transpose(y)+alpha*y*transpose(x)+A
 *
 * Reference: http://www.netlib.org/blas/sspr2.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSspr2Native
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jfloat alpha, jobject x, jint incx, jobject y, jint incy, jobject AP)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSspr2");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasSspr2");
        return;
    }
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasSspr2");
        return;
    }
    float* nativeX;
    float* nativeY;
    float* nativeAP;

    nativeX = (float*)getPointer(env, x);
    nativeY = (float*)getPointer(env, y);
    nativeAP = (float*)getPointer(env, AP);

    Logger::log(LOG_TRACE, "Executing cublasSspr2(%c, %d, %f, '%s', %d, '%s', %d, '%s')\n",
        uplo, n, alpha, "x", incx, "y", incy, "AP");

    cublasSspr2((char)uplo, n, alpha, nativeX, incx, nativeY, incy, nativeAP);
}




/**
 * <pre>
 * void
 * cublasSsymv (char uplo, int n, float alpha, const float *A, int lda,
 *              const float *x, int incx, float beta, float *y, int incy)
 *
 * performs the matrix-vector operation
 *
 *     y = alpha*A*x + beta*y
 *
 * Alpha and beta are single precision scalars, and x and y are single
 * precision vectors, each with n elements. A is a symmetric n x n matrix
 * consisting of single precision elements that is stored in either upper or
 * lower storage mode.
 *
 * Input
 * -----
 * uplo   specifies whether the upper or lower triangular part of the array A
 *        is to be referenced. If uplo == 'U' or 'u', the symmetric matrix A
 *        is stored in upper storage mode, i.e. only the upper triangular part
 *        of A is to be referenced while the lower triangular part of A is to
 *        be inferred. If uplo == 'L' or 'l', the symmetric matrix A is stored
 *        in lower storage mode, i.e. only the lower triangular part of A is
 *        to be referenced while the upper triangular part of A is to be
 *        inferred.
 * n      specifies the number of rows and the number of columns of the
 *        symmetric matrix A. n must be at least zero.
 * alpha  single precision scalar multiplier applied to A*x.
 * A      single precision array of dimensions (lda, n). If uplo == 'U' or 'u',
 *        the leading n x n upper triangular part of the array A must contain
 *        the upper triangular part of the symmetric matrix and the strictly
 *        lower triangular part of A is not referenced. If uplo == 'L' or 'l',
 *        the leading n x n lower triangular part of the array A must contain
 *        the lower triangular part of the symmetric matrix and the strictly
 *        upper triangular part of A is not referenced.
 * lda    leading dimension of A. It must be at least max (1, n).
 * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * beta   single precision scalar multiplier applied to vector y.
 * y      single precision array of length at least (1 + (n - 1) * abs(incy)).
 *        If beta is zero, y is not read.
 * incy   storage spacing between elements of y. incy must not be zero.
 *
 * Output
 * ------
 * y      updated according to y = alpha*A*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/ssymv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSsymvNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jfloat alpha, jobject A, jint lda, jobject x, jint incx, jfloat beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasSsymv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSsymv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasSsymv");
        return;
    }
    float* nativeA;
    float* nativeX;
    float* nativeY;

    nativeA = (float*)getPointer(env, A);
    nativeX = (float*)getPointer(env, x);
    nativeY = (float*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasSsymv(%c, %d, %f, '%s', %d, '%s', %d, %f, '%s', %d)\n",
        uplo, n, alpha, "A", lda, "x", incx, beta, "y", incy);

    cublasSsymv((char)uplo, n, alpha, nativeA, lda, nativeX, incx, beta, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasSsyr (char uplo, int n, float alpha, const float *x, int incx,
 *             float *A, int lda)
 *
 * performs the symmetric rank 1 operation
 *
 *    A = alpha * x * transpose(x) + A,
 *
 * where alpha is a single precision scalar, x is an n element single
 * precision vector and A is an n x n symmetric matrix consisting of
 * single precision elements. Matrix A is stored in column major format,
 * and lda is the leading dimension of the two-dimensional array
 * containing A.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or
 *        the lower triangular part of array A. If uplo = 'U' or 'u',
 *        then only the upper triangular part of A may be referenced.
 *        If uplo = 'L' or 'l', then only the lower triangular part of
 *        A may be referenced.
 * n      specifies the number of rows and columns of the matrix A. It
 *        must be at least 0.
 * alpha  single precision scalar multiplier applied to x * transpose(x)
 * x      single precision array of length at least (1 + (n - 1) * abs(incx))
 * incx   specifies the storage spacing between elements of x. incx must
 *        not be zero.
 * A      single precision array of dimensions (lda, n). If uplo = 'U' or
 *        'u', then A must contain the upper triangular part of a symmetric
 *        matrix, and the strictly lower triangular part is not referenced.
 *        If uplo = 'L' or 'l', then A contains the lower triangular part
 *        of a symmetric matrix, and the strictly upper triangular part is
 *        not referenced.
 * lda    leading dimension of the two-dimensional array containing A. lda
 *        must be at least max(1, n).
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * transpose(x) + A
 *
 * Reference: http://www.netlib.org/blas/ssyr.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSsyrNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jfloat alpha, jobject x, jint incx, jobject A, jint lda)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSsyr");
        return;
    }
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasSsyr");
        return;
    }
    float* nativeX;
    float* nativeA;

    nativeX = (float*)getPointer(env, x);
    nativeA = (float*)getPointer(env, A);

    Logger::log(LOG_TRACE, "Executing cublasSsyr(%c, %d, %f, '%s', %d, '%s', %d)\n",
        uplo, n, alpha, "x", incx, "A", lda);

    cublasSsyr((char)uplo, n, alpha, nativeX, incx, nativeA, lda);
}




/**
 * <pre>
 * void
 * cublasSsyr2 (char uplo, int n, float alpha, const float *x, int incx,
 *              const float *y, int incy, float *A, int lda)
 *
 * performs the symmetric rank 2 operation
 *
 *    A = alpha*x*transpose(y) + alpha*y*transpose(x) + A,
 *
 * where alpha is a single precision scalar, x and y are n element single
 * precision vector and A is an n by n symmetric matrix consisting of single
 * precision elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array A. If uplo == 'U' or 'u', then only the
 *        upper triangular part of A may be referenced and the lower triangular
 *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
 *        triangular part of A may be referenced and the upper triangular part
 *        of A is inferred.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  single precision scalar multiplier applied to x * transpose(y) +
 *        y * transpose(x).
 * x      single precision array of length at least (1 + (n - 1) * abs (incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * y      single precision array of length at least (1 + (n - 1) * abs (incy)).
 * incy   storage spacing between elements of y. incy must not be zero.
 * A      single precision array of dimensions (lda, n). If uplo == 'U' or 'u',
 *        then A must contains the upper triangular part of a symmetric matrix,
 *        and the strictly lower triangular parts is not referenced. If uplo ==
 *        'L' or 'l', then A contains the lower triangular part of a symmetric
 *        matrix, and the strictly upper triangular part is not referenced.
 * lda    leading dimension of A. It must be at least max(1, n).
 *
 * Output
 * ------
 * A      updated according to A = alpha*x*transpose(y)+alpha*y*transpose(x)+A
 *
 * Reference: http://www.netlib.org/blas/ssyr2.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSsyr2Native
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jfloat alpha, jobject x, jint incx, jobject y, jint incy, jobject A, jint lda)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasSsyr2");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasSsyr2");
        return;
    }
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasSsyr2");
        return;
    }
    float* nativeX;
    float* nativeY;
    float* nativeA;

    nativeX = (float*)getPointer(env, x);
    nativeY = (float*)getPointer(env, y);
    nativeA = (float*)getPointer(env, A);

    Logger::log(LOG_TRACE, "Executing cublasSsyr2(%c, %d, %f, '%s', %d, '%s', %d, '%s', %d)\n",
        uplo, n, alpha, "x", incx, "y", incy, "A", lda);

    cublasSsyr2((char)uplo, n, alpha, nativeX, incx, nativeY, incy, nativeA, lda);
}




/**
 * <pre>
 * void
 * cublasStbmv (char uplo, char trans, char diag, int n, int k, const float *A,
 *              int lda, float *x, int incx)
 *
 * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A
 * or op(A) = transpose(A). x is an n-element single precision vector, and A is
 * an n x n, unit or non-unit upper or lower triangular band matrix consisting
 * of single precision elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix A is an upper or lower triangular band
 *        matrix. If uplo == 'U' or 'u', A is an upper triangular band matrix.
 *        If uplo == 'L' or 'l', A is a lower triangular band matrix.
 * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
 *        't', 'C', or 'c', op(A) = transpose(A).
 * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
 *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
 *        is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero. In the current implementation n must not exceed 4070.
 * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
 *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
 *        'l', k specifies the number of sub-diagonals. k must at least be
 *        zero.
 * A      single precision array of dimension (lda, n). If uplo == 'U' or 'u',
 *        the leading (k + 1) x n part of the array A must contain the upper
 *        triangular band matrix, supplied column by column, with the leading
 *        diagonal of the matrix in row (k + 1) of the array, the first
 *        super-diagonal starting at position 2 in row k, and so on. The top
 *        left k x k triangle of the array A is not referenced. If uplo == 'L'
 *        or 'l', the leading (k + 1) x n part of the array A must constain the
 *        lower triangular band matrix, supplied column by column, with the
 *        leading diagonal of the matrix in row 1 of the array, the first
 *        sub-diagonal startingat position 1 in row 2, and so on. The bottom
 *        right k x k triangle of the array is not referenced.
 * lda    is the leading dimension of A. It must be at least (k + 1).
 * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the source vector. On exit, x is overwritten
 *        with the result vector.
 * incx   specifies the storage spacing for elements of x. incx must not be
 *        zero.
 *
 * Output
 * ------
 * x      updated according to x = op(A) * x
 *
 * Reference: http://www.netlib.org/blas/stbmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, k < 0, or incx == 0
 * CUBLAS_STATUS_ALLOC_FAILED     if function cannot allocate enough internal scratch vector memory
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasStbmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jint k, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasStbmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasStbmv");
        return;
    }
    float* nativeA;
    float* nativeX;

    nativeA = (float*)getPointer(env, A);
    nativeX = (float*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasStbmv(%c, %c, %c, %d, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, k, "A", lda, "x", incx);

    cublasStbmv((char)uplo, (char)trans, (char)diag, n, k, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void cublasStbsv (char uplo, char trans, char diag, int n, int k,
 *                   const float *A, int lda, float *X, int incx)
 *
 * solves one of the systems of equations op(A)*x = b, where op(A) is either
 * op(A) = A or op(A) = transpose(A). b and x are n-element vectors, and A is
 * an n x n unit or non-unit, upper or lower triangular band matrix with k + 1
 * diagonals. No test for singularity or near-singularity is included in this
 * function. Such tests must be performed before calling this function.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix is an upper or lower triangular band
 *        matrix as follows: If uplo == 'U' or 'u', A is an upper triangular
 *        band matrix. If uplo == 'L' or 'l', A is a lower triangular band
 *        matrix.
 * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
 *        't', 'C', or 'c', op(A) = transpose(A).
 * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
 *        assumed to be unit triangular; thas is, diagonal elements are not
 *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
 *        assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
 *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
 *        'l', k specifies the number of sub-diagonals. k must be at least
 *        zero.
 * A      single precision array of dimension (lda, n). If uplo == 'U' or 'u',
 *        the leading (k + 1) x n part of the array A must contain the upper
 *        triangular band matrix, supplied column by column, with the leading
 *        diagonal of the matrix in row (k + 1) of the array, the first super-
 *        diagonal starting at position 2 in row k, and so on. The top left
 *        k x k triangle of the array A is not referenced. If uplo == 'L' or
 *        'l', the leading (k + 1) x n part of the array A must constain the
 *        lower triangular band matrix, supplied column by column, with the
 *        leading diagonal of the matrix in row 1 of the array, the first
 *        sub-diagonal starting at position 1 in row 2, and so on. The bottom
 *        right k x k triangle of the array is not referenced.
 * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the n-element right-hand side vector b. On exit,
 *        it is overwritten with the solution vector x.
 * incx   storage spacing between elements of x. incx must not be zero.
 *
 * Output
 * ------
 * x      updated to contain the solution vector x that solves op(A) * x = b.
 *
 * Reference: http://www.netlib.org/blas/stbsv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0, n < 0 or n > 4070
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasStbsvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jint k, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasStbsv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasStbsv");
        return;
    }
    float* nativeA;
    float* nativeX;

    nativeA = (float*)getPointer(env, A);
    nativeX = (float*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasStbsv(%c, %c, %c, %d, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, k, "A", lda, "x", incx);

    cublasStbsv((char)uplo, (char)trans, (char)diag, n, k, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasStpmv (char uplo, char trans, char diag, int n, const float *AP,
 *              float *x, int incx);
 *
 * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
 * or op(A) = transpose(A). x is an n element single precision vector, and A
 * is an n x n, unit or non-unit, upper or lower triangular matrix composed
 * of single precision elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix. If uplo == 'U' or 'u', then A is an upper triangular matrix.
 *        If uplo == 'L' or 'l', then A is a lower triangular matrix.
 * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
 *        't', 'C', or 'c', op(A) = transpose(A)
 * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
 *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
 *        is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * AP     single precision array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i <= j, then A[i,j] is stored in AP[i+(j*(j+1)/2)]. If
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the source vector. On exit, x is overwritten
 *        with the result vector.
 * incx   specifies the storage spacing for elements of x. incx must not be
 *        zero.
 *
 * Output
 * ------
 * x      updated according to x = op(A) * x,
 *
 * Reference: http://www.netlib.org/blas/stpmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
 * CUBLAS_STATUS_ALLOC_FAILED     if function cannot allocate enough internal scratch vector memory
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasStpmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject AP, jobject x, jint incx)
{
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasStpmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasStpmv");
        return;
    }
    float* nativeAP;
    float* nativeX;

    nativeAP = (float*)getPointer(env, AP);
    nativeX = (float*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasStpmv(%c, %c, %c, %d, '%s', '%s', %d)\n",
        uplo, trans, diag, n, "AP", "x", incx);

    cublasStpmv((char)uplo, (char)trans, (char)diag, n, nativeAP, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasStpsv (char uplo, char trans, char diag, int n, const float *AP,
 *              float *X, int incx)
 *
 * solves one of the systems of equations op(A)*x = b, where op(A) is either
 * op(A) = A or op(A) = transpose(A). b and x are n element vectors, and A is
 * an n x n unit or non-unit, upper or lower triangular matrix. No test for
 * singularity or near-singularity is included in this function. Such tests
 * must be performed before calling this function.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix is an upper or lower triangular matrix
 *        as follows: If uplo == 'U' or 'u', A is an upper triangluar matrix.
 *        If uplo == 'L' or 'l', A is a lower triangular matrix.
 * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
 *        't', 'C', or 'c', op(A) = transpose(A).
 * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
 *        assumed to be unit triangular; thas is, diagonal elements are not
 *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
 *        assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero. In the current implementation n must not exceed 4070.
 * AP     single precision array with at least ((n*(n+1))/2) elements. If uplo
 *        == 'U' or 'u', the array AP contains the upper triangular matrix A,
 *        packed sequentially, column by column; that is, if i <= j, then
 *        A[i,j] is stored is AP[i+(j*(j+1)/2)]. If uplo == 'L' or 'L', the
 *        array AP contains the lower triangular matrix A, packed sequentially,
 *        column by column; that is, if i >= j, then A[i,j] is stored in
 *        AP[i+((2*n-j+1)*j)/2]. When diag = 'U' or 'u', the diagonal elements
 *        of A are not referenced and are assumed to be unity.
 * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the n-element right-hand side vector b. On exit,
 *        it is overwritten with the solution vector x.
 * incx   storage spacing between elements of x. It must not be zero.
 *
 * Output
 * ------
 * x      updated to contain the solution vector x that solves op(A) * x = b.
 *
 * Reference: http://www.netlib.org/blas/stpsv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0, n < 0, or n > 4070
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
* </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasStpsvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject AP, jobject x, jint incx)
{
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasStpsv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasStpsv");
        return;
    }
    float* nativeAP;
    float* nativeX;

    nativeAP = (float*)getPointer(env, AP);
    nativeX = (float*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasStpsv(%c, %c, %c, %d, '%s', '%s', %d)\n",
        uplo, trans, diag, n, "AP", "x", incx);

    cublasStpsv((char)uplo, (char)trans, (char)diag, n, nativeAP, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasStrmv (char uplo, char trans, char diag, int n, const float *A,
 *              int lda, float *x, int incx);
 *
 * performs one of the matrix-vector operations x = op(A) * x, where op(A) =
 = A, or op(A) = transpose(A). x is an n-element single precision vector, and
 * A is an n x n, unit or non-unit, upper or lower, triangular matrix composed
 * of single precision elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix. If uplo = 'U' or 'u', then A is an upper triangular matrix.
 *        If uplo = 'L' or 'l', then A is a lower triangular matrix.
 * trans  specifies op(A). If transa = 'N' or 'n', op(A) = A. If trans = 'T',
 *        't', 'C', or 'c', op(A) = transpose(A)
 * diag   specifies whether or not matrix A is unit triangular. If diag = 'U'
 *        or 'u', A is assumed to be unit triangular. If diag = 'N' or 'n', A
 *        is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * A      single precision array of dimension (lda, n). If uplo = 'U' or 'u',
 *        the leading n x n upper triangular part of the array A must contain
 *        the upper triangular matrix and the strictly lower triangular part
 *        of A is not referenced. If uplo = 'L' or 'l', the leading n x n lower
 *        triangular part of the array A must contain the lower triangular
 *        matrix and the strictly upper triangular part of A is not referenced.
 *        When diag = 'U' or 'u', the diagonal elements of A are not referenced
 *        either, but are are assumed to be unity.
 * lda    is the leading dimension of A. It must be at least max (1, n).
 * x      single precision array of length at least (1 + (n - 1) * abs(incx) ).
 *        On entry, x contains the source vector. On exit, x is overwritten
 *        with the result vector.
 * incx   specifies the storage spacing for elements of x. incx must not be
 *        zero.
 *
 * Output
 * ------
 * x      updated according to x = op(A) * x,
 *
 * Reference: http://www.netlib.org/blas/strmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasStrmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasStrmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasStrmv");
        return;
    }
    float* nativeA;
    float* nativeX;

    nativeA = (float*)getPointer(env, A);
    nativeX = (float*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasStrmv(%c, %c, %c, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, "A", lda, "x", incx);

    cublasStrmv((char)uplo, (char)trans, (char)diag, n, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasStrsv (char uplo, char trans, char diag, int n, const float *A,
 *              int lda, float *x, int incx)
 *
 * solves a system of equations op(A) * x = b, where op(A) is either A or
 * transpose(A). b and x are single precision vectors consisting of n
 * elements, and A is an n x n matrix composed of a unit or non-unit, upper
 * or lower triangular matrix. Matrix A is stored in column major format,
 * and lda is the leading dimension of the two-dimensional array containing
 * A.
 *
 * No test for singularity or near-singularity is included in this function.
 * Such tests must be performed before calling this function.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the
 *        lower triangular part of array A. If uplo = 'U' or 'u', then only
 *        the upper triangular part of A may be referenced. If uplo = 'L' or
 *        'l', then only the lower triangular part of A may be referenced.
 * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If transa = 't',
 *        'T', 'c', or 'C', op(A) = transpose(A)
 * diag   specifies whether or not A is a unit triangular matrix like so:
 *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
 *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. It
 *        must be at least 0.
 * A      is a single precision array of dimensions (lda, n). If uplo = 'U'
 *        or 'u', then A must contains the upper triangular part of a symmetric
 *        matrix, and the strictly lower triangular parts is not referenced.
 *        If uplo = 'L' or 'l', then A contains the lower triangular part of
 *        a symmetric matrix, and the strictly upper triangular part is not
 *        referenced.
 * lda    is the leading dimension of the two-dimensional array containing A.
 *        lda must be at least max(1, n).
 * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the n element right-hand side vector b. On exit,
 *        it is overwritten with the solution vector x.
 * incx   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 *
 * Output
 * ------
 * x      updated to contain the solution vector x that solves op(A) * x = b.
 *
 * Reference: http://www.netlib.org/blas/strsv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasStrsvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasStrsv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasStrsv");
        return;
    }
    float* nativeA;
    float* nativeX;

    nativeA = (float*)getPointer(env, A);
    nativeX = (float*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasStrsv(%c, %c, %c, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, "A", lda, "x", incx);

    cublasStrsv((char)uplo, (char)trans, (char)diag, n, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasZtrmv (char uplo, char trans, char diag, int n, const cuDoubleComplex *A,
 *              int lda, cuDoubleComplex *x, int incx);
 *
 * performs one of the matrix-vector operations x = op(A) * x,
 * where op(A) = A, or op(A) = transpose(A) or op(A) = conjugate(transpose(A)).
 * x is an n-element double precision complex vector, and
 * A is an n x n, unit or non-unit, upper or lower, triangular matrix composed
 * of double precision complex elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix. If uplo = 'U' or 'u', then A is an upper triangular matrix.
 *        If uplo = 'L' or 'l', then A is a lower triangular matrix.
 * trans  specifies op(A). If trans = 'n' or 'N', op(A) = A. If trans = 't' or
 *        'T', op(A) = transpose(A).  If trans = 'c' or 'C', op(A) =
 *        conjugate(transpose(A)).
 * diag   specifies whether or not matrix A is unit triangular. If diag = 'U'
 *        or 'u', A is assumed to be unit triangular. If diag = 'N' or 'n', A
 *        is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * A      double precision array of dimension (lda, n). If uplo = 'U' or 'u',
 *        the leading n x n upper triangular part of the array A must contain
 *        the upper triangular matrix and the strictly lower triangular part
 *        of A is not referenced. If uplo = 'L' or 'l', the leading n x n lower
 *        triangular part of the array A must contain the lower triangular
 *        matrix and the strictly upper triangular part of A is not referenced.
 *        When diag = 'U' or 'u', the diagonal elements of A are not referenced
 *        either, but are are assumed to be unity.
 * lda    is the leading dimension of A. It must be at least max (1, n).
 * x      double precision array of length at least (1 + (n - 1) * abs(incx) ).
 *        On entry, x contains the source vector. On exit, x is overwritten
 *        with the result vector.
 * incx   specifies the storage spacing for elements of x. incx must not be
 *        zero.
 *
 * Output
 * ------
 * x      updated according to x = op(A) * x,
 *
 * Reference: http://www.netlib.org/blas/ztrmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZtrmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZtrmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZtrmv");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeX;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeX = (cuDoubleComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasZtrmv(%c, %c, %c, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, "A", lda, "x", incx);

    cublasZtrmv((char)uplo, (char)trans, (char)diag, n, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasZgbmv (char trans, int m, int n, int kl, int ku, cuDoubleComplex alpha,
 *              const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex beta,
 *              cuDoubleComplex *y, int incy);
 *
 * performs one of the matrix-vector operations
 *
 *    y = alpha*op(A)*x + beta*y,  op(A)=A or op(A) = transpose(A)
 *
 * alpha and beta are double precision complex scalars. x and y are double precision
 * complex vectors. A is an m by n band matrix consisting of double precision complex elements
 * with kl sub-diagonals and ku super-diagonals.
 *
 * Input
 * -----
 * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
 *        or 't', op(A) = transpose(A). If trans == 'C' or 'c',
 *        op(A) = conjugate(transpose(A)).
 * m      specifies the number of rows of the matrix A. m must be at least
 *        zero.
 * n      specifies the number of columns of the matrix A. n must be at least
 *        zero.
 * kl     specifies the number of sub-diagonals of matrix A. It must be at
 *        least zero.
 * ku     specifies the number of super-diagonals of matrix A. It must be at
 *        least zero.
 * alpha  double precision complex scalar multiplier applied to op(A).
 * A      double precision complex array of dimensions (lda, n). The leading
 *        (kl + ku + 1) x n part of the array A must contain the band matrix A,
 *        supplied column by column, with the leading diagonal of the matrix
 *        in row (ku + 1) of the array, the first super-diagonal starting at
 *        position 2 in row ku, the first sub-diagonal starting at position 1
 *        in row (ku + 2), and so on. Elements in the array A that do not
 *        correspond to elements in the band matrix (such as the top left
 *        ku x ku triangle) are not referenced.
 * lda    leading dimension of A. lda must be at least (kl + ku + 1).
 * x      double precision complex array of length at least (1+(n-1)*abs(incx)) when
 *        trans == 'N' or 'n' and at least (1+(m-1)*abs(incx)) otherwise.
 * incx   specifies the increment for the elements of x. incx must not be zero.
 * beta   double precision complex scalar multiplier applied to vector y. If beta is
 *        zero, y is not read.
 * y      double precision complex array of length at least (1+(m-1)*abs(incy)) when
 *        trans == 'N' or 'n' and at least (1+(n-1)*abs(incy)) otherwise. If
 *        beta is zero, y is not read.
 * incy   On entry, incy specifies the increment for the elements of y. incy
 *        must not be zero.
 *
 * Output
 * ------
 * y      updated according to y = alpha*op(A)*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/zgbmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZgbmvNative
    (JNIEnv *env, jclass cls, jchar trans, jint m, jint n, jint kl, jint ku, jobject alpha, jobject A, jint lda, jobject x, jint incx, jobject beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZgbmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZgbmv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZgbmv");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;
    cuDoubleComplex dobuleComplexAlpha;
    cuDoubleComplex dobuleComplexBeta;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    dobuleComplexBeta.x = env->GetDoubleField(beta, cuDoubleComplex_x);
    dobuleComplexBeta.y = env->GetDoubleField(beta, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZgbmv(%c, %d, %d, %d, %d, [%lf,%lf], '%s', %d, '%s', %d, [%lf,%lf], '%s', %d)\n",
        trans, m, n, kl, ku, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "A", lda, "x", incx, dobuleComplexBeta.x, dobuleComplexBeta.y, "y", incy);

    cublasZgbmv((char)trans, m, n, kl, ku, dobuleComplexAlpha, nativeA, lda, nativeX, incx, dobuleComplexBeta, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasZtbmv (char uplo, char trans, char diag, int n, int k, const cuDoubleComplex *A,
 *              int lda, cuDoubleComplex *x, int incx)
 *
 * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
 * op(A) = transpose(A) or op(A) = conjugate(transpose(A)). x is an n-element
 * double precision complex vector, and A is an n x n, unit or non-unit, upper
 * or lower triangular band matrix composed of double precision complex elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix A is an upper or lower triangular band
 *        matrix. If uplo == 'U' or 'u', A is an upper triangular band matrix.
 *        If uplo == 'L' or 'l', A is a lower triangular band matrix.
 * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
 *        or 't', op(A) = transpose(A). If trans == 'C' or 'c',
 *        op(A) = conjugate(transpose(A)).
 * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
 *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
 *        is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
 *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
 *        'l', k specifies the number of sub-diagonals. k must at least be
 *        zero.
 * A      double precision complex array of dimension (lda, n). If uplo == 'U' or 'u',
 *        the leading (k + 1) x n part of the array A must contain the upper
 *        triangular band matrix, supplied column by column, with the leading
 *        diagonal of the matrix in row (k + 1) of the array, the first
 *        super-diagonal starting at position 2 in row k, and so on. The top
 *        left k x k triangle of the array A is not referenced. If uplo == 'L'
 *        or 'l', the leading (k + 1) x n part of the array A must constain the
 *        lower triangular band matrix, supplied column by column, with the
 *        leading diagonal of the matrix in row 1 of the array, the first
 *        sub-diagonal startingat position 1 in row 2, and so on. The bottom
 *        right k x k triangle of the array is not referenced.
 * lda    is the leading dimension of A. It must be at least (k + 1).
 * x      double precision complex array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the source vector. On exit, x is overwritten
 *        with the result vector.
 * incx   specifies the storage spacing for elements of x. incx must not be
 *        zero.
 *
 * Output
 * ------
 * x      updated according to x = op(A) * x
 *
 * Reference: http://www.netlib.org/blas/ztbmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n or k < 0, or if incx == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZtbmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jint k, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZtbmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZtbmv");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeX;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeX = (cuDoubleComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasZtbmv(%c, %c, %c, %d, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, k, "A", lda, "x", incx);

    cublasZtbmv((char)uplo, (char)trans, (char)diag, n, k, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void cublasZtbsv (char uplo, char trans, char diag, int n, int k,
 *                   const cuDoubleComplex *A, int lda, cuDoubleComplex *X, int incx)
 *
 * solves one of the systems of equations op(A)*x = b, where op(A) is either
 * op(A) = A , op(A) = transpose(A) or op(A) = conjugate(transpose(A)).
 * b and x are n element vectors, and A is an n x n unit or non-unit,
 * upper or lower triangular band matrix with k + 1 diagonals. No test
 * for singularity or near-singularity is included in this function.
 * Such tests must be performed before calling this function.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix is an upper or lower triangular band
 *        matrix as follows: If uplo == 'U' or 'u', A is an upper triangular
 *        band matrix. If uplo == 'L' or 'l', A is a lower triangular band
 *        matrix.
 * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
 *        't', op(A) = transpose(A). If trans == 'C' or 'c',
 *        op(A) = conjugate(transpose(A)).
 * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
 *        assumed to be unit triangular; thas is, diagonal elements are not
 *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
 *        assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
 *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
 *        'l', k specifies the number of sub-diagonals. k must at least be
 *        zero.
 * A      double precision complex array of dimension (lda, n). If uplo == 'U' or 'u',
 *        the leading (k + 1) x n part of the array A must contain the upper
 *        triangular band matrix, supplied column by column, with the leading
 *        diagonal of the matrix in row (k + 1) of the array, the first super-
 *        diagonal starting at position 2 in row k, and so on. The top left
 *        k x k triangle of the array A is not referenced. If uplo == 'L' or
 *        'l', the leading (k + 1) x n part of the array A must constain the
 *        lower triangular band matrix, supplied column by column, with the
 *        leading diagonal of the matrix in row 1 of the array, the first
 *        sub-diagonal starting at position 1 in row 2, and so on. The bottom
 *        right k x k triangle of the array is not referenced.
 * x      double precision complex array of length at least (1+(n-1)*abs(incx)).
 * incx   storage spacing between elements of x. It must not be zero.
 *
 * Output
 * ------
 * x      updated to contain the solution vector x that solves op(A) * x = b.
 *
 * Reference: http://www.netlib.org/blas/ztbsv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0, n < 0 or n > 1016
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZtbsvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jint k, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZtbsv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZtbsv");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeX;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeX = (cuDoubleComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasZtbsv(%c, %c, %c, %d, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, k, "A", lda, "x", incx);

    cublasZtbsv((char)uplo, (char)trans, (char)diag, n, k, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasZhemv (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
 *              const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy)
 *
 * performs the matrix-vector operation
 *
 *     y = alpha*A*x + beta*y
 *
 * Alpha and beta are double precision complex scalars, and x and y are double
 * precision complex vectors, each with n elements. A is a hermitian n x n matrix
 * consisting of double precision complex elements that is stored in either upper or
 * lower storage mode.
 *
 * Input
 * -----
 * uplo   specifies whether the upper or lower triangular part of the array A
 *        is to be referenced. If uplo == 'U' or 'u', the hermitian matrix A
 *        is stored in upper storage mode, i.e. only the upper triangular part
 *        of A is to be referenced while the lower triangular part of A is to
 *        be inferred. If uplo == 'L' or 'l', the hermitian matrix A is stored
 *        in lower storage mode, i.e. only the lower triangular part of A is
 *        to be referenced while the upper triangular part of A is to be
 *        inferred.
 * n      specifies the number of rows and the number of columns of the
 *        hermitian matrix A. n must be at least zero.
 * alpha  double precision complex scalar multiplier applied to A*x.
 * A      double precision complex array of dimensions (lda, n). If uplo == 'U' or 'u',
 *        the leading n x n upper triangular part of the array A must contain
 *        the upper triangular part of the hermitian matrix and the strictly
 *        lower triangular part of A is not referenced. If uplo == 'L' or 'l',
 *        the leading n x n lower triangular part of the array A must contain
 *        the lower triangular part of the hermitian matrix and the strictly
 *        upper triangular part of A is not referenced. The imaginary parts
 *        of the diagonal elements need not be set, they are assumed to be zero.
 * lda    leading dimension of A. It must be at least max (1, n).
 * x      double precision complex array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * beta   double precision complex scalar multiplier applied to vector y.
 * y      double precision complex array of length at least (1 + (n - 1) * abs(incy)).
 *        If beta is zero, y is not read.
 * incy   storage spacing between elements of y. incy must not be zero.
 *
 * Output
 * ------
 * y      updated according to y = alpha*A*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/zhemv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZhemvNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jobject alpha, jobject A, jint lda, jobject x, jint incx, jobject beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZhemv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZhemv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZhemv");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;
    cuDoubleComplex dobuleComplexAlpha;
    cuDoubleComplex dobuleComplexBeta;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    dobuleComplexBeta.x = env->GetDoubleField(beta, cuDoubleComplex_x);
    dobuleComplexBeta.y = env->GetDoubleField(beta, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZhemv(%c, %d, [%lf,%lf], '%s', %d, '%s', %d, [%lf,%lf], '%s', %d)\n",
        uplo, n, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "A", lda, "x", incx, dobuleComplexBeta.x, dobuleComplexBeta.y, "y", incy);

    cublasZhemv((char)uplo, n, dobuleComplexAlpha, nativeA, lda, nativeX, incx, dobuleComplexBeta, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasZhpmv (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *AP, const cuDoubleComplex *x,
 *              int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy)
 *
 * performs the matrix-vector operation
 *
 *    y = alpha * A * x + beta * y
 *
 * Alpha and beta are double precision complex scalars, and x and y are double
 * precision complex vectors with n elements. A is an hermitian n x n matrix
 * consisting of double precision complex elements that is supplied in packed form.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array AP. If uplo == 'U' or 'u', then the upper
 *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then
 *        the lower triangular part of A is supplied in AP.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  double precision complex scalar multiplier applied to A*x.
 * AP     double precision complex array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part
 *        of the hermitian matrix A, packed sequentially, column by column;
 *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part
 *        of the hermitian matrix A, packed sequentially, column by column;
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 *        The imaginary parts of the diagonal elements need not be set, they
 *        are assumed to be zero.
 * x      double precision complex array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * beta   double precision complex scalar multiplier applied to vector y;
 * y      double precision array of length at least (1 + (n - 1) * abs(incy)).
 *        If beta is zero, y is not read.
 * incy   storage spacing between elements of y. incy must not be zero.
 *
 * Output
 * ------
 * y      updated according to y = alpha*A*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/zhpmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZhpmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jobject alpha, jobject AP, jobject x, jint incx, jobject beta, jobject y, jint incy)
{
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasZhpmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZhpmv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZhpmv");
        return;
    }
    cuDoubleComplex* nativeAP;
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;
    cuDoubleComplex dobuleComplexAlpha;
    cuDoubleComplex dobuleComplexBeta;

    nativeAP = (cuDoubleComplex*)getPointer(env, AP);
    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    dobuleComplexBeta.x = env->GetDoubleField(beta, cuDoubleComplex_x);
    dobuleComplexBeta.y = env->GetDoubleField(beta, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZhpmv(%c, %d, [%lf,%lf], '%s', '%s', %d, [%lf,%lf], '%s', %d)\n",
        uplo, n, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "AP", "x", incx, dobuleComplexBeta.x, dobuleComplexBeta.y, "y", incy);

    cublasZhpmv((char)uplo, n, dobuleComplexAlpha, nativeAP, nativeX, incx, dobuleComplexBeta, nativeY, incy);
}




/**
 * <pre>
 * cublasZgemv (char trans, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
 *              const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy)
 *
 * performs one of the matrix-vector operations
 *
 *    y = alpha * op(A) * x + beta * y,
 *
 * where op(A) is one of
 *
 *    op(A) = A   or   op(A) = transpose(A)
 *
 * where alpha and beta are double precision scalars, x and y are double
 * precision vectors, and A is an m x n matrix consisting of double precision
 * elements. Matrix A is stored in column major format, and lda is the leading
 * dimension of the two-dimensional array in which A is stored.
 *
 * Input
 * -----
 * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If trans =
 *        trans = 't', 'T', 'c', or 'C', op(A) = transpose(A)
 * m      specifies the number of rows of the matrix A. m must be at least
 *        zero.
 * n      specifies the number of columns of the matrix A. n must be at least
 *        zero.
 * alpha  double precision scalar multiplier applied to op(A).
 * A      double precision array of dimensions (lda, n) if trans = 'n' or
 *        'N'), and of dimensions (lda, m) otherwise. lda must be at least
 *        max(1, m) and at least max(1, n) otherwise.
 * lda    leading dimension of two-dimensional array used to store matrix A
 * x      double precision array of length at least (1 + (n - 1) * abs(incx))
 *        when trans = 'N' or 'n' and at least (1 + (m - 1) * abs(incx))
 *        otherwise.
 * incx   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 * beta   double precision scalar multiplier applied to vector y. If beta
 *        is zero, y is not read.
 * y      double precision array of length at least (1 + (m - 1) * abs(incy))
 *        when trans = 'N' or 'n' and at least (1 + (n - 1) * abs(incy))
 *        otherwise.
 * incy   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 *
 * Output
 * ------
 * y      updated according to alpha * op(A) * x + beta * y
 *
 * Reference: http://www.netlib.org/blas/zgemv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0, or if incx or incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZgemvNative
    (JNIEnv *env, jclass cls, jchar trans, jint m, jint n, jobject alpha, jobject A, jint lda, jobject x, jint incx, jobject beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZgemv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZgemv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZgemv");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;
    cuDoubleComplex dobuleComplexAlpha;
    cuDoubleComplex dobuleComplexBeta;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    dobuleComplexBeta.x = env->GetDoubleField(beta, cuDoubleComplex_x);
    dobuleComplexBeta.y = env->GetDoubleField(beta, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZgemv(%c, %d, %d, [%lf,%lf], '%s', %d, '%s', %d, [%lf,%lf], '%s', %d)\n",
        trans, m, n, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "A", lda, "x", incx, dobuleComplexBeta.x, dobuleComplexBeta.y, "y", incy);

    cublasZgemv((char)trans, m, n, dobuleComplexAlpha, nativeA, lda, nativeX, incx, dobuleComplexBeta, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasZtpmv (char uplo, char trans, char diag, int n, const cuDoubleComplex *AP,
 *              cuDoubleComplex *x, int incx);
 *
 * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
 * op(A) = transpose(A) or op(A) = conjugate(transpose(A)) . x is an n element
 * double precision complex vector, and A is an n x n, unit or non-unit, upper
 * or lower triangular matrix composed of double precision complex elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix. If uplo == 'U' or 'u', then A is an upper triangular matrix.
 *        If uplo == 'L' or 'l', then A is a lower triangular matrix.
 * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
 *        or 't', op(A) = transpose(A). If trans == 'C' or 'c',
 *        op(A) = conjugate(transpose(A)).
 *
 * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
 *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
 *        is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero. In the current implementation n must not exceed 4070.
 * AP     double precision complex array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i <= j, then A[i,j] is stored in AP[i+(j*(j+1)/2)]. If
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 * x      double precision complex array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the source vector. On exit, x is overwritten
 *        with the result vector.
 * incx   specifies the storage spacing for elements of x. incx must not be
 *        zero.
 *
 * Output
 * ------
 * x      updated according to x = op(A) * x,
 *
 * Reference: http://www.netlib.org/blas/ztpmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or n < 0
 * CUBLAS_STATUS_ALLOC_FAILED     if function cannot allocate enough internal scratch vector memory
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZtpmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject AP, jobject x, jint incx)
{
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasZtpmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZtpmv");
        return;
    }
    cuDoubleComplex* nativeAP;
    cuDoubleComplex* nativeX;

    nativeAP = (cuDoubleComplex*)getPointer(env, AP);
    nativeX = (cuDoubleComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasZtpmv(%c, %c, %c, %d, '%s', '%s', %d)\n",
        uplo, trans, diag, n, "AP", "x", incx);

    cublasZtpmv((char)uplo, (char)trans, (char)diag, n, nativeAP, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasZtpsv (char uplo, char trans, char diag, int n, const cuDoubleComplex *AP,
 *              cuDoubleComplex *X, int incx)
 *
 * solves one of the systems of equations op(A)*x = b, where op(A) is either
 * op(A) = A , op(A) = transpose(A) or op(A) = conjugate(transpose)). b and
 * x are n element complex vectors, and A is an n x n unit or non-unit,
 * upper or lower triangular matrix. No test for singularity or near-singularity
 * is included in this routine. Such tests must be performed before calling this routine.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix is an upper or lower triangular matrix
 *        as follows: If uplo == 'U' or 'u', A is an upper triangluar matrix.
 *        If uplo == 'L' or 'l', A is a lower triangular matrix.
 * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T'
 *        or 't', op(A) = transpose(A). If trans == 'C' or 'c', op(A) =
 *        conjugate(transpose(A)).
 * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
 *        assumed to be unit triangular; thas is, diagonal elements are not
 *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
 *        assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * AP     double precision complex array with at least ((n*(n+1))/2) elements.
 *        If uplo == 'U' or 'u', the array AP contains the upper triangular
 *        matrix A, packed sequentially, column by column; that is, if i <= j, then
 *        A[i,j] is stored is AP[i+(j*(j+1)/2)]. If uplo == 'L' or 'L', the
 *        array AP contains the lower triangular matrix A, packed sequentially,
 *        column by column; that is, if i >= j, then A[i,j] is stored in
 *        AP[i+((2*n-j+1)*j)/2]. When diag = 'U' or 'u', the diagonal elements
 *        of A are not referenced and are assumed to be unity.
 * x      double precision complex array of length at least (1+(n-1)*abs(incx)).
 * incx   storage spacing between elements of x. It must not be zero.
 *
 * Output
 * ------
 * x      updated to contain the solution vector x that solves op(A) * x = b.
 *
 * Reference: http://www.netlib.org/blas/ztpsv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0 or n > 2035
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZtpsvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject AP, jobject x, jint incx)
{
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasZtpsv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZtpsv");
        return;
    }
    cuDoubleComplex* nativeAP;
    cuDoubleComplex* nativeX;

    nativeAP = (cuDoubleComplex*)getPointer(env, AP);
    nativeX = (cuDoubleComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasZtpsv(%c, %c, %c, %d, '%s', '%s', %d)\n",
        uplo, trans, diag, n, "AP", "x", incx);

    cublasZtpsv((char)uplo, (char)trans, (char)diag, n, nativeAP, nativeX, incx);
}




/**
 * <pre>
 * cublasCgemv (char trans, int m, int n, cuComplex alpha, const cuComplex *A,
 *              int lda, const cuComplex *x, int incx, cuComplex beta, cuComplex *y,
 *              int incy)
 *
 * performs one of the matrix-vector operations
 *
 *    y = alpha * op(A) * x + beta * y,
 *
 * where op(A) is one of
 *
 *    op(A) = A   or   op(A) = transpose(A) or op(A) = conjugate(transpose(A))
 *
 * where alpha and beta are single precision scalars, x and y are single
 * precision vectors, and A is an m x n matrix consisting of single precision
 * elements. Matrix A is stored in column major format, and lda is the leading
 * dimension of the two-dimensional array in which A is stored.
 *
 * Input
 * -----
 * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If trans =
 *        trans = 't' or 'T', op(A) = transpose(A). If trans = 'c' or 'C',
 *        op(A) = conjugate(transpose(A))
 * m      specifies the number of rows of the matrix A. m must be at least
 *        zero.
 * n      specifies the number of columns of the matrix A. n must be at least
 *        zero.
 * alpha  single precision scalar multiplier applied to op(A).
 * A      single precision array of dimensions (lda, n) if trans = 'n' or
 *        'N'), and of dimensions (lda, m) otherwise. lda must be at least
 *        max(1, m) and at least max(1, n) otherwise.
 * lda    leading dimension of two-dimensional array used to store matrix A
 * x      single precision array of length at least (1 + (n - 1) * abs(incx))
 *        when trans = 'N' or 'n' and at least (1 + (m - 1) * abs(incx))
 *        otherwise.
 * incx   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 * beta   single precision scalar multiplier applied to vector y. If beta
 *        is zero, y is not read.
 * y      single precision array of length at least (1 + (m - 1) * abs(incy))
 *        when trans = 'N' or 'n' and at least (1 + (n - 1) * abs(incy))
 *        otherwise.
 * incy   specifies the storage spacing between elements of y. incy must not
 *        be zero.
 *
 * Output
 * ------
 * y      updated according to alpha * op(A) * x + beta * y
 *
 * Reference: http://www.netlib.org/blas/cgemv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0, or if incx or incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCgemvNative
    (JNIEnv *env, jclass cls, jchar trans, jint m, jint n, jobject alpha, jobject A, jint lda, jobject x, jint incx, jobject beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCgemv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCgemv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasCgemv");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeX;
    cuComplex* nativeY;
    cuComplex complexAlpha;
    cuComplex complexBeta;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeX = (cuComplex*)getPointer(env, x);
    nativeY = (cuComplex*)getPointer(env, y);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    complexBeta.x = env->GetFloatField(beta, cuComplex_x);
    complexBeta.y = env->GetFloatField(beta, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCgemv(%c, %d, %d, [%f,%f], '%s', %d, '%s', %d, [%f,%f], '%s', %d)\n",
        trans, m, n, complexAlpha.x, complexAlpha.y, "A", lda, "x", incx, complexBeta.x, complexBeta.y, "y", incy);

    cublasCgemv((char)trans, m, n, complexAlpha, nativeA, lda, nativeX, incx, complexBeta, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasCgbmv (char trans, int m, int n, int kl, int ku, cuComplex alpha,
 *              const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex beta,
 *              cuComplex *y, int incy);
 *
 * performs one of the matrix-vector operations
 *
 *    y = alpha*op(A)*x + beta*y,  op(A)=A or op(A) = transpose(A)
 *
 * alpha and beta are single precision complex scalars. x and y are single precision
 * complex vectors. A is an m by n band matrix consisting of single precision complex elements
 * with kl sub-diagonals and ku super-diagonals.
 *
 * Input
 * -----
 * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
 *        or 't', op(A) = transpose(A). If trans == 'C' or 'c',
 *        op(A) = conjugate(transpose(A)).
 * m      specifies the number of rows of the matrix A. m must be at least
 *        zero.
 * n      specifies the number of columns of the matrix A. n must be at least
 *        zero.
 * kl     specifies the number of sub-diagonals of matrix A. It must be at
 *        least zero.
 * ku     specifies the number of super-diagonals of matrix A. It must be at
 *        least zero.
 * alpha  single precision complex scalar multiplier applied to op(A).
 * A      single precision complex array of dimensions (lda, n). The leading
 *        (kl + ku + 1) x n part of the array A must contain the band matrix A,
 *        supplied column by column, with the leading diagonal of the matrix
 *        in row (ku + 1) of the array, the first super-diagonal starting at
 *        position 2 in row ku, the first sub-diagonal starting at position 1
 *        in row (ku + 2), and so on. Elements in the array A that do not
 *        correspond to elements in the band matrix (such as the top left
 *        ku x ku triangle) are not referenced.
 * lda    leading dimension of A. lda must be at least (kl + ku + 1).
 * x      single precision complex array of length at least (1+(n-1)*abs(incx)) when
 *        trans == 'N' or 'n' and at least (1+(m-1)*abs(incx)) otherwise.
 * incx   specifies the increment for the elements of x. incx must not be zero.
 * beta   single precision complex scalar multiplier applied to vector y. If beta is
 *        zero, y is not read.
 * y      single precision complex array of length at least (1+(m-1)*abs(incy)) when
 *        trans == 'N' or 'n' and at least (1+(n-1)*abs(incy)) otherwise. If
 *        beta is zero, y is not read.
 * incy   On entry, incy specifies the increment for the elements of y. incy
 *        must not be zero.
 *
 * Output
 * ------
 * y      updated according to y = alpha*op(A)*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/cgbmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCgbmvNative
    (JNIEnv *env, jclass cls, jchar trans, jint m, jint n, jint kl, jint ku, jobject alpha, jobject A, jint lda, jobject x, jint incx, jobject beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCgbmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCgbmv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasCgbmv");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeX;
    cuComplex* nativeY;
    cuComplex complexAlpha;
    cuComplex complexBeta;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeX = (cuComplex*)getPointer(env, x);
    nativeY = (cuComplex*)getPointer(env, y);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    complexBeta.x = env->GetFloatField(beta, cuComplex_x);
    complexBeta.y = env->GetFloatField(beta, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCgbmv(%c, %d, %d, %d, %d, [%f,%f], '%s', %d, '%s', %d, [%f,%f], '%s', %d)\n",
        trans, m, n, kl, ku, complexAlpha.x, complexAlpha.y, "A", lda, "x", incx, complexBeta.x, complexBeta.y, "y", incy);

    cublasCgbmv((char)trans, m, n, kl, ku, complexAlpha, nativeA, lda, nativeX, incx, complexBeta, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasChemv (char uplo, int n, cuComplex alpha, const cuComplex *A, int lda,
 *              const cuComplex *x, int incx, cuComplex beta, cuComplex *y, int incy)
 *
 * performs the matrix-vector operation
 *
 *     y = alpha*A*x + beta*y
 *
 * Alpha and beta are single precision complex scalars, and x and y are single
 * precision complex vectors, each with n elements. A is a hermitian n x n matrix
 * consisting of single precision complex elements that is stored in either upper or
 * lower storage mode.
 *
 * Input
 * -----
 * uplo   specifies whether the upper or lower triangular part of the array A
 *        is to be referenced. If uplo == 'U' or 'u', the hermitian matrix A
 *        is stored in upper storage mode, i.e. only the upper triangular part
 *        of A is to be referenced while the lower triangular part of A is to
 *        be inferred. If uplo == 'L' or 'l', the hermitian matrix A is stored
 *        in lower storage mode, i.e. only the lower triangular part of A is
 *        to be referenced while the upper triangular part of A is to be
 *        inferred.
 * n      specifies the number of rows and the number of columns of the
 *        hermitian matrix A. n must be at least zero.
 * alpha  single precision complex scalar multiplier applied to A*x.
 * A      single precision complex array of dimensions (lda, n). If uplo == 'U' or 'u',
 *        the leading n x n upper triangular part of the array A must contain
 *        the upper triangular part of the hermitian matrix and the strictly
 *        lower triangular part of A is not referenced. If uplo == 'L' or 'l',
 *        the leading n x n lower triangular part of the array A must contain
 *        the lower triangular part of the hermitian matrix and the strictly
 *        upper triangular part of A is not referenced. The imaginary parts
 *        of the diagonal elements need not be set, they are assumed to be zero.
 * lda    leading dimension of A. It must be at least max (1, n).
 * x      single precision complex array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * beta   single precision complex scalar multiplier applied to vector y.
 * y      single precision complex array of length at least (1 + (n - 1) * abs(incy)).
 *        If beta is zero, y is not read.
 * incy   storage spacing between elements of y. incy must not be zero.
 *
 * Output
 * ------
 * y      updated according to y = alpha*A*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/chemv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasChemvNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jobject alpha, jobject A, jint lda, jobject x, jint incx, jobject beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasChemv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasChemv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasChemv");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeX;
    cuComplex* nativeY;
    cuComplex complexAlpha;
    cuComplex complexBeta;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeX = (cuComplex*)getPointer(env, x);
    nativeY = (cuComplex*)getPointer(env, y);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    complexBeta.x = env->GetFloatField(beta, cuComplex_x);
    complexBeta.y = env->GetFloatField(beta, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasChemv(%c, %d, [%f,%f], '%s', %d, '%s', %d, [%f,%f], '%s', %d)\n",
        uplo, n, complexAlpha.x, complexAlpha.y, "A", lda, "x", incx, complexBeta.x, complexBeta.y, "y", incy);

    cublasChemv((char)uplo, n, complexAlpha, nativeA, lda, nativeX, incx, complexBeta, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasChbmv (char uplo, int n, int k, cuComplex alpha, const cuComplex *A, int lda,
 *              const cuComplex *x, int incx, cuComplex beta, cuComplex *y, int incy)
 *
 * performs the matrix-vector operation
 *
 *     y := alpha*A*x + beta*y
 *
 * alpha and beta are single precision complex scalars. x and y are single precision
 * complex vectors with n elements. A is an n by n hermitian band matrix consisting
 * of single precision complex elements, with k super-diagonals and the same number
 * of subdiagonals.
 *
 * Input
 * -----
 * uplo   specifies whether the upper or lower triangular part of the hermitian
 *        band matrix A is being supplied. If uplo == 'U' or 'u', the upper
 *        triangular part is being supplied. If uplo == 'L' or 'l', the lower
 *        triangular part is being supplied.
 * n      specifies the number of rows and the number of columns of the
 *        hermitian matrix A. n must be at least zero.
 * k      specifies the number of super-diagonals of matrix A. Since the matrix
 *        is hermitian, this is also the number of sub-diagonals. k must be at
 *        least zero.
 * alpha  single precision complex scalar multiplier applied to A*x.
 * A      single precision complex array of dimensions (lda, n). When uplo == 'U' or
 *        'u', the leading (k + 1) x n part of array A must contain the upper
 *        triangular band of the hermitian matrix, supplied column by column,
 *        with the leading diagonal of the matrix in row (k+1) of the array,
 *        the first super-diagonal starting at position 2 in row k, and so on.
 *        The top left k x k triangle of the array A is not referenced. When
 *        uplo == 'L' or 'l', the leading (k + 1) x n part of the array A must
 *        contain the lower triangular band part of the hermitian matrix,
 *        supplied column by column, with the leading diagonal of the matrix in
 *        row 1 of the array, the first sub-diagonal starting at position 1 in
 *        row 2, and so on. The bottom right k x k triangle of the array A is
 *        not referenced. The imaginary parts of the diagonal elements need
 *        not be set, they are assumed to be zero.
 * lda    leading dimension of A. lda must be at least (k + 1).
 * x      single precision complex array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * beta   single precision complex scalar multiplier applied to vector y. If beta is
 *        zero, y is not read.
 * y      single precision complex array of length at least (1 + (n - 1) * abs(incy)).
 *        If beta is zero, y is not read.
 * incy   storage spacing between elements of y. incy must not be zero.
 *
 * Output
 * ------
 * y      updated according to alpha*A*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/chbmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if k or n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasChbmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jint k, jobject alpha, jobject A, jint lda, jobject x, jint incx, jobject beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasChbmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasChbmv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasChbmv");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeX;
    cuComplex* nativeY;
    cuComplex complexAlpha;
    cuComplex complexBeta;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeX = (cuComplex*)getPointer(env, x);
    nativeY = (cuComplex*)getPointer(env, y);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    complexBeta.x = env->GetFloatField(beta, cuComplex_x);
    complexBeta.y = env->GetFloatField(beta, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasChbmv(%c, %d, %d, [%f,%f], '%s', %d, '%s', %d, [%f,%f], '%s', %d)\n",
        uplo, n, k, complexAlpha.x, complexAlpha.y, "A", lda, "x", incx, complexBeta.x, complexBeta.y, "y", incy);

    cublasChbmv((char)uplo, n, k, complexAlpha, nativeA, lda, nativeX, incx, complexBeta, nativeY, incy);
}




/**
 * <pre>
 *
 * cublasCtrmv (char uplo, char trans, char diag, int n, const cuComplex *A,
 *              int lda, cuComplex *x, int incx);
 *
 * performs one of the matrix-vector operations x = op(A) * x,
 * where op(A) = A, or op(A) = transpose(A) or op(A) = conjugate(transpose(A)).
 * x is an n-element signle precision complex vector, and
 * A is an n x n, unit or non-unit, upper or lower, triangular matrix composed
 * of single precision complex elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix. If uplo = 'U' or 'u', then A is an upper triangular matrix.
 *        If uplo = 'L' or 'l', then A is a lower triangular matrix.
 * trans  specifies op(A). If trans = 'n' or 'N', op(A) = A. If trans = 't' or
 *        'T', op(A) = transpose(A).  If trans = 'c' or 'C', op(A) =
 *        conjugate(transpose(A)).
 * diag   specifies whether or not matrix A is unit triangular. If diag = 'U'
 *        or 'u', A is assumed to be unit triangular. If diag = 'N' or 'n', A
 *        is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * A      single precision array of dimension (lda, n). If uplo = 'U' or 'u',
 *        the leading n x n upper triangular part of the array A must contain
 *        the upper triangular matrix and the strictly lower triangular part
 *        of A is not referenced. If uplo = 'L' or 'l', the leading n x n lower
 *        triangular part of the array A must contain the lower triangular
 *        matrix and the strictly upper triangular part of A is not referenced.
 *        When diag = 'U' or 'u', the diagonal elements of A are not referenced
 *        either, but are are assumed to be unity.
 * lda    is the leading dimension of A. It must be at least max (1, n).
 * x      single precision array of length at least (1 + (n - 1) * abs(incx) ).
 *        On entry, x contains the source vector. On exit, x is overwritten
 *        with the result vector.
 * incx   specifies the storage spacing for elements of x. incx must not be
 *        zero.
 *
 * Output
 * ------
 * x      updated according to x = op(A) * x,
 *
 * Reference: http://www.netlib.org/blas/ctrmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCtrmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCtrmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCtrmv");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeX;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeX = (cuComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasCtrmv(%c, %c, %c, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, "A", lda, "x", incx);

    cublasCtrmv((char)uplo, (char)trans, (char)diag, n, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasCtbmv (char uplo, char trans, char diag, int n, int k, const cuComplex *A,
 *              int lda, cuComplex *x, int incx)
 *
 * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
 * op(A) = transpose(A) or op(A) = conjugate(transpose(A)). x is an n-element
 * single precision complex vector, and A is an n x n, unit or non-unit, upper
 * or lower triangular band matrix composed of single precision complex elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix A is an upper or lower triangular band
 *        matrix. If uplo == 'U' or 'u', A is an upper triangular band matrix.
 *        If uplo == 'L' or 'l', A is a lower triangular band matrix.
 * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
 *        or 't', op(A) = transpose(A). If trans == 'C' or 'c',
 *        op(A) = conjugate(transpose(A)).
 * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
 *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
 *        is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
 *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
 *        'l', k specifies the number of sub-diagonals. k must at least be
 *        zero.
 * A      single precision complex array of dimension (lda, n). If uplo == 'U' or 'u',
 *        the leading (k + 1) x n part of the array A must contain the upper
 *        triangular band matrix, supplied column by column, with the leading
 *        diagonal of the matrix in row (k + 1) of the array, the first
 *        super-diagonal starting at position 2 in row k, and so on. The top
 *        left k x k triangle of the array A is not referenced. If uplo == 'L'
 *        or 'l', the leading (k + 1) x n part of the array A must constain the
 *        lower triangular band matrix, supplied column by column, with the
 *        leading diagonal of the matrix in row 1 of the array, the first
 *        sub-diagonal startingat position 1 in row 2, and so on. The bottom
 *        right k x k triangle of the array is not referenced.
 * lda    is the leading dimension of A. It must be at least (k + 1).
 * x      single precision complex array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the source vector. On exit, x is overwritten
 *        with the result vector.
 * incx   specifies the storage spacing for elements of x. incx must not be
 *        zero.
 *
 * Output
 * ------
 * x      updated according to x = op(A) * x
 *
 * Reference: http://www.netlib.org/blas/ctbmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n or k < 0, or if incx == 0
 * CUBLAS_STATUS_ALLOC_FAILED     if function cannot allocate enough internal scratch vector memory
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCtbmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jint k, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCtbmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCtbmv");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeX;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeX = (cuComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasCtbmv(%c, %c, %c, %d, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, k, "A", lda, "x", incx);

    cublasCtbmv((char)uplo, (char)trans, (char)diag, n, k, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasCtpmv (char uplo, char trans, char diag, int n, const cuComplex *AP,
 *              cuComplex *x, int incx);
 *
 * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
 * op(A) = transpose(A) or op(A) = conjugate(transpose(A)) . x is an n element
 * single precision complex vector, and A is an n x n, unit or non-unit, upper
 * or lower triangular matrix composed of single precision complex elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix. If uplo == 'U' or 'u', then A is an upper triangular matrix.
 *        If uplo == 'L' or 'l', then A is a lower triangular matrix.
 * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
 *        or 't', op(A) = transpose(A). If trans == 'C' or 'c',
 *        op(A) = conjugate(transpose(A)).
 *
 * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
 *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
 *        is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero. In the current implementation n must not exceed 4070.
 * AP     single precision complex array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i <= j, then A[i,j] is stored in AP[i+(j*(j+1)/2)]. If
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 * x      single precision complex array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the source vector. On exit, x is overwritten
 *        with the result vector.
 * incx   specifies the storage spacing for elements of x. incx must not be
 *        zero.
 *
 * Output
 * ------
 * x      updated according to x = op(A) * x,
 *
 * Reference: http://www.netlib.org/blas/ctpmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or n < 0
 * CUBLAS_STATUS_ALLOC_FAILED     if function cannot allocate enough internal scratch vector memory
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCtpmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject AP, jobject x, jint incx)
{
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasCtpmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCtpmv");
        return;
    }
    cuComplex* nativeAP;
    cuComplex* nativeX;

    nativeAP = (cuComplex*)getPointer(env, AP);
    nativeX = (cuComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasCtpmv(%c, %c, %c, %d, '%s', '%s', %d)\n",
        uplo, trans, diag, n, "AP", "x", incx);

    cublasCtpmv((char)uplo, (char)trans, (char)diag, n, nativeAP, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasCtrsv (char uplo, char trans, char diag, int n, const cuComplex *A,
 *              int lda, cuComplex *x, int incx)
 *
 * solves a system of equations op(A) * x = b, where op(A) is either A,
 * transpose(A) or conjugate(transpose(A)). b and x are single precision
 * complex vectors consisting of n elements, and A is an n x n matrix
 * composed of a unit or non-unit, upper or lower triangular matrix.
 * Matrix A is stored in column major format, and lda is the leading
 * dimension of the two-dimensional array containing A.
 *
 * No test for singularity or near-singularity is included in this function.
 * Such tests must be performed before calling this function.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the
 *        lower triangular part of array A. If uplo = 'U' or 'u', then only
 *        the upper triangular part of A may be referenced. If uplo = 'L' or
 *        'l', then only the lower triangular part of A may be referenced.
 * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If transa = 't',
 *        'T', 'c', or 'C', op(A) = transpose(A)
 * diag   specifies whether or not A is a unit triangular matrix like so:
 *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
 *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. It
 *        must be at least 0.
 * A      is a single precision complex array of dimensions (lda, n). If uplo = 'U'
 *        or 'u', then A must contains the upper triangular part of a symmetric
 *        matrix, and the strictly lower triangular parts is not referenced.
 *        If uplo = 'L' or 'l', then A contains the lower triangular part of
 *        a symmetric matrix, and the strictly upper triangular part is not
 *        referenced.
 * lda    is the leading dimension of the two-dimensional array containing A.
 *        lda must be at least max(1, n).
 * x      single precision complex array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the n element right-hand side vector b. On exit,
 *        it is overwritten with the solution vector x.
 * incx   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 *
 * Output
 * ------
 * x      updated to contain the solution vector x that solves op(A) * x = b.
 *
 * Reference: http://www.netlib.org/blas/ctrsv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCtrsvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCtrsv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCtrsv");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeX;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeX = (cuComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasCtrsv(%c, %c, %c, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, "A", lda, "x", incx);

    cublasCtrsv((char)uplo, (char)trans, (char)diag, n, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void cublasCtbsv (char uplo, char trans, char diag, int n, int k,
 *                   const cuComplex *A, int lda, cuComplex *X, int incx)
 *
 * solves one of the systems of equations op(A)*x = b, where op(A) is either
 * op(A) = A , op(A) = transpose(A) or op(A) = conjugate(transpose(A)).
 * b and x are n element vectors, and A is an n x n unit or non-unit,
 * upper or lower triangular band matrix with k + 1 diagonals. No test
 * for singularity or near-singularity is included in this function.
 * Such tests must be performed before calling this function.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix is an upper or lower triangular band
 *        matrix as follows: If uplo == 'U' or 'u', A is an upper triangular
 *        band matrix. If uplo == 'L' or 'l', A is a lower triangular band
 *        matrix.
 * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
 *        't', op(A) = transpose(A). If trans == 'C' or 'c',
 *        op(A) = conjugate(transpose(A)).
 * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
 *        assumed to be unit triangular; thas is, diagonal elements are not
 *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
 *        assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
 *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
 *        'l', k specifies the number of sub-diagonals. k must at least be
 *        zero.
 * A      single precision complex array of dimension (lda, n). If uplo == 'U' or 'u',
 *        the leading (k + 1) x n part of the array A must contain the upper
 *        triangular band matrix, supplied column by column, with the leading
 *        diagonal of the matrix in row (k + 1) of the array, the first super-
 *        diagonal starting at position 2 in row k, and so on. The top left
 *        k x k triangle of the array A is not referenced. If uplo == 'L' or
 *        'l', the leading (k + 1) x n part of the array A must constain the
 *        lower triangular band matrix, supplied column by column, with the
 *        leading diagonal of the matrix in row 1 of the array, the first
 *        sub-diagonal starting at position 1 in row 2, and so on. The bottom
 *        right k x k triangle of the array is not referenced.
 * x      single precision complex array of length at least (1+(n-1)*abs(incx)).
 * incx   storage spacing between elements of x. It must not be zero.
 *
 * Output
 * ------
 * x      updated to contain the solution vector x that solves op(A) * x = b.
 *
 * Reference: http://www.netlib.org/blas/ctbsv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0, n < 0 or n > 2035
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCtbsvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jint k, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCtbsv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCtbsv");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeX;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeX = (cuComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasCtbsv(%c, %c, %c, %d, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, k, "A", lda, "x", incx);

    cublasCtbsv((char)uplo, (char)trans, (char)diag, n, k, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasCtpsv (char uplo, char trans, char diag, int n, const cuComplex *AP,
 *              cuComplex *X, int incx)
 *
 * solves one of the systems of equations op(A)*x = b, where op(A) is either
 * op(A) = A , op(A) = transpose(A) or op(A) = conjugate(transpose)). b and
 * x are n element complex vectors, and A is an n x n unit or non-unit,
 * upper or lower triangular matrix. No test for singularity or near-singularity
 * is included in this routine. Such tests must be performed before calling this routine.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix is an upper or lower triangular matrix
 *        as follows: If uplo == 'U' or 'u', A is an upper triangluar matrix.
 *        If uplo == 'L' or 'l', A is a lower triangular matrix.
 * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T'
 *        or 't', op(A) = transpose(A). If trans == 'C' or 'c', op(A) =
 *        conjugate(transpose(A)).
 * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
 *        assumed to be unit triangular; thas is, diagonal elements are not
 *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
 *        assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * AP     single precision complex array with at least ((n*(n+1))/2) elements.
 *        If uplo == 'U' or 'u', the array AP contains the upper triangular
 *        matrix A, packed sequentially, column by column; that is, if i <= j, then
 *        A[i,j] is stored is AP[i+(j*(j+1)/2)]. If uplo == 'L' or 'L', the
 *        array AP contains the lower triangular matrix A, packed sequentially,
 *        column by column; that is, if i >= j, then A[i,j] is stored in
 *        AP[i+((2*n-j+1)*j)/2]. When diag = 'U' or 'u', the diagonal elements
 *        of A are not referenced and are assumed to be unity.
 * x      single precision complex array of length at least (1+(n-1)*abs(incx)).
 * incx   storage spacing between elements of x. It must not be zero.
 *
 * Output
 * ------
 * x      updated to contain the solution vector x that solves op(A) * x = b.
 *
 * Reference: http://www.netlib.org/blas/ctpsv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0 or n > 2035
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCtpsvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject AP, jobject x, jint incx)
{
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasCtpsv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCtpsv");
        return;
    }
    cuComplex* nativeAP;
    cuComplex* nativeX;

    nativeAP = (cuComplex*)getPointer(env, AP);
    nativeX = (cuComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasCtpsv(%c, %c, %c, %d, '%s', '%s', %d)\n",
        uplo, trans, diag, n, "AP", "x", incx);

    cublasCtpsv((char)uplo, (char)trans, (char)diag, n, nativeAP, nativeX, incx);
}




/**
 * <pre>
 * cublasCgeru (int m, int n, cuComplex alpha, const cuComplex *x, int incx,
 *             const cuComplex *y, int incy, cuComplex *A, int lda)
 *
 * performs the symmetric rank 1 operation
 *
 *    A = alpha * x * transpose(y) + A,
 *
 * where alpha is a single precision complex scalar, x is an m element single
 * precision complex vector, y is an n element single precision complex vector, and A
 * is an m by n matrix consisting of single precision complex elements. Matrix A
 * is stored in column major format, and lda is the leading dimension of
 * the two-dimensional array used to store A.
 *
 * Input
 * -----
 * m      specifies the number of rows of the matrix A. It must be at least
 *        zero.
 * n      specifies the number of columns of the matrix A. It must be at
 *        least zero.
 * alpha  single precision complex scalar multiplier applied to x * transpose(y)
 * x      single precision complex array of length at least (1 + (m - 1) * abs(incx))
 * incx   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 * y      single precision complex array of length at least (1 + (n - 1) * abs(incy))
 * incy   specifies the storage spacing between elements of y. incy must not
 *        be zero.
 * A      single precision complex array of dimensions (lda, n).
 * lda    leading dimension of two-dimensional array used to store matrix A
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * transpose(y) + A
 *
 * Reference: http://www.netlib.org/blas/cgeru.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m <0, n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCgeruNative
    (JNIEnv *env, jclass cls, jint m, jint n, jobject alpha, jobject x, jint incx, jobject y, jint incy, jobject A, jint lda)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCgeru");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasCgeru");
        return;
    }
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCgeru");
        return;
    }
    cuComplex* nativeX;
    cuComplex* nativeY;
    cuComplex* nativeA;
    cuComplex complexAlpha;

    nativeX = (cuComplex*)getPointer(env, x);
    nativeY = (cuComplex*)getPointer(env, y);
    nativeA = (cuComplex*)getPointer(env, A);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCgeru(%d, %d, [%f,%f], '%s', %d, '%s', %d, '%s', %d)\n",
        m, n, complexAlpha.x, complexAlpha.y, "x", incx, "y", incy, "A", lda);

    cublasCgeru(m, n, complexAlpha, nativeX, incx, nativeY, incy, nativeA, lda);
}




/**
 * <pre>
 * cublasCgerc (int m, int n, cuComplex alpha, const cuComplex *x, int incx,
 *             const cuComplex *y, int incy, cuComplex *A, int lda)
 *
 * performs the symmetric rank 1 operation
 *
 *    A = alpha * x * conjugate(transpose(y)) + A,
 *
 * where alpha is a single precision complex scalar, x is an m element single
 * precision complex vector, y is an n element single precision complex vector, and A
 * is an m by n matrix consisting of single precision complex elements. Matrix A
 * is stored in column major format, and lda is the leading dimension of
 * the two-dimensional array used to store A.
 *
 * Input
 * -----
 * m      specifies the number of rows of the matrix A. It must be at least
 *        zero.
 * n      specifies the number of columns of the matrix A. It must be at
 *        least zero.
 * alpha  single precision complex scalar multiplier applied to x * transpose(y)
 * x      single precision complex array of length at least (1 + (m - 1) * abs(incx))
 * incx   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 * y      single precision complex array of length at least (1 + (n - 1) * abs(incy))
 * incy   specifies the storage spacing between elements of y. incy must not
 *        be zero.
 * A      single precision complex array of dimensions (lda, n).
 * lda    leading dimension of two-dimensional array used to store matrix A
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * conjugate(transpose(y)) + A
 *
 * Reference: http://www.netlib.org/blas/cgerc.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m <0, n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCgercNative
    (JNIEnv *env, jclass cls, jint m, jint n, jobject alpha, jobject x, jint incx, jobject y, jint incy, jobject A, jint lda)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCgerc");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasCgerc");
        return;
    }
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCgerc");
        return;
    }
    cuComplex* nativeX;
    cuComplex* nativeY;
    cuComplex* nativeA;
    cuComplex complexAlpha;

    nativeX = (cuComplex*)getPointer(env, x);
    nativeY = (cuComplex*)getPointer(env, y);
    nativeA = (cuComplex*)getPointer(env, A);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCgerc(%d, %d, [%f,%f], '%s', %d, '%s', %d, '%s', %d)\n",
        m, n, complexAlpha.x, complexAlpha.y, "x", incx, "y", incy, "A", lda);

    cublasCgerc(m, n, complexAlpha, nativeX, incx, nativeY, incy, nativeA, lda);
}




/**
 * <pre>
 * void
 * cublasCher (char uplo, int n, float alpha, const cuComplex *x, int incx,
 *             cuComplex *A, int lda)
 *
 * performs the hermitian rank 1 operation
 *
 *    A = alpha * x * conjugate(transpose(x)) + A,
 *
 * where alpha is a single precision real scalar, x is an n element single
 * precision complex vector and A is an n x n hermitian matrix consisting of
 * single precision complex elements. Matrix A is stored in column major format,
 * and lda is the leading dimension of the two-dimensional array
 * containing A.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or
 *        the lower triangular part of array A. If uplo = 'U' or 'u',
 *        then only the upper triangular part of A may be referenced.
 *        If uplo = 'L' or 'l', then only the lower triangular part of
 *        A may be referenced.
 * n      specifies the number of rows and columns of the matrix A. It
 *        must be at least 0.
 * alpha  single precision real scalar multiplier applied to
 *        x * conjugate(transpose(x))
 * x      single precision complex array of length at least (1 + (n - 1) * abs(incx))
 * incx   specifies the storage spacing between elements of x. incx must
 *        not be zero.
 * A      single precision complex array of dimensions (lda, n). If uplo = 'U' or
 *        'u', then A must contain the upper triangular part of a hermitian
 *        matrix, and the strictly lower triangular part is not referenced.
 *        If uplo = 'L' or 'l', then A contains the lower triangular part
 *        of a hermitian matrix, and the strictly upper triangular part is
 *        not referenced. The imaginary parts of the diagonal elements need
 *        not be set, they are assumed to be zero, and on exit they
 *        are set to zero.
 * lda    leading dimension of the two-dimensional array containing A. lda
 *        must be at least max(1, n).
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * conjugate(transpose(x)) + A
 *
 * Reference: http://www.netlib.org/blas/cher.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCherNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jfloat alpha, jobject x, jint incx, jobject A, jint lda)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCher");
        return;
    }
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCher");
        return;
    }
    cuComplex* nativeX;
    cuComplex* nativeA;

    nativeX = (cuComplex*)getPointer(env, x);
    nativeA = (cuComplex*)getPointer(env, A);

    Logger::log(LOG_TRACE, "Executing cublasCher(%c, %d, %f, '%s', %d, '%s', %d)\n",
        uplo, n, alpha, "x", incx, "A", lda);

    cublasCher((char)uplo, n, alpha, nativeX, incx, nativeA, lda);
}




/**
 * <pre>
 * void
 * cublasChpr (char uplo, int n, float alpha, const cuComplex *x, int incx,
 *             cuComplex *AP)
 *
 * performs the hermitian rank 1 operation
 *
 *    A = alpha * x * conjugate(transpose(x)) + A,
 *
 * where alpha is a single precision real scalar and x is an n element single
 * precision complex vector. A is a hermitian n x n matrix consisting of single
 * precision complex elements that is supplied in packed form.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array AP. If uplo == 'U' or 'u', then the upper
 *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then
 *        the lower triangular part of A is supplied in AP.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  single precision real scalar multiplier applied to x * conjugate(transpose(x)).
 * x      single precision array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * AP     single precision complex array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part
 *        of the hermitian matrix A, packed sequentially, column by column;
 *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part
 *        of the hermitian matrix A, packed sequentially, column by column;
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 *        The imaginary parts of the diagonal elements need not be set, they
 *        are assumed to be zero, and on exit they are set to zero.
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * conjugate(transpose(x)) + A
 *
 * Reference: http://www.netlib.org/blas/chpr.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasChprNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jfloat alpha, jobject x, jint incx, jobject AP)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasChpr");
        return;
    }
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasChpr");
        return;
    }
    cuComplex* nativeX;
    cuComplex* nativeAP;

    nativeX = (cuComplex*)getPointer(env, x);
    nativeAP = (cuComplex*)getPointer(env, AP);

    Logger::log(LOG_TRACE, "Executing cublasChpr(%c, %d, %f, '%s', %d, '%s')\n",
        uplo, n, alpha, "x", incx, "AP");

    cublasChpr((char)uplo, n, alpha, nativeX, incx, nativeAP);
}




/**
 * <pre>
 * void
 * cublasChpr2 (char uplo, int n, cuComplex alpha, const cuComplex *x, int incx,
 *              const cuComplex *y, int incy, cuComplex *AP)
 *
 * performs the hermitian rank 2 operation
 *
 *    A = alpha*x*conjugate(transpose(y)) + conjugate(alpha)*y*conjugate(transpose(x)) + A,
 *
 * where alpha is a single precision complex scalar, and x and y are n element single
 * precision complex vectors. A is a hermitian n x n matrix consisting of single
 * precision complex elements that is supplied in packed form.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array A. If uplo == 'U' or 'u', then only the
 *        upper triangular part of A may be referenced and the lower triangular
 *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
 *        triangular part of A may be referenced and the upper triangular part
 *        of A is inferred.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  single precision complex scalar multiplier applied to x * conjugate(transpose(y)) +
 *        y * conjugate(transpose(x)).
 * x      single precision complex array of length at least (1 + (n - 1) * abs (incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * y      single precision complex array of length at least (1 + (n - 1) * abs (incy)).
 * incy   storage spacing between elements of y. incy must not be zero.
 * AP     single precision complex array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part
 *        of the hermitian matrix A, packed sequentially, column by column;
 *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part
 *        of the hermitian matrix A, packed sequentially, column by column;
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 *        The imaginary parts of the diagonal elements need not be set, they
 *        are assumed to be zero, and on exit they are set to zero.
 *
 * Output
 * ------
 * A      updated according to A = alpha*x*conjugate(transpose(y))
 *                               + conjugate(alpha)*y*conjugate(transpose(x))+A
 *
 * Reference: http://www.netlib.org/blas/chpr2.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasChpr2Native
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jobject alpha, jobject x, jint incx, jobject y, jint incy, jobject AP)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasChpr2");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasChpr2");
        return;
    }
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasChpr2");
        return;
    }
    cuComplex* nativeX;
    cuComplex* nativeY;
    cuComplex* nativeAP;
    cuComplex complexAlpha;

    nativeX = (cuComplex*)getPointer(env, x);
    nativeY = (cuComplex*)getPointer(env, y);
    nativeAP = (cuComplex*)getPointer(env, AP);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasChpr2(%c, %d, [%f,%f], '%s', %d, '%s', %d, '%s')\n",
        uplo, n, complexAlpha.x, complexAlpha.y, "x", incx, "y", incy, "AP");

    cublasChpr2((char)uplo, n, complexAlpha, nativeX, incx, nativeY, incy, nativeAP);
}




/**
 * <pre>
 * void cublasCher2 (char uplo, int n, cuComplex alpha, const cuComplex *x, int incx,
 *                   const cuComplex *y, int incy, cuComplex *A, int lda)
 *
 * performs the hermitian rank 2 operation
 *
 *    A = alpha*x*conjugate(transpose(y)) + conjugate(alpha)*y*conjugate(transpose(x)) + A,
 *
 * where alpha is a single precision complex scalar, x and y are n element single
 * precision complex vector and A is an n by n hermitian matrix consisting of single
 * precision complex elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array A. If uplo == 'U' or 'u', then only the
 *        upper triangular part of A may be referenced and the lower triangular
 *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
 *        triangular part of A may be referenced and the upper triangular part
 *        of A is inferred.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  single precision complex scalar multiplier applied to x * conjugate(transpose(y)) +
 *        y * conjugate(transpose(x)).
 * x      single precision array of length at least (1 + (n - 1) * abs (incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * y      single precision array of length at least (1 + (n - 1) * abs (incy)).
 * incy   storage spacing between elements of y. incy must not be zero.
 * A      single precision complex array of dimensions (lda, n). If uplo == 'U' or 'u',
 *        then A must contains the upper triangular part of a hermitian matrix,
 *        and the strictly lower triangular parts is not referenced. If uplo ==
 *        'L' or 'l', then A contains the lower triangular part of a hermitian
 *        matrix, and the strictly upper triangular part is not referenced.
 *        The imaginary parts of the diagonal elements need not be set,
 *        they are assumed to be zero, and on exit they are set to zero.
 *
 * lda    leading dimension of A. It must be at least max(1, n).
 *
 * Output
 * ------
 * A      updated according to A = alpha*x*conjugate(transpose(y))
 *                               + conjugate(alpha)*y*conjugate(transpose(x))+A
 *
 * Reference: http://www.netlib.org/blas/cher2.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCher2Native
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jobject alpha, jobject x, jint incx, jobject y, jint incy, jobject A, jint lda)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasCher2");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasCher2");
        return;
    }
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCher2");
        return;
    }
    cuComplex* nativeX;
    cuComplex* nativeY;
    cuComplex* nativeA;
    cuComplex complexAlpha;

    nativeX = (cuComplex*)getPointer(env, x);
    nativeY = (cuComplex*)getPointer(env, y);
    nativeA = (cuComplex*)getPointer(env, A);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCher2(%c, %d, [%f,%f], '%s', %d, '%s', %d, '%s', %d)\n",
        uplo, n, complexAlpha.x, complexAlpha.y, "x", incx, "y", incy, "A", lda);

    cublasCher2((char)uplo, n, complexAlpha, nativeX, incx, nativeY, incy, nativeA, lda);
}




/**
 * <pre>
 * void
 * cublasSgemm (char transa, char transb, int m, int n, int k, float alpha,
 *              const float *A, int lda, const float *B, int ldb, float beta,
 *              float *C, int ldc)
 *
 * computes the product of matrix A and matrix B, multiplies the result
 * by a scalar alpha, and adds the sum to the product of matrix C and
 * scalar beta. sgemm() performs one of the matrix-matrix operations:
 *
 *     C = alpha * op(A) * op(B) + beta * C,
 *
 * where op(X) is one of
 *
 *     op(X) = X   or   op(X) = transpose(X)
 *
 * alpha and beta are single precision scalars, and A, B and C are
 * matrices consisting of single precision elements, with op(A) an m x k
 * matrix, op(B) a k x n matrix, and C an m x n matrix. Matrices A, B,
 * and C are stored in column major format, and lda, ldb, and ldc are
 * the leading dimensions of the two-dimensional arrays containing A,
 * B, and C.
 *
 * Input
 * -----
 * transa specifies op(A). If transa = 'n' or 'N', op(A) = A. If
 *        transa = 't', 'T', 'c', or 'C', op(A) = transpose(A)
 * transb specifies op(B). If transb = 'n' or 'N', op(B) = B. If
 *        transb = 't', 'T', 'c', or 'C', op(B) = transpose(B)
 * m      number of rows of matrix op(A) and rows of matrix C
 * n      number of columns of matrix op(B) and number of columns of C
 * k      number of columns of matrix op(A) and number of rows of op(B)
 * alpha  single precision scalar multiplier applied to op(A)op(B)
 * A      single precision array of dimensions (lda, k) if transa =
 *        'n' or 'N'), and of dimensions (lda, m) otherwise. When transa =
 *        'N' or 'n' then lda must be at least  max( 1, m ), otherwise lda
 *        must be at least max(1, k).
 * lda    leading dimension of two-dimensional array used to store matrix A
 * B      single precision array of dimensions  (ldb, n) if transb =
 *        'n' or 'N'), and of dimensions (ldb, k) otherwise. When transb =
 *        'N' or 'n' then ldb must be at least  max (1, k), otherwise ldb
 *        must be at least max (1, n).
 * ldb    leading dimension of two-dimensional array used to store matrix B
 * beta   single precision scalar multiplier applied to C. If 0, C does
 *        not have to be a valid input
 * C      single precision array of dimensions (ldc, n). ldc must be at
 *        least max (1, m).
 * ldc    leading dimension of two-dimensional array used to store matrix C
 *
 * Output
 * ------
 * C      updated based on C = alpha * op(A)*op(B) + beta * C
 *
 * Reference: http://www.netlib.org/blas/sgemm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if any of m, n, or k are < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSgemmNative
    (JNIEnv *env, jclass cls, jchar transa, jchar transb, jint m, jint n, jint k, jfloat alpha, jobject A, jint lda, jobject B, jint ldb, jfloat beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasSgemm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasSgemm");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasSgemm");
        return;
    }
    float* nativeA;
    float* nativeB;
    float* nativeC;

    nativeA = (float*)getPointer(env, A);
    nativeB = (float*)getPointer(env, B);
    nativeC = (float*)getPointer(env, C);

    Logger::log(LOG_TRACE, "Executing cublasSgemm(%c, %c, %d, %d, %d, %f, '%s', %d, '%s', %d, %f, '%s', %d)\n",
        transa, transb, m, n, k, alpha, "A", lda, "B", ldb, beta, "C", ldc);

    cublasSgemm((char)transa, (char)transb, m, n, k, alpha, nativeA, lda, nativeB, ldb, beta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasSsymm (char side, char uplo, int m, int n, float alpha,
 *              const float *A, int lda, const float *B, int ldb,
 *              float beta, float *C, int ldc);
 *
 * performs one of the matrix-matrix operations
 *
 *   C = alpha * A * B + beta * C, or
 *   C = alpha * B * A + beta * C,
 *
 * where alpha and beta are single precision scalars, A is a symmetric matrix
 * consisting of single precision elements and stored in either lower or upper
 * storage mode, and B and C are m x n matrices consisting of single precision
 * elements.
 *
 * Input
 * -----
 * side   specifies whether the symmetric matrix A appears on the left side
 *        hand side or right hand side of matrix B, as follows. If side == 'L'
 *        or 'l', then C = alpha * A * B + beta * C. If side = 'R' or 'r',
 *        then C = alpha * B * A + beta * C.
 * uplo   specifies whether the symmetric matrix A is stored in upper or lower
 *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the symmetric matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the symmetric matrix is to be referenced,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * m      specifies the number of rows of the matrix C, and the number of rows
 *        of matrix B. It also specifies the dimensions of symmetric matrix A
 *        when side == 'L' or 'l'. m must be at least zero.
 * n      specifies the number of columns of the matrix C, and the number of
 *        columns of matrix B. It also specifies the dimensions of symmetric
 *        matrix A when side == 'R' or 'r'. n must be at least zero.
 * alpha  single precision scalar multiplier applied to A * B, or B * A
 * A      single precision array of dimensions (lda, ka), where ka is m when
 *        side == 'L' or 'l' and is n otherwise. If side == 'L' or 'l' the
 *        leading m x m part of array A must contain the symmetric matrix,
 *        such that when uplo == 'U' or 'u', the leading m x m part stores the
 *        upper triangular part of the symmetric matrix, and the strictly lower
 *        triangular part of A is not referenced, and when uplo == 'U' or 'u',
 *        the leading m x m part stores the lower triangular part of the
 *        symmetric matrix and the strictly upper triangular part is not
 *        referenced. If side == 'R' or 'r' the leading n x n part of array A
 *        must contain the symmetric matrix, such that when uplo == 'U' or 'u',
 *        the leading n x n part stores the upper triangular part of the
 *        symmetric matrix and the strictly lower triangular part of A is not
 *        referenced, and when uplo == 'U' or 'u', the leading n x n part
 *        stores the lower triangular part of the symmetric matrix and the
 *        strictly upper triangular part is not referenced.
 * lda    leading dimension of A. When side == 'L' or 'l', it must be at least
 *        max(1, m) and at least max(1, n) otherwise.
 * B      single precision array of dimensions (ldb, n). On entry, the leading
 *        m x n part of the array contains the matrix B.
 * ldb    leading dimension of B. It must be at least max (1, m).
 * beta   single precision scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input
 * C      single precision array of dimensions (ldc, n)
 * ldc    leading dimension of C. Must be at least max(1, m)
 *
 * Output
 * ------
 * C      updated according to C = alpha * A * B + beta * C, or C = alpha *
 *        B * A + beta * C
 *
 * Reference: http://www.netlib.org/blas/ssymm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSsymmNative
    (JNIEnv *env, jclass cls, jchar side, jchar uplo, jint m, jint n, jfloat alpha, jobject A, jint lda, jobject B, jint ldb, jfloat beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasSsymm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasSsymm");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasSsymm");
        return;
    }
    float* nativeA;
    float* nativeB;
    float* nativeC;

    nativeA = (float*)getPointer(env, A);
    nativeB = (float*)getPointer(env, B);
    nativeC = (float*)getPointer(env, C);

    Logger::log(LOG_TRACE, "Executing cublasSsymm(%c, %c, %d, %d, %f, '%s', %d, '%s', %d, %f, '%s', %d)\n",
        side, uplo, m, n, alpha, "A", lda, "B", ldb, beta, "C", ldc);

    cublasSsymm((char)side, (char)uplo, m, n, alpha, nativeA, lda, nativeB, ldb, beta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasSsyrk (char uplo, char trans, int n, int k, float alpha,
 *              const float *A, int lda, float beta, float *C, int ldc)
 *
 * performs one of the symmetric rank k operations
 *
 *   C = alpha * A * transpose(A) + beta * C, or
 *   C = alpha * transpose(A) * A + beta * C.
 *
 * Alpha and beta are single precision scalars. C is an n x n symmetric matrix
 * consisting of single precision elements and stored in either lower or
 * upper storage mode. A is a matrix consisting of single precision elements
 * with dimension of n x k in the first case, and k x n in the second case.
 *
 * Input
 * -----
 * uplo   specifies whether the symmetric matrix C is stored in upper or lower
 *        storage mode as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the symmetric matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the symmetric matrix is to be referenced,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * trans  specifies the operation to be performed. If trans == 'N' or 'n', C =
 *        alpha * transpose(A) + beta * C. If trans == 'T', 't', 'C', or 'c',
 *        C = transpose(A) * A + beta * C.
 * n      specifies the number of rows and the number columns of matrix C. If
 *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
 *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
 *        n must be at least zero.
 * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
 *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
 *        matrix A. k must be at least zero.
 * alpha  single precision scalar multiplier applied to A * transpose(A) or
 *        transpose(A) * A.
 * A      single precision array of dimensions (lda, ka), where ka is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array A must contain the matrix A,
 *        otherwise the leading k x n part of the array must contains the
 *        matrix A.
 * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
 *        least max(1, n). Otherwise lda must be at least max(1, k).
 * beta   single precision scalar multiplier applied to C. If beta izs zero, C
 *        does not have to be a valid input
 * C      single precision array of dimensions (ldc, n). If uplo == 'U' or 'u',
 *        the leading n x n triangular part of the array C must contain the
 *        upper triangular part of the symmetric matrix C and the strictly
 *        lower triangular part of C is not referenced. On exit, the upper
 *        triangular part of C is overwritten by the upper triangular part of
 *        the updated matrix. If uplo == 'L' or 'l', the leading n x n
 *        triangular part of the array C must contain the lower triangular part
 *        of the symmetric matrix C and the strictly upper triangular part of C
 *        is not referenced. On exit, the lower triangular part of C is
 *        overwritten by the lower triangular part of the updated matrix.
 * ldc    leading dimension of C. It must be at least max(1, n).
 *
 * Output
 * ------
 * C      updated according to C = alpha * A * transpose(A) + beta * C, or C =
 *        alpha * transpose(A) * A + beta * C
 *
 * Reference: http://www.netlib.org/blas/ssyrk.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSsyrkNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jint n, jint k, jfloat alpha, jobject A, jint lda, jfloat beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasSsyrk");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasSsyrk");
        return;
    }
    float* nativeA;
    float* nativeC;

    nativeA = (float*)getPointer(env, A);
    nativeC = (float*)getPointer(env, C);

    Logger::log(LOG_TRACE, "Executing cublasSsyrk(%c, %c, %d, %d, %f, '%s', %d, %f, '%s', %d)\n",
        uplo, trans, n, k, alpha, "A", lda, beta, "C", ldc);

    cublasSsyrk((char)uplo, (char)trans, n, k, alpha, nativeA, lda, beta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasSsyr2k (char uplo, char trans, int n, int k, float alpha,
 *               const float *A, int lda, const float *B, int ldb,
 *               float beta, float *C, int ldc)
 *
 * performs one of the symmetric rank 2k operations
 *
 *    C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C, or
 *    C = alpha * transpose(A) * B + alpha * transpose(B) * A + beta * C.
 *
 * Alpha and beta are single precision scalars. C is an n x n symmetric matrix
 * consisting of single precision elements and stored in either lower or upper
 * storage mode. A and B are matrices consisting of single precision elements
 * with dimension of n x k in the first case, and k x n in the second case.
 *
 * Input
 * -----
 * uplo   specifies whether the symmetric matrix C is stored in upper or lower
 *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the symmetric matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the symmetric matrix is to be references,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * trans  specifies the operation to be performed. If trans == 'N' or 'n',
 *        C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C,
 *        If trans == 'T', 't', 'C', or 'c', C = alpha * transpose(A) * B +
 *        alpha * transpose(B) * A + beta * C.
 * n      specifies the number of rows and the number columns of matrix C. If
 *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
 *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
 *        n must be at least zero.
 * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
 *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
 *        matrix A. k must be at least zero.
 * alpha  single precision scalar multiplier.
 * A      single precision array of dimensions (lda, ka), where ka is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array A must contain the matrix A,
 *        otherwise the leading k x n part of the array must contain the matrix
 *        A.
 * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
 *        least max(1, n). Otherwise lda must be at least max(1,k).
 * B      single precision array of dimensions (lda, kb), where kb is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array B must contain the matrix B,
 *        otherwise the leading k x n part of the array must contain the matrix
 *        B.
 * ldb    leading dimension of N. When trans == 'N' or 'n' then ldb must be at
 *        least max(1, n). Otherwise ldb must be at least max(1, k).
 * beta   single precision scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input.
 * C      single precision array of dimensions (ldc, n). If uplo == 'U' or 'u',
 *        the leading n x n triangular part of the array C must contain the
 *        upper triangular part of the symmetric matrix C and the strictly
 *        lower triangular part of C is not referenced. On exit, the upper
 *        triangular part of C is overwritten by the upper triangular part of
 *        the updated matrix. If uplo == 'L' or 'l', the leading n x n
 *        triangular part of the array C must contain the lower triangular part
 *        of the symmetric matrix C and the strictly upper triangular part of C
 *        is not referenced. On exit, the lower triangular part of C is
 *        overwritten by the lower triangular part of the updated matrix.
 * ldc    leading dimension of C. Must be at least max(1, n).
 *
 * Output
 * ------
 * C      updated according to alpha*A*transpose(B) + alpha*B*transpose(A) +
 *        beta*C or alpha*transpose(A)*B + alpha*transpose(B)*A + beta*C
 *
 * Reference:   http://www.netlib.org/blas/ssyr2k.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasSsyr2kNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jint n, jint k, jfloat alpha, jobject A, jint lda, jobject B, jint ldb, jfloat beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasSsyr2k");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasSsyr2k");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasSsyr2k");
        return;
    }
    float* nativeA;
    float* nativeB;
    float* nativeC;

    nativeA = (float*)getPointer(env, A);
    nativeB = (float*)getPointer(env, B);
    nativeC = (float*)getPointer(env, C);

    Logger::log(LOG_TRACE, "Executing cublasSsyr2k(%c, %c, %d, %d, %f, '%s', %d, '%s', %d, %f, '%s', %d)\n",
        uplo, trans, n, k, alpha, "A", lda, "B", ldb, beta, "C", ldc);

    cublasSsyr2k((char)uplo, (char)trans, n, k, alpha, nativeA, lda, nativeB, ldb, beta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasStrmm (char side, char uplo, char transa, char diag, int m, int n,
 *              float alpha, const float *A, int lda, const float *B, int ldb)
 *
 * performs one of the matrix-matrix operations
 *
 *   B = alpha * op(A) * B,  or  B = alpha * B * op(A)
 *
 * where alpha is a single-precision scalar, B is an m x n matrix composed
 * of single precision elements, and A is a unit or non-unit, upper or lower,
 * triangular matrix composed of single precision elements. op(A) is one of
 *
 *   op(A) = A  or  op(A) = transpose(A)
 *
 * Matrices A and B are stored in column major format, and lda and ldb are
 * the leading dimensions of the two-dimensonials arrays that contain A and
 * B, respectively.
 *
 * Input
 * -----
 * side   specifies whether op(A) multiplies B from the left or right.
 *        If side = 'L' or 'l', then B = alpha * op(A) * B. If side =
 *        'R' or 'r', then B = alpha * B * op(A).
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix. If uplo = 'U' or 'u', A is an upper triangular matrix.
 *        If uplo = 'L' or 'l', A is a lower triangular matrix.
 * transa specifies the form of op(A) to be used in the matrix
 *        multiplication. If transa = 'N' or 'n', then op(A) = A. If
 *        transa = 'T', 't', 'C', or 'c', then op(A) = transpose(A).
 * diag   specifies whether or not A is unit triangular. If diag = 'U'
 *        or 'u', A is assumed to be unit triangular. If diag = 'N' or
 *        'n', A is not assumed to be unit triangular.
 * m      the number of rows of matrix B. m must be at least zero.
 * n      the number of columns of matrix B. n must be at least zero.
 * alpha  single precision scalar multiplier applied to op(A)*B, or
 *        B*op(A), respectively. If alpha is zero no accesses are made
 *        to matrix A, and no read accesses are made to matrix B.
 * A      single precision array of dimensions (lda, k). k = m if side =
 *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
 *        the leading k x k upper triangular part of the array A must
 *        contain the upper triangular matrix, and the strictly lower
 *        triangular part of A is not referenced. If uplo = 'L' or 'l'
 *        the leading k x k lower triangular part of the array A must
 *        contain the lower triangular matrix, and the strictly upper
 *        triangular part of A is not referenced. When diag = 'U' or 'u'
 *        the diagonal elements of A are no referenced and are assumed
 *        to be unity.
 * lda    leading dimension of A. When side = 'L' or 'l', it must be at
 *        least max(1,m) and at least max(1,n) otherwise
 * B      single precision array of dimensions (ldb, n). On entry, the
 *        leading m x n part of the array contains the matrix B. It is
 *        overwritten with the transformed matrix on exit.
 * ldb    leading dimension of B. It must be at least max (1, m).
 *
 * Output
 * ------
 * B      updated according to B = alpha * op(A) * B  or B = alpha * B * op(A)
 *
 * Reference: http://www.netlib.org/blas/strmm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasStrmmNative
    (JNIEnv *env, jclass cls, jchar side, jchar uplo, jchar transa, jchar diag, jint m, jint n, jfloat alpha, jobject A, jint lda, jobject B, jint ldb)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasStrmm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasStrmm");
        return;
    }
    float* nativeA;
    float* nativeB;

    nativeA = (float*)getPointer(env, A);
    nativeB = (float*)getPointer(env, B);

    Logger::log(LOG_TRACE, "Executing cublasStrmm(%c, %c, %c, %c, %d, %d, %f, '%s', %d, '%s', %d)\n",
        side, uplo, transa, diag, m, n, alpha, "A", lda, "B", ldb);

    cublasStrmm((char)side, (char)uplo, (char)transa, (char)diag, m, n, alpha, nativeA, lda, nativeB, ldb);
}




/**
 * <pre>
 * void
 * cublasStrsm (char side, char uplo, char transa, char diag, int m, int n,
 *              float alpha, const float *A, int lda, float *B, int ldb)
 *
 * solves one of the matrix equations
 *
 *    op(A) * X = alpha * B,   or   X * op(A) = alpha * B,
 *
 * where alpha is a single precision scalar, and X and B are m x n matrices
 * that are composed of single precision elements. A is a unit or non-unit,
 * upper or lower triangular matrix, and op(A) is one of
 *
 *    op(A) = A  or  op(A) = transpose(A)
 *
 * The result matrix X overwrites input matrix B; that is, on exit the result
 * is stored in B. Matrices A and B are stored in column major format, and
 * lda and ldb are the leading dimensions of the two-dimensonials arrays that
 * contain A and B, respectively.
 *
 * Input
 * -----
 * side   specifies whether op(A) appears on the left or right of X as
 *        follows: side = 'L' or 'l' indicates solve op(A) * X = alpha * B.
 *        side = 'R' or 'r' indicates solve X * op(A) = alpha * B.
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix as follows: uplo = 'U' or 'u' indicates A is an upper
 *        triangular matrix. uplo = 'L' or 'l' indicates A is a lower
 *        triangular matrix.
 * transa specifies the form of op(A) to be used in matrix multiplication
 *        as follows: If transa = 'N' or 'N', then op(A) = A. If transa =
 *        'T', 't', 'C', or 'c', then op(A) = transpose(A).
 * diag   specifies whether or not A is a unit triangular matrix like so:
 *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
 *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
 * m      specifies the number of rows of B. m must be at least zero.
 * n      specifies the number of columns of B. n must be at least zero.
 * alpha  is a single precision scalar to be multiplied with B. When alpha is
 *        zero, then A is not referenced and B need not be set before entry.
 * A      is a single precision array of dimensions (lda, k), where k is
 *        m when side = 'L' or 'l', and is n when side = 'R' or 'r'. If
 *        uplo = 'U' or 'u', the leading k x k upper triangular part of
 *        the array A must contain the upper triangular matrix and the
 *        strictly lower triangular matrix of A is not referenced. When
 *        uplo = 'L' or 'l', the leading k x k lower triangular part of
 *        the array A must contain the lower triangular matrix and the
 *        strictly upper triangular part of A is not referenced. Note that
 *        when diag = 'U' or 'u', the diagonal elements of A are not
 *        referenced, and are assumed to be unity.
 * lda    is the leading dimension of the two dimensional array containing A.
 *        When side = 'L' or 'l' then lda must be at least max(1, m), when
 *        side = 'R' or 'r' then lda must be at least max(1, n).
 * B      is a single precision array of dimensions (ldb, n). ldb must be
 *        at least max (1,m). The leading m x n part of the array B must
 *        contain the right-hand side matrix B. On exit B is overwritten
 *        by the solution matrix X.
 * ldb    is the leading dimension of the two dimensional array containing B.
 *        ldb must be at least max(1, m).
 *
 * Output
 * ------
 * B      contains the solution matrix X satisfying op(A) * X = alpha * B,
 *        or X * op(A) = alpha * B
 *
 * Reference: http://www.netlib.org/blas/strsm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasStrsmNative
    (JNIEnv *env, jclass cls, jchar side, jchar uplo, jchar transa, jchar diag, jint m, jint n, jfloat alpha, jobject A, jint lda, jobject B, jint ldb)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasStrsm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasStrsm");
        return;
    }
    float* nativeA;
    float* nativeB;

    nativeA = (float*)getPointer(env, A);
    nativeB = (float*)getPointer(env, B);

    Logger::log(LOG_TRACE, "Executing cublasStrsm(%c, %c, %c, %c, %d, %d, %f, '%s', %d, '%s', %d)\n",
        side, uplo, transa, diag, m, n, alpha, "A", lda, "B", ldb);

    cublasStrsm((char)side, (char)uplo, (char)transa, (char)diag, m, n, alpha, nativeA, lda, nativeB, ldb);
}




/**
 * <pre>
 * void cublasCgemm (char transa, char transb, int m, int n, int k,
 *                   cuComplex alpha, const cuComplex *A, int lda,
 *                   const cuComplex *B, int ldb, cuComplex beta,
 *                   cuComplex *C, int ldc)
 *
 * performs one of the matrix-matrix operations
 *
 *    C = alpha * op(A) * op(B) + beta*C,
 *
 * where op(X) is one of
 *
 *    op(X) = X   or   op(X) = transpose  or  op(X) = conjg(transpose(X))
 *
 * alpha and beta are single-complex scalars, and A, B and C are matrices
 * consisting of single-complex elements, with op(A) an m x k matrix, op(B)
 * a k x n matrix and C an m x n matrix.
 *
 * Input
 * -----
 * transa specifies op(A). If transa == 'N' or 'n', op(A) = A. If transa ==
 *        'T' or 't', op(A) = transpose(A). If transa == 'C' or 'c', op(A) =
 *        conjg(transpose(A)).
 * transb specifies op(B). If transa == 'N' or 'n', op(B) = B. If transb ==
 *        'T' or 't', op(B) = transpose(B). If transb == 'C' or 'c', op(B) =
 *        conjg(transpose(B)).
 * m      number of rows of matrix op(A) and rows of matrix C. It must be at
 *        least zero.
 * n      number of columns of matrix op(B) and number of columns of C. It
 *        must be at least zero.
 * k      number of columns of matrix op(A) and number of rows of op(B). It
 *        must be at least zero.
 * alpha  single-complex scalar multiplier applied to op(A)op(B)
 * A      single-complex array of dimensions (lda, k) if transa ==  'N' or
 *        'n'), and of dimensions (lda, m) otherwise.
 * lda    leading dimension of A. When transa == 'N' or 'n', it must be at
 *        least max(1, m) and at least max(1, k) otherwise.
 * B      single-complex array of dimensions (ldb, n) if transb == 'N' or 'n',
 *        and of dimensions (ldb, k) otherwise
 * ldb    leading dimension of B. When transb == 'N' or 'n', it must be at
 *        least max(1, k) and at least max(1, n) otherwise.
 * beta   single-complex scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input.
 * C      single precision array of dimensions (ldc, n)
 * ldc    leading dimension of C. Must be at least max(1, m).
 *
 * Output
 * ------
 * C      updated according to C = alpha*op(A)*op(B) + beta*C
 *
 * Reference: http://www.netlib.org/blas/cgemm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if any of m, n, or k are < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCgemmNative
    (JNIEnv *env, jclass cls, jchar transa, jchar transb, jint m, jint n, jint k, jobject alpha, jobject A, jint lda, jobject B, jint ldb, jobject beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCgemm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasCgemm");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasCgemm");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeB;
    cuComplex* nativeC;
    cuComplex complexAlpha;
    cuComplex complexBeta;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeB = (cuComplex*)getPointer(env, B);
    nativeC = (cuComplex*)getPointer(env, C);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    complexBeta.x = env->GetFloatField(beta, cuComplex_x);
    complexBeta.y = env->GetFloatField(beta, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCgemm(%c, %c, %d, %d, %d, [%f,%f], '%s', %d, '%s', %d, [%f,%f], '%s', %d)\n",
        transa, transb, m, n, k, complexAlpha.x, complexAlpha.y, "A", lda, "B", ldb, complexBeta.x, complexBeta.y, "C", ldc);

    cublasCgemm((char)transa, (char)transb, m, n, k, complexAlpha, nativeA, lda, nativeB, ldb, complexBeta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasCsymm (char side, char uplo, int m, int n, cuComplex alpha,
 *              const cuComplex *A, int lda, const cuComplex *B, int ldb,
 *              cuComplex beta, cuComplex *C, int ldc);
 *
 * performs one of the matrix-matrix operations
 *
 *   C = alpha * A * B + beta * C, or
 *   C = alpha * B * A + beta * C,
 *
 * where alpha and beta are single precision complex scalars, A is a symmetric matrix
 * consisting of single precision complex elements and stored in either lower or upper
 * storage mode, and B and C are m x n matrices consisting of single precision
 * complex elements.
 *
 * Input
 * -----
 * side   specifies whether the symmetric matrix A appears on the left side
 *        hand side or right hand side of matrix B, as follows. If side == 'L'
 *        or 'l', then C = alpha * A * B + beta * C. If side = 'R' or 'r',
 *        then C = alpha * B * A + beta * C.
 * uplo   specifies whether the symmetric matrix A is stored in upper or lower
 *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the symmetric matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the symmetric matrix is to be referenced,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * m      specifies the number of rows of the matrix C, and the number of rows
 *        of matrix B. It also specifies the dimensions of symmetric matrix A
 *        when side == 'L' or 'l'. m must be at least zero.
 * n      specifies the number of columns of the matrix C, and the number of
 *        columns of matrix B. It also specifies the dimensions of symmetric
 *        matrix A when side == 'R' or 'r'. n must be at least zero.
 * alpha  single precision scalar multiplier applied to A * B, or B * A
 * A      single precision array of dimensions (lda, ka), where ka is m when
 *        side == 'L' or 'l' and is n otherwise. If side == 'L' or 'l' the
 *        leading m x m part of array A must contain the symmetric matrix,
 *        such that when uplo == 'U' or 'u', the leading m x m part stores the
 *        upper triangular part of the symmetric matrix, and the strictly lower
 *        triangular part of A is not referenced, and when uplo == 'U' or 'u',
 *        the leading m x m part stores the lower triangular part of the
 *        symmetric matrix and the strictly upper triangular part is not
 *        referenced. If side == 'R' or 'r' the leading n x n part of array A
 *        must contain the symmetric matrix, such that when uplo == 'U' or 'u',
 *        the leading n x n part stores the upper triangular part of the
 *        symmetric matrix and the strictly lower triangular part of A is not
 *        referenced, and when uplo == 'U' or 'u', the leading n x n part
 *        stores the lower triangular part of the symmetric matrix and the
 *        strictly upper triangular part is not referenced.
 * lda    leading dimension of A. When side == 'L' or 'l', it must be at least
 *        max(1, m) and at least max(1, n) otherwise.
 * B      single precision array of dimensions (ldb, n). On entry, the leading
 *        m x n part of the array contains the matrix B.
 * ldb    leading dimension of B. It must be at least max (1, m).
 * beta   single precision scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input
 * C      single precision array of dimensions (ldc, n)
 * ldc    leading dimension of C. Must be at least max(1, m)
 *
 * Output
 * ------
 * C      updated according to C = alpha * A * B + beta * C, or C = alpha *
 *        B * A + beta * C
 *
 * Reference: http://www.netlib.org/blas/csymm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCsymmNative
    (JNIEnv *env, jclass cls, jchar side, jchar uplo, jint m, jint n, jobject alpha, jobject A, jint lda, jobject B, jint ldb, jobject beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCsymm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasCsymm");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasCsymm");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeB;
    cuComplex* nativeC;
    cuComplex complexAlpha;
    cuComplex complexBeta;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeB = (cuComplex*)getPointer(env, B);
    nativeC = (cuComplex*)getPointer(env, C);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    complexBeta.x = env->GetFloatField(beta, cuComplex_x);
    complexBeta.y = env->GetFloatField(beta, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCsymm(%c, %c, %d, %d, [%f,%f], '%s', %d, '%s', %d, [%f,%f], '%s', %d)\n",
        side, uplo, m, n, complexAlpha.x, complexAlpha.y, "A", lda, "B", ldb, complexBeta.x, complexBeta.y, "C", ldc);

    cublasCsymm((char)side, (char)uplo, m, n, complexAlpha, nativeA, lda, nativeB, ldb, complexBeta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasChemm (char side, char uplo, int m, int n, cuComplex alpha,
 *              const cuComplex *A, int lda, const cuComplex *B, int ldb,
 *              cuComplex beta, cuComplex *C, int ldc);
 *
 * performs one of the matrix-matrix operations
 *
 *   C = alpha * A * B + beta * C, or
 *   C = alpha * B * A + beta * C,
 *
 * where alpha and beta are single precision complex scalars, A is a hermitian matrix
 * consisting of single precision complex elements and stored in either lower or upper
 * storage mode, and B and C are m x n matrices consisting of single precision
 * complex elements.
 *
 * Input
 * -----
 * side   specifies whether the hermitian matrix A appears on the left side
 *        hand side or right hand side of matrix B, as follows. If side == 'L'
 *        or 'l', then C = alpha * A * B + beta * C. If side = 'R' or 'r',
 *        then C = alpha * B * A + beta * C.
 * uplo   specifies whether the hermitian matrix A is stored in upper or lower
 *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the hermitian matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the hermitian matrix is to be referenced,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * m      specifies the number of rows of the matrix C, and the number of rows
 *        of matrix B. It also specifies the dimensions of hermitian matrix A
 *        when side == 'L' or 'l'. m must be at least zero.
 * n      specifies the number of columns of the matrix C, and the number of
 *        columns of matrix B. It also specifies the dimensions of hermitian
 *        matrix A when side == 'R' or 'r'. n must be at least zero.
 * alpha  single precision complex scalar multiplier applied to A * B, or B * A
 * A      single precision complex array of dimensions (lda, ka), where ka is m when
 *        side == 'L' or 'l' and is n otherwise. If side == 'L' or 'l' the
 *        leading m x m part of array A must contain the hermitian matrix,
 *        such that when uplo == 'U' or 'u', the leading m x m part stores the
 *        upper triangular part of the hermitian matrix, and the strictly lower
 *        triangular part of A is not referenced, and when uplo == 'U' or 'u',
 *        the leading m x m part stores the lower triangular part of the
 *        hermitian matrix and the strictly upper triangular part is not
 *        referenced. If side == 'R' or 'r' the leading n x n part of array A
 *        must contain the hermitian matrix, such that when uplo == 'U' or 'u',
 *        the leading n x n part stores the upper triangular part of the
 *        hermitian matrix and the strictly lower triangular part of A is not
 *        referenced, and when uplo == 'U' or 'u', the leading n x n part
 *        stores the lower triangular part of the hermitian matrix and the
 *        strictly upper triangular part is not referenced. The imaginary parts
 *        of the diagonal elements need not be set, they are assumed to be zero.
 * lda    leading dimension of A. When side == 'L' or 'l', it must be at least
 *        max(1, m) and at least max(1, n) otherwise.
 * B      single precision complex array of dimensions (ldb, n). On entry, the leading
 *        m x n part of the array contains the matrix B.
 * ldb    leading dimension of B. It must be at least max (1, m).
 * beta   single precision complex scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input
 * C      single precision complex array of dimensions (ldc, n)
 * ldc    leading dimension of C. Must be at least max(1, m)
 *
 * Output
 * ------
 * C      updated according to C = alpha * A * B + beta * C, or C = alpha *
 *        B * A + beta * C
 *
 * Reference: http://www.netlib.org/blas/chemm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasChemmNative
    (JNIEnv *env, jclass cls, jchar side, jchar uplo, jint m, jint n, jobject alpha, jobject A, jint lda, jobject B, jint ldb, jobject beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasChemm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasChemm");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasChemm");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeB;
    cuComplex* nativeC;
    cuComplex complexAlpha;
    cuComplex complexBeta;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeB = (cuComplex*)getPointer(env, B);
    nativeC = (cuComplex*)getPointer(env, C);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    complexBeta.x = env->GetFloatField(beta, cuComplex_x);
    complexBeta.y = env->GetFloatField(beta, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasChemm(%c, %c, %d, %d, [%f,%f], '%s', %d, '%s', %d, [%f,%f], '%s', %d)\n",
        side, uplo, m, n, complexAlpha.x, complexAlpha.y, "A", lda, "B", ldb, complexBeta.x, complexBeta.y, "C", ldc);

    cublasChemm((char)side, (char)uplo, m, n, complexAlpha, nativeA, lda, nativeB, ldb, complexBeta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasCsyrk (char uplo, char trans, int n, int k, cuComplex alpha,
 *              const cuComplex *A, int lda, cuComplex beta, cuComplex *C, int ldc)
 *
 * performs one of the symmetric rank k operations
 *
 *   C = alpha * A * transpose(A) + beta * C, or
 *   C = alpha * transpose(A) * A + beta * C.
 *
 * Alpha and beta are single precision complex scalars. C is an n x n symmetric matrix
 * consisting of single precision complex elements and stored in either lower or
 * upper storage mode. A is a matrix consisting of single precision complex elements
 * with dimension of n x k in the first case, and k x n in the second case.
 *
 * Input
 * -----
 * uplo   specifies whether the symmetric matrix C is stored in upper or lower
 *        storage mode as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the symmetric matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the symmetric matrix is to be referenced,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * trans  specifies the operation to be performed. If trans == 'N' or 'n', C =
 *        alpha * transpose(A) + beta * C. If trans == 'T', 't', 'C', or 'c',
 *        C = transpose(A) * A + beta * C.
 * n      specifies the number of rows and the number columns of matrix C. If
 *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
 *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
 *        n must be at least zero.
 * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
 *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
 *        matrix A. k must be at least zero.
 * alpha  single precision complex scalar multiplier applied to A * transpose(A) or
 *        transpose(A) * A.
 * A      single precision complex array of dimensions (lda, ka), where ka is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array A must contain the matrix A,
 *        otherwise the leading k x n part of the array must contains the
 *        matrix A.
 * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
 *        least max(1, n). Otherwise lda must be at least max(1, k).
 * beta   single precision complex scalar multiplier applied to C. If beta izs zero, C
 *        does not have to be a valid input
 * C      single precision complex array of dimensions (ldc, n). If uplo = 'U' or 'u',
 *        the leading n x n triangular part of the array C must contain the
 *        upper triangular part of the symmetric matrix C and the strictly
 *        lower triangular part of C is not referenced. On exit, the upper
 *        triangular part of C is overwritten by the upper triangular part of
 *        the updated matrix. If uplo = 'L' or 'l', the leading n x n
 *        triangular part of the array C must contain the lower triangular part
 *        of the symmetric matrix C and the strictly upper triangular part of C
 *        is not referenced. On exit, the lower triangular part of C is
 *        overwritten by the lower triangular part of the updated matrix.
 * ldc    leading dimension of C. It must be at least max(1, n).
 *
 * Output
 * ------
 * C      updated according to C = alpha * A * transpose(A) + beta * C, or C =
 *        alpha * transpose(A) * A + beta * C
 *
 * Reference: http://www.netlib.org/blas/csyrk.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCsyrkNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jint n, jint k, jobject alpha, jobject A, jint lda, jobject beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCsyrk");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasCsyrk");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeC;
    cuComplex complexAlpha;
    cuComplex complexBeta;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeC = (cuComplex*)getPointer(env, C);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    complexBeta.x = env->GetFloatField(beta, cuComplex_x);
    complexBeta.y = env->GetFloatField(beta, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCsyrk(%c, %c, %d, %d, [%f,%f], '%s', %d, [%f,%f], '%s', %d)\n",
        uplo, trans, n, k, complexAlpha.x, complexAlpha.y, "A", lda, complexBeta.x, complexBeta.y, "C", ldc);

    cublasCsyrk((char)uplo, (char)trans, n, k, complexAlpha, nativeA, lda, complexBeta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasCherk (char uplo, char trans, int n, int k, float alpha,
 *              const cuComplex *A, int lda, float beta, cuComplex *C, int ldc)
 *
 * performs one of the hermitian rank k operations
 *
 *   C = alpha * A * conjugate(transpose(A)) + beta * C, or
 *   C = alpha * conjugate(transpose(A)) * A + beta * C.
 *
 * Alpha and beta are single precision real scalars. C is an n x n hermitian matrix
 * consisting of single precision complex elements and stored in either lower or
 * upper storage mode. A is a matrix consisting of single precision complex elements
 * with dimension of n x k in the first case, and k x n in the second case.
 *
 * Input
 * -----
 * uplo   specifies whether the hermitian matrix C is stored in upper or lower
 *        storage mode as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the hermitian matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the hermitian matrix is to be referenced,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * trans  specifies the operation to be performed. If trans == 'N' or 'n', C =
 *        alpha * A * conjugate(transpose(A)) + beta * C. If trans == 'T', 't', 'C', or 'c',
 *        C = alpha * conjugate(transpose(A)) * A + beta * C.
 * n      specifies the number of rows and the number columns of matrix C. If
 *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
 *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
 *        n must be at least zero.
 * k      If trans == 'N' or 'n', k specifies the number of columns of matrix A.
 *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
 *        matrix A. k must be at least zero.
 * alpha  single precision scalar multiplier applied to A * conjugate(transpose(A)) or
 *        conjugate(transpose(A)) * A.
 * A      single precision complex array of dimensions (lda, ka), where ka is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array A must contain the matrix A,
 *        otherwise the leading k x n part of the array must contains the
 *        matrix A.
 * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
 *        least max(1, n). Otherwise lda must be at least max(1, k).
 * beta   single precision scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input.
 * C      single precision complex array of dimensions (ldc, n). If uplo = 'U' or 'u',
 *        the leading n x n triangular part of the array C must contain the
 *        upper triangular part of the hermitian matrix C and the strictly
 *        lower triangular part of C is not referenced. On exit, the upper
 *        triangular part of C is overwritten by the upper triangular part of
 *        the updated matrix. If uplo = 'L' or 'l', the leading n x n
 *        triangular part of the array C must contain the lower triangular part
 *        of the hermitian matrix C and the strictly upper triangular part of C
 *        is not referenced. On exit, the lower triangular part of C is
 *        overwritten by the lower triangular part of the updated matrix.
 *        The imaginary parts of the diagonal elements need
 *        not be set,  they are assumed to be zero,  and on exit they
 *        are set to zero.
 * ldc    leading dimension of C. It must be at least max(1, n).
 *
 * Output
 * ------
 * C      updated according to C = alpha * A * conjugate(transpose(A)) + beta * C, or C =
 *        alpha * conjugate(transpose(A)) * A + beta * C
 *
 * Reference: http://www.netlib.org/blas/cherk.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCherkNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jint n, jint k, jfloat alpha, jobject A, jint lda, jfloat beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCherk");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasCherk");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeC;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeC = (cuComplex*)getPointer(env, C);

    Logger::log(LOG_TRACE, "Executing cublasCherk(%c, %c, %d, %d, %f, '%s', %d, %f, '%s', %d)\n",
        uplo, trans, n, k, alpha, "A", lda, beta, "C", ldc);

    cublasCherk((char)uplo, (char)trans, n, k, alpha, nativeA, lda, beta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasCsyr2k (char uplo, char trans, int n, int k, cuComplex alpha,
 *               const cuComplex *A, int lda, const cuComplex *B, int ldb,
 *               cuComplex beta, cuComplex *C, int ldc)
 *
 * performs one of the symmetric rank 2k operations
 *
 *    C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C, or
 *    C = alpha * transpose(A) * B + alpha * transpose(B) * A + beta * C.
 *
 * Alpha and beta are single precision complex scalars. C is an n x n symmetric matrix
 * consisting of single precision complex elements and stored in either lower or upper
 * storage mode. A and B are matrices consisting of single precision complex elements
 * with dimension of n x k in the first case, and k x n in the second case.
 *
 * Input
 * -----
 * uplo   specifies whether the symmetric matrix C is stored in upper or lower
 *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the symmetric matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the symmetric matrix is to be references,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * trans  specifies the operation to be performed. If trans == 'N' or 'n',
 *        C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C,
 *        If trans == 'T', 't', 'C', or 'c', C = alpha * transpose(A) * B +
 *        alpha * transpose(B) * A + beta * C.
 * n      specifies the number of rows and the number columns of matrix C. If
 *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
 *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
 *        n must be at least zero.
 * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
 *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
 *        matrix A. k must be at least zero.
 * alpha  single precision complex scalar multiplier.
 * A      single precision complex array of dimensions (lda, ka), where ka is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array A must contain the matrix A,
 *        otherwise the leading k x n part of the array must contain the matrix
 *        A.
 * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
 *        least max(1, n). Otherwise lda must be at least max(1,k).
 * B      single precision complex array of dimensions (lda, kb), where kb is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array B must contain the matrix B,
 *        otherwise the leading k x n part of the array must contain the matrix
 *        B.
 * ldb    leading dimension of N. When trans == 'N' or 'n' then ldb must be at
 *        least max(1, n). Otherwise ldb must be at least max(1, k).
 * beta   single precision complex scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input.
 * C      single precision complex array of dimensions (ldc, n). If uplo == 'U' or 'u',
 *        the leading n x n triangular part of the array C must contain the
 *        upper triangular part of the symmetric matrix C and the strictly
 *        lower triangular part of C is not referenced. On exit, the upper
 *        triangular part of C is overwritten by the upper triangular part of
 *        the updated matrix. If uplo == 'L' or 'l', the leading n x n
 *        triangular part of the array C must contain the lower triangular part
 *        of the symmetric matrix C and the strictly upper triangular part of C
 *        is not referenced. On exit, the lower triangular part of C is
 *        overwritten by the lower triangular part of the updated matrix.
 * ldc    leading dimension of C. Must be at least max(1, n).
 *
 * Output
 * ------
 * C      updated according to alpha*A*transpose(B) + alpha*B*transpose(A) +
 *        beta*C or alpha*transpose(A)*B + alpha*transpose(B)*A + beta*C
 *
 * Reference:   http://www.netlib.org/blas/csyr2k.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCsyr2kNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jint n, jint k, jobject alpha, jobject A, jint lda, jobject B, jint ldb, jobject beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCsyr2k");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasCsyr2k");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasCsyr2k");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeB;
    cuComplex* nativeC;
    cuComplex complexAlpha;
    cuComplex complexBeta;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeB = (cuComplex*)getPointer(env, B);
    nativeC = (cuComplex*)getPointer(env, C);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    complexBeta.x = env->GetFloatField(beta, cuComplex_x);
    complexBeta.y = env->GetFloatField(beta, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCsyr2k(%c, %c, %d, %d, [%f,%f], '%s', %d, '%s', %d, [%f,%f], '%s', %d)\n",
        uplo, trans, n, k, complexAlpha.x, complexAlpha.y, "A", lda, "B", ldb, complexBeta.x, complexBeta.y, "C", ldc);

    cublasCsyr2k((char)uplo, (char)trans, n, k, complexAlpha, nativeA, lda, nativeB, ldb, complexBeta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasCher2k (char uplo, char trans, int n, int k, cuComplex alpha,
 *               const cuComplex *A, int lda, const cuComplex *B, int ldb,
 *               float beta, cuComplex *C, int ldc)
 *
 * performs one of the hermitian rank 2k operations
 *
 *    C =   alpha * A * conjugate(transpose(B))
 *        + conjugate(alpha) * B * conjugate(transpose(A))
 *        + beta * C ,
 *    or
 *    C =  alpha * conjugate(transpose(A)) * B
 *       + conjugate(alpha) * conjugate(transpose(B)) * A
 *       + beta * C.
 *
 * Alpha is single precision complex scalar whereas Beta is a single preocision real scalar.
 * C is an n x n hermitian matrix consisting of single precision complex elements
 * and stored in either lower or upper storage mode. A and B are matrices consisting
 * of single precision complex elements with dimension of n x k in the first case,
 * and k x n in the second case.
 *
 * Input
 * -----
 * uplo   specifies whether the hermitian matrix C is stored in upper or lower
 *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the hermitian matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the hermitian matrix is to be references,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * trans  specifies the operation to be performed. If trans == 'N' or 'n',
 *        C =   alpha * A * conjugate(transpose(B))
 *            + conjugate(alpha) * B * conjugate(transpose(A))
 *            + beta * C .
 *        If trans == 'T', 't', 'C', or 'c',
 *        C =  alpha * conjugate(transpose(A)) * B
 *          + conjugate(alpha) * conjugate(transpose(B)) * A
 *          + beta * C.
 * n      specifies the number of rows and the number columns of matrix C. If
 *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
 *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
 *        n must be at least zero.
 * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
 *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
 *        matrix A. k must be at least zero.
 * alpha  single precision complex scalar multiplier.
 * A      single precision complex array of dimensions (lda, ka), where ka is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array A must contain the matrix A,
 *        otherwise the leading k x n part of the array must contain the matrix
 *        A.
 * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
 *        least max(1, n). Otherwise lda must be at least max(1,k).
 * B      single precision complex array of dimensions (lda, kb), where kb is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array B must contain the matrix B,
 *        otherwise the leading k x n part of the array must contain the matrix
 *        B.
 * ldb    leading dimension of N. When trans == 'N' or 'n' then ldb must be at
 *        least max(1, n). Otherwise ldb must be at least max(1, k).
 * beta   single precision scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input.
 * C      single precision complex array of dimensions (ldc, n). If uplo == 'U' or 'u',
 *        the leading n x n triangular part of the array C must contain the
 *        upper triangular part of the hermitian matrix C and the strictly
 *        lower triangular part of C is not referenced. On exit, the upper
 *        triangular part of C is overwritten by the upper triangular part of
 *        the updated matrix. If uplo == 'L' or 'l', the leading n x n
 *        triangular part of the array C must contain the lower triangular part
 *        of the hermitian matrix C and the strictly upper triangular part of C
 *        is not referenced. On exit, the lower triangular part of C is
 *        overwritten by the lower triangular part of the updated matrix.
 *        The imaginary parts of the diagonal elements need
 *        not be set,  they are assumed to be zero,  and on exit they
 *        are set to zero.
 * ldc    leading dimension of C. Must be at least max(1, n).
 *
 * Output
 * ------
 * C      updated according to alpha*A*conjugate(transpose(B)) +
 *        + conjugate(alpha)*B*conjugate(transpose(A)) + beta*C or
 *        alpha*conjugate(transpose(A))*B + conjugate(alpha)*conjugate(transpose(B))*A
 *        + beta*C.
 *
 * Reference:   http://www.netlib.org/blas/cher2k.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCher2kNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jint n, jint k, jobject alpha, jobject A, jint lda, jobject B, jint ldb, jfloat beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCher2k");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasCher2k");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasCher2k");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeB;
    cuComplex* nativeC;
    cuComplex complexAlpha;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeB = (cuComplex*)getPointer(env, B);
    nativeC = (cuComplex*)getPointer(env, C);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCher2k(%c, %c, %d, %d, [%f,%f], '%s', %d, '%s', %d, %f, '%s', %d)\n",
        uplo, trans, n, k, complexAlpha.x, complexAlpha.y, "A", lda, "B", ldb, beta, "C", ldc);

    cublasCher2k((char)uplo, (char)trans, n, k, complexAlpha, nativeA, lda, nativeB, ldb, beta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasCtrmm (char side, char uplo, char transa, char diag, int m, int n,
 *              cuComplex alpha, const cuComplex *A, int lda, const cuComplex *B,
 *              int ldb)
 *
 * performs one of the matrix-matrix operations
 *
 *   B = alpha * op(A) * B,  or  B = alpha * B * op(A)
 *
 * where alpha is a single-precision complex scalar, B is an m x n matrix composed
 * of single precision complex elements, and A is a unit or non-unit, upper or lower,
 * triangular matrix composed of single precision complex elements. op(A) is one of
 *
 *   op(A) = A  , op(A) = transpose(A) or op(A) = conjugate(transpose(A))
 *
 * Matrices A and B are stored in column major format, and lda and ldb are
 * the leading dimensions of the two-dimensonials arrays that contain A and
 * B, respectively.
 *
 * Input
 * -----
 * side   specifies whether op(A) multiplies B from the left or right.
 *        If side = 'L' or 'l', then B = alpha * op(A) * B. If side =
 *        'R' or 'r', then B = alpha * B * op(A).
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix. If uplo = 'U' or 'u', A is an upper triangular matrix.
 *        If uplo = 'L' or 'l', A is a lower triangular matrix.
 * transa specifies the form of op(A) to be used in the matrix
 *        multiplication. If transa = 'N' or 'n', then op(A) = A. If
 *        transa = 'T' or 't', then op(A) = transpose(A).
 *        If transa = 'C' or 'c', then op(A) = conjugate(transpose(A)).
 * diag   specifies whether or not A is unit triangular. If diag = 'U'
 *        or 'u', A is assumed to be unit triangular. If diag = 'N' or
 *        'n', A is not assumed to be unit triangular.
 * m      the number of rows of matrix B. m must be at least zero.
 * n      the number of columns of matrix B. n must be at least zero.
 * alpha  single precision complex scalar multiplier applied to op(A)*B, or
 *        B*op(A), respectively. If alpha is zero no accesses are made
 *        to matrix A, and no read accesses are made to matrix B.
 * A      single precision complex array of dimensions (lda, k). k = m if side =
 *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
 *        the leading k x k upper triangular part of the array A must
 *        contain the upper triangular matrix, and the strictly lower
 *        triangular part of A is not referenced. If uplo = 'L' or 'l'
 *        the leading k x k lower triangular part of the array A must
 *        contain the lower triangular matrix, and the strictly upper
 *        triangular part of A is not referenced. When diag = 'U' or 'u'
 *        the diagonal elements of A are no referenced and are assumed
 *        to be unity.
 * lda    leading dimension of A. When side = 'L' or 'l', it must be at
 *        least max(1,m) and at least max(1,n) otherwise
 * B      single precision complex array of dimensions (ldb, n). On entry, the
 *        leading m x n part of the array contains the matrix B. It is
 *        overwritten with the transformed matrix on exit.
 * ldb    leading dimension of B. It must be at least max (1, m).
 *
 * Output
 * ------
 * B      updated according to B = alpha * op(A) * B  or B = alpha * B * op(A)
 *
 * Reference: http://www.netlib.org/blas/ctrmm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCtrmmNative
    (JNIEnv *env, jclass cls, jchar side, jchar uplo, jchar transa, jchar diag, jint m, jint n, jobject alpha, jobject A, jint lda, jobject B, jint ldb)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCtrmm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasCtrmm");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeB;
    cuComplex complexAlpha;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeB = (cuComplex*)getPointer(env, B);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCtrmm(%c, %c, %c, %c, %d, %d, [%f,%f], '%s', %d, '%s', %d)\n",
        side, uplo, transa, diag, m, n, complexAlpha.x, complexAlpha.y, "A", lda, "B", ldb);

    cublasCtrmm((char)side, (char)uplo, (char)transa, (char)diag, m, n, complexAlpha, nativeA, lda, nativeB, ldb);
}




/**
 * <pre>
 * void
 * cublasCtrsm (char side, char uplo, char transa, char diag, int m, int n,
 *              cuComplex alpha, const cuComplex *A, int lda,
 *              cuComplex *B, int ldb)
 *
 * solves one of the matrix equations
 *
 *    op(A) * X = alpha * B,   or   X * op(A) = alpha * B,
 *
 * where alpha is a single precision complex scalar, and X and B are m x n matrices
 * that are composed of single precision complex elements. A is a unit or non-unit,
 * upper or lower triangular matrix, and op(A) is one of
 *
 *    op(A) = A  or  op(A) = transpose(A)  or  op( A ) = conj( A' ).
 *
 * The result matrix X overwrites input matrix B; that is, on exit the result
 * is stored in B. Matrices A and B are stored in column major format, and
 * lda and ldb are the leading dimensions of the two-dimensonials arrays that
 * contain A and B, respectively.
 *
 * Input
 * -----
 * side   specifies whether op(A) appears on the left or right of X as
 *        follows: side = 'L' or 'l' indicates solve op(A) * X = alpha * B.
 *        side = 'R' or 'r' indicates solve X * op(A) = alpha * B.
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix as follows: uplo = 'U' or 'u' indicates A is an upper
 *        triangular matrix. uplo = 'L' or 'l' indicates A is a lower
 *        triangular matrix.
 * transa specifies the form of op(A) to be used in matrix multiplication
 *        as follows: If transa = 'N' or 'N', then op(A) = A. If transa =
 *        'T', 't', 'C', or 'c', then op(A) = transpose(A).
 * diag   specifies whether or not A is a unit triangular matrix like so:
 *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
 *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
 * m      specifies the number of rows of B. m must be at least zero.
 * n      specifies the number of columns of B. n must be at least zero.
 * alpha  is a single precision complex scalar to be multiplied with B. When alpha is
 *        zero, then A is not referenced and B need not be set before entry.
 * A      is a single precision complex array of dimensions (lda, k), where k is
 *        m when side = 'L' or 'l', and is n when side = 'R' or 'r'. If
 *        uplo = 'U' or 'u', the leading k x k upper triangular part of
 *        the array A must contain the upper triangular matrix and the
 *        strictly lower triangular matrix of A is not referenced. When
 *        uplo = 'L' or 'l', the leading k x k lower triangular part of
 *        the array A must contain the lower triangular matrix and the
 *        strictly upper triangular part of A is not referenced. Note that
 *        when diag = 'U' or 'u', the diagonal elements of A are not
 *        referenced, and are assumed to be unity.
 * lda    is the leading dimension of the two dimensional array containing A.
 *        When side = 'L' or 'l' then lda must be at least max(1, m), when
 *        side = 'R' or 'r' then lda must be at least max(1, n).
 * B      is a single precision complex array of dimensions (ldb, n). ldb must be
 *        at least max (1,m). The leading m x n part of the array B must
 *        contain the right-hand side matrix B. On exit B is overwritten
 *        by the solution matrix X.
 * ldb    is the leading dimension of the two dimensional array containing B.
 *        ldb must be at least max(1, m).
 *
 * Output
 * ------
 * B      contains the solution matrix X satisfying op(A) * X = alpha * B,
 *        or X * op(A) = alpha * B
 *
 * Reference: http://www.netlib.org/blas/ctrsm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasCtrsmNative
    (JNIEnv *env, jclass cls, jchar side, jchar uplo, jchar transa, jchar diag, jint m, jint n, jobject alpha, jobject A, jint lda, jobject B, jint ldb)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasCtrsm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasCtrsm");
        return;
    }
    cuComplex* nativeA;
    cuComplex* nativeB;
    cuComplex complexAlpha;

    nativeA = (cuComplex*)getPointer(env, A);
    nativeB = (cuComplex*)getPointer(env, B);

    complexAlpha.x = env->GetFloatField(alpha, cuComplex_x);
    complexAlpha.y = env->GetFloatField(alpha, cuComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasCtrsm(%c, %c, %c, %c, %d, %d, [%f,%f], '%s', %d, '%s', %d)\n",
        side, uplo, transa, diag, m, n, complexAlpha.x, complexAlpha.y, "A", lda, "B", ldb);

    cublasCtrsm((char)side, (char)uplo, (char)transa, (char)diag, m, n, complexAlpha, nativeA, lda, nativeB, ldb);
}




/**
 * <pre>
 * double
 * cublasDasum (int n, const double *x, int incx)
 *
 * computes the sum of the absolute values of the elements of double
 * precision vector x; that is, the result is the sum from i = 0 to n - 1 of
 * abs(x[1 + i * incx]).
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      double-precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns the double-precision sum of absolute values
 * (0 if n <= 0 or incx <= 0, or if an error occurs)
 *
 * Reference: http://www.netlib.org/blas/dasum.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jdouble JNICALL Java_jcuda_jcublas_JCublas_cublasDasumNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDasum");
        return 0.0;
    }
    double* nativeX;

    nativeX = (double*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasDasum(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasDasum(n, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasDaxpy (int n, double alpha, const double *x, int incx, double *y,
 *              int incy)
 *
 * multiplies double-precision vector x by double-precision scalar alpha
 * and adds the result to double-precision vector y; that is, it overwrites
 * double-precision y with double-precision alpha * x + y. For i = 0 to n-1,
 * it replaces y[ly + i * incy] with alpha * x[lx + i * incx] + y[ly + i*incy],
 * where lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx; ly is defined in a
 * similar way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * alpha  double-precision scalar multiplier
 * x      double-precision vector with n elements
 * incx   storage spacing between elements of x
 * y      double-precision vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * y      double-precision result (unchanged if n <= 0)
 *
 * Reference: http://www.netlib.org/blas/daxpy.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library was not initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDaxpyNative
    (JNIEnv *env, jclass cls, jint n, jdouble alpha, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDaxpy");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasDaxpy");
        return;
    }
    double* nativeX;
    double* nativeY;

    nativeX = (double*)getPointer(env, x);
    nativeY = (double*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasDaxpy(%d, %lf, '%s', %d, '%s', %d)\n",
        n, alpha, "x", incx, "y", incy);

    cublasDaxpy(n, alpha, nativeX, incx, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasDcopy (int n, const double *x, int incx, double *y, int incy)
 *
 * copies the double-precision vector x to the double-precision vector y. For
 * i = 0 to n-1, copies x[lx + i * incx] to y[ly + i * incy], where lx = 1 if
 * incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a similar
 * way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      double-precision vector with n elements
 * incx   storage spacing between elements of x
 * y      double-precision vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * y      contains double precision vector x
 *
 * Reference: http://www.netlib.org/blas/dcopy.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDcopyNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDcopy");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasDcopy");
        return;
    }
    double* nativeX;
    double* nativeY;

    nativeX = (double*)getPointer(env, x);
    nativeY = (double*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasDcopy(%d, '%s', %d, '%s', %d)\n",
        n, "x", incx, "y", incy);

    cublasDcopy(n, nativeX, incx, nativeY, incy);
}




/**
 * <pre>
 * double
 * cublasDdot (int n, const double *x, int incx, const double *y, int incy)
 *
 * computes the dot product of two double-precision vectors. It returns the
 * dot product of the double precision vectors x and y if successful, and
 * 0.0f otherwise. It computes the sum for i = 0 to n - 1 of x[lx + i *
 * incx] * y[ly + i * incy], where lx = 1 if incx >= 0, else lx = 1 + (1 - n)
 * *incx, and ly is defined in a similar way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      double-precision vector with n elements
 * incx   storage spacing between elements of x
 * y      double-precision vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * returns double-precision dot product (zero if n <= 0)
 *
 * Reference: http://www.netlib.org/blas/ddot.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has nor been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to execute on GPU
 * </pre>
 */
JNIEXPORT jdouble JNICALL Java_jcuda_jcublas_JCublas_cublasDdotNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDdot");
        return 0.0;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasDdot");
        return 0.0;
    }
    double* nativeX;
    double* nativeY;

    nativeX = (double*)getPointer(env, x);
    nativeY = (double*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasDdot(%d, '%s', %d, '%s', %d)\n",
        n, "x", incx, "y", incy);

    return cublasDdot(n, nativeX, incx, nativeY, incy);
}




/**
 * <pre>
 * double
 * dnrm2 (int n, const double *x, int incx)
 *
 * computes the Euclidean norm of the double-precision n-vector x (with
 * storage increment incx). This code uses a multiphase model of
 * accumulation to avoid intermediate underflow and overflow.
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      double-precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns Euclidian norm (0 if n <= 0 or incx <= 0, or if an error occurs)
 *
 * Reference: http://www.netlib.org/blas/dnrm2.f
 * Reference: http://www.netlib.org/slatec/lin/dnrm2.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jdouble JNICALL Java_jcuda_jcublas_JCublas_cublasDnrm2Native
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDnrm2");
        return 0.0;
    }
    double* nativeX;

    nativeX = (double*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasDnrm2(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasDnrm2(n, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasDrot (int n, double *x, int incx, double *y, int incy, double sc,
 *             double ss)
 *
 * multiplies a 2x2 matrix ( sc ss) with the 2xn matrix ( transpose(x) )
 *                         (-ss sc)                     ( transpose(y) )
 *
 * The elements of x are in x[lx + i * incx], i = 0 ... n - 1, where lx = 1 if
 * incx >= 0, else lx = 1 + (1 - n) * incx, and similarly for y using ly and
 * incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      double-precision vector with n elements
 * incx   storage spacing between elements of x
 * y      double-precision vector with n elements
 * incy   storage spacing between elements of y
 * sc     element of rotation matrix
 * ss     element of rotation matrix
 *
 * Output
 * ------
 * x      rotated vector x (unchanged if n <= 0)
 * y      rotated vector y (unchanged if n <= 0)
 *
 * Reference  http://www.netlib.org/blas/drot.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDrotNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy, jdouble sc, jdouble ss)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDrot");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasDrot");
        return;
    }
    double* nativeX;
    double* nativeY;

    nativeX = (double*)getPointer(env, x);
    nativeY = (double*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasDrot(%d, '%s', %d, '%s', %d, %lf, %lf)\n",
        n, "x", incx, "y", incy, sc, ss);

    cublasDrot(n, nativeX, incx, nativeY, incy, sc, ss);
}




/**
 * <pre>
 * void
 * cublasDrotg (double *host_sa, double *host_sb, double *host_sc, double *host_ss)
 *
 * constructs the Givens tranformation
 *
 *        ( sc  ss )
 *    G = (        ) ,  sc^2 + ss^2 = 1,
 *        (-ss  sc )
 *
 * which zeros the second entry of the 2-vector transpose(sa, sb).
 *
 * The quantity r = (+/-) sqrt (sa^2 + sb^2) overwrites sa in storage. The
 * value of sb is overwritten by a value z which allows sc and ss to be
 * recovered by the following algorithm:
 *
 *    if z=1          set sc = 0.0 and ss = 1.0
 *    if abs(z) < 1   set sc = sqrt(1-z^2) and ss = z
 *    if abs(z) > 1   set sc = 1/z and ss = sqrt(1-sc^2)
 *
 * The function drot (n, x, incx, y, incy, sc, ss) normally is called next
 * to apply the transformation to a 2 x n matrix.
 * Note that is function is provided for completeness and run exclusively
 * on the Host.
 *
 * Input
 * -----
 * sa     double-precision scalar
 * sb     double-precision scalar
 *
 * Output
 * ------
 * sa     double-precision r
 * sb     double-precision z
 * sc     double-precision result
 * ss     double-precision result
 *
 * Reference: http://www.netlib.org/blas/drotg.f
 *
 * This function does not set any error status.
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDrotgNative
    (JNIEnv *env, jclass cls, jobject host_sa, jobject host_sb, jobject host_sc, jobject host_ss)
{
    if (host_sa == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'host_sa' is null for cublasDrotg");
        return;
    }
    if (host_sb == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'host_sb' is null for cublasDrotg");
        return;
    }
    if (host_sc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'host_sc' is null for cublasDrotg");
        return;
    }
    if (host_ss == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'host_ss' is null for cublasDrotg");
        return;
    }
    double* nativeHOST_SA;
    double* nativeHOST_SB;
    double* nativeHOST_SC;
    double* nativeHOST_SS;

    nativeHOST_SA = (double*)getPointer(env, host_sa);
    nativeHOST_SB = (double*)getPointer(env, host_sb);
    nativeHOST_SC = (double*)getPointer(env, host_sc);
    nativeHOST_SS = (double*)getPointer(env, host_ss);

    Logger::log(LOG_TRACE, "Executing cublasDrotg('%s', '%s', '%s', '%s')\n",
        "host_sa", "host_sb", "host_sc", "host_ss");

    cublasDrotg(nativeHOST_SA, nativeHOST_SB, nativeHOST_SC, nativeHOST_SS);
}




/**
 * <pre>
 * void
 * cublasDscal (int n, double alpha, double *x, int incx)
 *
 * replaces double-precision vector x with double-precision alpha * x. For
 * i = 0 to n-1, it replaces x[lx + i * incx] with alpha * x[lx + i * incx],
 * where lx = 1 if incx >= 0, else lx = 1 + (1 - n) * incx.
 *
 * Input
 * -----
 * n      number of elements in input vector
 * alpha  double-precision scalar multiplier
 * x      double-precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * x      double-precision result (unchanged if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/dscal.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library was not initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDscalNative
    (JNIEnv *env, jclass cls, jint n, jdouble alpha, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDscal");
        return;
    }
    double* nativeX;

    nativeX = (double*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasDscal(%d, %lf, '%s', %d)\n",
        n, alpha, "x", incx);

    cublasDscal(n, alpha, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasDswap (int n, double *x, int incx, double *y, int incy)
 *
 * replaces double-precision vector x with double-precision alpha * x. For i
 * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx],
 * where ix = 1 if incx >= 0, else ix = 1 + (1 - n) * incx.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * alpha  double-precision scalar multiplier
 * x      double-precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * x      double precision result (unchanged if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/dswap.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDswapNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx, jobject y, jint incy)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDswap");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasDswap");
        return;
    }
    double* nativeX;
    double* nativeY;

    nativeX = (double*)getPointer(env, x);
    nativeY = (double*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasDswap(%d, '%s', %d, '%s', %d)\n",
        n, "x", incx, "y", incy);

    cublasDswap(n, nativeX, incx, nativeY, incy);
}




/**
 * <pre>
 * int
 * idamax (int n, const double *x, int incx)
 *
 * finds the smallest index of the maximum magnitude element of double-
 * precision vector x; that is, the result is the first i, i = 0 to n - 1,
 * that maximizes abs(x[1 + i * incx])).
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      double-precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns the smallest index (0 if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/idamax.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasIdamaxNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasIdamax");
        return 0;
    }
    double* nativeX;

    nativeX = (double*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasIdamax(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasIdamax(n, nativeX, incx);
}




/**
 * <pre>
 * int
 * idamin (int n, const double *x, int incx)
 *
 * finds the smallest index of the minimum magnitude element of double-
 * precision vector x; that is, the result is the first i, i = 0 to n - 1,
 * that minimizes abs(x[1 + i * incx])).
 *
 * Input
 * -----
 * n      number of elements in input vector
 * x      double-precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * returns the smallest index (0 if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/scilib/blass.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT jint JNICALL Java_jcuda_jcublas_JCublas_cublasIdaminNative
    (JNIEnv *env, jclass cls, jint n, jobject x, jint incx)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasIdamin");
        return 0;
    }
    double* nativeX;

    nativeX = (double*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasIdamin(%d, '%s', %d)\n",
        n, "x", incx);

    return cublasIdamin(n, nativeX, incx);
}




/**
 * <pre>
 * cublasDgemv (char trans, int m, int n, double alpha, const double *A,
 *              int lda, const double *x, int incx, double beta, double *y,
 *              int incy)
 *
 * performs one of the matrix-vector operations
 *
 *    y = alpha * op(A) * x + beta * y,
 *
 * where op(A) is one of
 *
 *    op(A) = A   or   op(A) = transpose(A)
 *
 * where alpha and beta are double precision scalars, x and y are double
 * precision vectors, and A is an m x n matrix consisting of double precision
 * elements. Matrix A is stored in column major format, and lda is the leading
 * dimension of the two-dimensional array in which A is stored.
 *
 * Input
 * -----
 * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If trans =
 *        trans = 't', 'T', 'c', or 'C', op(A) = transpose(A)
 * m      specifies the number of rows of the matrix A. m must be at least
 *        zero.
 * n      specifies the number of columns of the matrix A. n must be at least
 *        zero.
 * alpha  double precision scalar multiplier applied to op(A).
 * A      double precision array of dimensions (lda, n) if trans = 'n' or
 *        'N'), and of dimensions (lda, m) otherwise. lda must be at least
 *        max(1, m) and at least max(1, n) otherwise.
 * lda    leading dimension of two-dimensional array used to store matrix A
 * x      double precision array of length at least (1 + (n - 1) * abs(incx))
 *        when trans = 'N' or 'n' and at least (1 + (m - 1) * abs(incx))
 *        otherwise.
 * incx   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 * beta   double precision scalar multiplier applied to vector y. If beta
 *        is zero, y is not read.
 * y      double precision array of length at least (1 + (m - 1) * abs(incy))
 *        when trans = 'N' or 'n' and at least (1 + (n - 1) * abs(incy))
 *        otherwise.
 * incy   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 *
 * Output
 * ------
 * y      updated according to alpha * op(A) * x + beta * y
 *
 * Reference: http://www.netlib.org/blas/dgemv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0, or if incx or incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDgemvNative
    (JNIEnv *env, jclass cls, jchar trans, jint m, jint n, jdouble alpha, jobject A, jint lda, jobject x, jint incx, jdouble beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDgemv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDgemv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasDgemv");
        return;
    }
    double* nativeA;
    double* nativeX;
    double* nativeY;

    nativeA = (double*)getPointer(env, A);
    nativeX = (double*)getPointer(env, x);
    nativeY = (double*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasDgemv(%c, %d, %d, %lf, '%s', %d, '%s', %d, %lf, '%s', %d)\n",
        trans, m, n, alpha, "A", lda, "x", incx, beta, "y", incy);

    cublasDgemv((char)trans, m, n, alpha, nativeA, lda, nativeX, incx, beta, nativeY, incy);
}




/**
 * <pre>
 * cublasDger (int m, int n, double alpha, const double *x, int incx,
 *             const double *y, int incy, double *A, int lda)
 *
 * performs the symmetric rank 1 operation
 *
 *    A = alpha * x * transpose(y) + A,
 *
 * where alpha is a double precision scalar, x is an m element double
 * precision vector, y is an n element double precision vector, and A
 * is an m by n matrix consisting of double precision elements. Matrix A
 * is stored in column major format, and lda is the leading dimension of
 * the two-dimensional array used to store A.
 *
 * Input
 * -----
 * m      specifies the number of rows of the matrix A. It must be at least
 *        zero.
 * n      specifies the number of columns of the matrix A. It must be at
 *        least zero.
 * alpha  double precision scalar multiplier applied to x * transpose(y)
 * x      double precision array of length at least (1 + (m - 1) * abs(incx))
 * incx   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 * y      double precision array of length at least (1 + (n - 1) * abs(incy))
 * incy   specifies the storage spacing between elements of y. incy must not
 *        be zero.
 * A      double precision array of dimensions (lda, n).
 * lda    leading dimension of two-dimensional array used to store matrix A
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * transpose(y) + A
 *
 * Reference: http://www.netlib.org/blas/dger.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDgerNative
    (JNIEnv *env, jclass cls, jint m, jint n, jdouble alpha, jobject x, jint incx, jobject y, jint incy, jobject A, jint lda)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDger");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasDger");
        return;
    }
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDger");
        return;
    }
    double* nativeX;
    double* nativeY;
    double* nativeA;

    nativeX = (double*)getPointer(env, x);
    nativeY = (double*)getPointer(env, y);
    nativeA = (double*)getPointer(env, A);

    Logger::log(LOG_TRACE, "Executing cublasDger(%d, %d, %lf, '%s', %d, '%s', %d, '%s', %d)\n",
        m, n, alpha, "x", incx, "y", incy, "A", lda);

    cublasDger(m, n, alpha, nativeX, incx, nativeY, incy, nativeA, lda);
}




/**
 * <pre>
 * void
 * cublasDsyr (char uplo, int n, double alpha, const double *x, int incx,
 *             double *A, int lda)
 *
 * performs the symmetric rank 1 operation
 *
 *    A = alpha * x * transpose(x) + A,
 *
 * where alpha is a double precision scalar, x is an n element double
 * precision vector and A is an n x n symmetric matrix consisting of
 * double precision elements. Matrix A is stored in column major format,
 * and lda is the leading dimension of the two-dimensional array
 * containing A.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or
 *        the lower triangular part of array A. If uplo = 'U' or 'u',
 *        then only the upper triangular part of A may be referenced.
 *        If uplo = 'L' or 'l', then only the lower triangular part of
 *        A may be referenced.
 * n      specifies the number of rows and columns of the matrix A. It
 *        must be at least 0.
 * alpha  double precision scalar multiplier applied to x * transpose(x)
 * x      double precision array of length at least (1 + (n - 1) * abs(incx))
 * incx   specifies the storage spacing between elements of x. incx must
 *        not be zero.
 * A      double precision array of dimensions (lda, n). If uplo = 'U' or
 *        'u', then A must contain the upper triangular part of a symmetric
 *        matrix, and the strictly lower triangular part is not referenced.
 *        If uplo = 'L' or 'l', then A contains the lower triangular part
 *        of a symmetric matrix, and the strictly upper triangular part is
 *        not referenced.
 * lda    leading dimension of the two-dimensional array containing A. lda
 *        must be at least max(1, n).
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * transpose(x) + A
 *
 * Reference: http://www.netlib.org/blas/dsyr.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDsyrNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jdouble alpha, jobject x, jint incx, jobject A, jint lda)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDsyr");
        return;
    }
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDsyr");
        return;
    }
    double* nativeX;
    double* nativeA;

    nativeX = (double*)getPointer(env, x);
    nativeA = (double*)getPointer(env, A);

    Logger::log(LOG_TRACE, "Executing cublasDsyr(%c, %d, %lf, '%s', %d, '%s', %d)\n",
        uplo, n, alpha, "x", incx, "A", lda);

    cublasDsyr((char)uplo, n, alpha, nativeX, incx, nativeA, lda);
}




/**
 * <pre>
 * void cublasDsyr2 (char uplo, int n, double alpha, const double *x, int incx,
 *                   const double *y, int incy, double *A, int lda)
 *
 * performs the symmetric rank 2 operation
 *
 *    A = alpha*x*transpose(y) + alpha*y*transpose(x) + A,
 *
 * where alpha is a double precision scalar, x and y are n element double
 * precision vector and A is an n by n symmetric matrix consisting of double
 * precision elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array A. If uplo == 'U' or 'u', then only the
 *        upper triangular part of A may be referenced and the lower triangular
 *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
 *        triangular part of A may be referenced and the upper triangular part
 *        of A is inferred.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  double precision scalar multiplier applied to x * transpose(y) +
 *        y * transpose(x).
 * x      double precision array of length at least (1 + (n - 1) * abs (incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * y      double precision array of length at least (1 + (n - 1) * abs (incy)).
 * incy   storage spacing between elements of y. incy must not be zero.
 * A      double precision array of dimensions (lda, n). If uplo == 'U' or 'u',
 *        then A must contains the upper triangular part of a symmetric matrix,
 *        and the strictly lower triangular parts is not referenced. If uplo ==
 *        'L' or 'l', then A contains the lower triangular part of a symmetric
 *        matrix, and the strictly upper triangular part is not referenced.
 * lda    leading dimension of A. It must be at least max(1, n).
 *
 * Output
 * ------
 * A      updated according to A = alpha*x*transpose(y)+alpha*y*transpose(x)+A
 *
 * Reference: http://www.netlib.org/blas/dsyr2.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDsyr2Native
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jdouble alpha, jobject x, jint incx, jobject y, jint incy, jobject A, jint lda)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDsyr2");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasDsyr2");
        return;
    }
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDsyr2");
        return;
    }
    double* nativeX;
    double* nativeY;
    double* nativeA;

    nativeX = (double*)getPointer(env, x);
    nativeY = (double*)getPointer(env, y);
    nativeA = (double*)getPointer(env, A);

    Logger::log(LOG_TRACE, "Executing cublasDsyr2(%c, %d, %lf, '%s', %d, '%s', %d, '%s', %d)\n",
        uplo, n, alpha, "x", incx, "y", incy, "A", lda);

    cublasDsyr2((char)uplo, n, alpha, nativeX, incx, nativeY, incy, nativeA, lda);
}




/**
 * <pre>
 * void
 * cublasDspr (char uplo, int n, double alpha, const double *x, int incx,
 *             double *AP)
 *
 * performs the symmetric rank 1 operation
 *
 *    A = alpha * x * transpose(x) + A,
 *
 * where alpha is a double precision scalar and x is an n element double
 * precision vector. A is a symmetric n x n matrix consisting of double
 * precision elements that is supplied in packed form.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array AP. If uplo == 'U' or 'u', then the upper
 *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then
 *        the lower triangular part of A is supplied in AP.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  double precision scalar multiplier applied to x * transpose(x).
 * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * AP     double precision array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * transpose(x) + A
 *
 * Reference: http://www.netlib.org/blas/dspr.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDsprNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jdouble alpha, jobject x, jint incx, jobject AP)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDspr");
        return;
    }
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasDspr");
        return;
    }
    double* nativeX;
    double* nativeAP;

    nativeX = (double*)getPointer(env, x);
    nativeAP = (double*)getPointer(env, AP);

    Logger::log(LOG_TRACE, "Executing cublasDspr(%c, %d, %lf, '%s', %d, '%s')\n",
        uplo, n, alpha, "x", incx, "AP");

    cublasDspr((char)uplo, n, alpha, nativeX, incx, nativeAP);
}




/**
 * <pre>
 * void
 * cublasDspr2 (char uplo, int n, double alpha, const double *x, int incx,
 *              const double *y, int incy, double *AP)
 *
 * performs the symmetric rank 2 operation
 *
 *    A = alpha*x*transpose(y) + alpha*y*transpose(x) + A,
 *
 * where alpha is a double precision scalar, and x and y are n element double
 * precision vectors. A is a symmetric n x n matrix consisting of double
 * precision elements that is supplied in packed form.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array A. If uplo == 'U' or 'u', then only the
 *        upper triangular part of A may be referenced and the lower triangular
 *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
 *        triangular part of A may be referenced and the upper triangular part
 *        of A is inferred.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  double precision scalar multiplier applied to x * transpose(y) +
 *        y * transpose(x).
 * x      double precision array of length at least (1 + (n - 1) * abs (incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * y      double precision array of length at least (1 + (n - 1) * abs (incy)).
 * incy   storage spacing between elements of y. incy must not be zero.
 * AP     double precision array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 *
 * Output
 * ------
 * A      updated according to A = alpha*x*transpose(y)+alpha*y*transpose(x)+A
 *
 * Reference: http://www.netlib.org/blas/dspr2.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDspr2Native
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jdouble alpha, jobject x, jint incx, jobject y, jint incy, jobject AP)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDspr2");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasDspr2");
        return;
    }
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasDspr2");
        return;
    }
    double* nativeX;
    double* nativeY;
    double* nativeAP;

    nativeX = (double*)getPointer(env, x);
    nativeY = (double*)getPointer(env, y);
    nativeAP = (double*)getPointer(env, AP);

    Logger::log(LOG_TRACE, "Executing cublasDspr2(%c, %d, %lf, '%s', %d, '%s', %d, '%s')\n",
        uplo, n, alpha, "x", incx, "y", incy, "AP");

    cublasDspr2((char)uplo, n, alpha, nativeX, incx, nativeY, incy, nativeAP);
}




/**
 * <pre>
 * void
 * cublasDtrsv (char uplo, char trans, char diag, int n, const double *A,
 *              int lda, double *x, int incx)
 *
 * solves a system of equations op(A) * x = b, where op(A) is either A or
 * transpose(A). b and x are double precision vectors consisting of n
 * elements, and A is an n x n matrix composed of a unit or non-unit, upper
 * or lower triangular matrix. Matrix A is stored in column major format,
 * and lda is the leading dimension of the two-dimensional array containing
 * A.
 *
 * No test for singularity or near-singularity is included in this function.
 * Such tests must be performed before calling this function.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the
 *        lower triangular part of array A. If uplo = 'U' or 'u', then only
 *        the upper triangular part of A may be referenced. If uplo = 'L' or
 *        'l', then only the lower triangular part of A may be referenced.
 * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If transa = 't',
 *        'T', 'c', or 'C', op(A) = transpose(A)
 * diag   specifies whether or not A is a unit triangular matrix like so:
 *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
 *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. It
 *        must be at least 0.
 * A      is a double precision array of dimensions (lda, n). If uplo = 'U'
 *        or 'u', then A must contains the upper triangular part of a symmetric
 *        matrix, and the strictly lower triangular parts is not referenced.
 *        If uplo = 'L' or 'l', then A contains the lower triangular part of
 *        a symmetric matrix, and the strictly upper triangular part is not
 *        referenced.
 * lda    is the leading dimension of the two-dimensional array containing A.
 *        lda must be at least max(1, n).
 * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the n element right-hand side vector b. On exit,
 *        it is overwritten with the solution vector x.
 * incx   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 *
 * Output
 * ------
 * x      updated to contain the solution vector x that solves op(A) * x = b.
 *
 * Reference: http://www.netlib.org/blas/dtrsv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDtrsvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDtrsv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDtrsv");
        return;
    }
    double* nativeA;
    double* nativeX;

    nativeA = (double*)getPointer(env, A);
    nativeX = (double*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasDtrsv(%c, %c, %c, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, "A", lda, "x", incx);

    cublasDtrsv((char)uplo, (char)trans, (char)diag, n, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasDtrmv (char uplo, char trans, char diag, int n, const double *A,
 *              int lda, double *x, int incx);
 *
 * performs one of the matrix-vector operations x = op(A) * x, where op(A) =
 = A, or op(A) = transpose(A). x is an n-element single precision vector, and
 * A is an n x n, unit or non-unit, upper or lower, triangular matrix composed
 * of single precision elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix. If uplo = 'U' or 'u', then A is an upper triangular matrix.
 *        If uplo = 'L' or 'l', then A is a lower triangular matrix.
 * trans  specifies op(A). If transa = 'N' or 'n', op(A) = A. If trans = 'T',
 *        't', 'C', or 'c', op(A) = transpose(A)
 * diag   specifies whether or not matrix A is unit triangular. If diag = 'U'
 *        or 'u', A is assumed to be unit triangular. If diag = 'N' or 'n', A
 *        is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * A      single precision array of dimension (lda, n). If uplo = 'U' or 'u',
 *        the leading n x n upper triangular part of the array A must contain
 *        the upper triangular matrix and the strictly lower triangular part
 *        of A is not referenced. If uplo = 'L' or 'l', the leading n x n lower
 *        triangular part of the array A must contain the lower triangular
 *        matrix and the strictly upper triangular part of A is not referenced.
 *        When diag = 'U' or 'u', the diagonal elements of A are not referenced
 *        either, but are are assumed to be unity.
 * lda    is the leading dimension of A. It must be at least max (1, n).
 * x      single precision array of length at least (1 + (n - 1) * abs(incx) ).
 *        On entry, x contains the source vector. On exit, x is overwritten
 *        with the result vector.
 * incx   specifies the storage spacing for elements of x. incx must not be
 *        zero.
 *
 * Output
 * ------
 * x      updated according to x = op(A) * x,
 *
 * Reference: http://www.netlib.org/blas/dtrmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDtrmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDtrmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDtrmv");
        return;
    }
    double* nativeA;
    double* nativeX;

    nativeA = (double*)getPointer(env, A);
    nativeX = (double*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasDtrmv(%c, %c, %c, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, "A", lda, "x", incx);

    cublasDtrmv((char)uplo, (char)trans, (char)diag, n, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasDgbmv (char trans, int m, int n, int kl, int ku, double alpha,
 *              const double *A, int lda, const double *x, int incx, double beta,
 *              double *y, int incy);
 *
 * performs one of the matrix-vector operations
 *
 *    y = alpha*op(A)*x + beta*y,  op(A)=A or op(A) = transpose(A)
 *
 * alpha and beta are double precision scalars. x and y are double precision
 * vectors. A is an m by n band matrix consisting of double precision elements
 * with kl sub-diagonals and ku super-diagonals.
 *
 * Input
 * -----
 * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
 *        't', 'C', or 'c', op(A) = transpose(A)
 * m      specifies the number of rows of the matrix A. m must be at least
 *        zero.
 * n      specifies the number of columns of the matrix A. n must be at least
 *        zero.
 * kl     specifies the number of sub-diagonals of matrix A. It must be at
 *        least zero.
 * ku     specifies the number of super-diagonals of matrix A. It must be at
 *        least zero.
 * alpha  double precision scalar multiplier applied to op(A).
 * A      double precision array of dimensions (lda, n). The leading
 *        (kl + ku + 1) x n part of the array A must contain the band matrix A,
 *        supplied column by column, with the leading diagonal of the matrix
 *        in row (ku + 1) of the array, the first super-diagonal starting at
 *        position 2 in row ku, the first sub-diagonal starting at position 1
 *        in row (ku + 2), and so on. Elements in the array A that do not
 *        correspond to elements in the band matrix (such as the top left
 *        ku x ku triangle) are not referenced.
 * lda    leading dimension of A. lda must be at least (kl + ku + 1).
 * x      double precision array of length at least (1+(n-1)*abs(incx)) when
 *        trans == 'N' or 'n' and at least (1+(m-1)*abs(incx)) otherwise.
 * incx   specifies the increment for the elements of x. incx must not be zero.
 * beta   double precision scalar multiplier applied to vector y. If beta is
 *        zero, y is not read.
 * y      double precision array of length at least (1+(m-1)*abs(incy)) when
 *        trans == 'N' or 'n' and at least (1+(n-1)*abs(incy)) otherwise. If
 *        beta is zero, y is not read.
 * incy   On entry, incy specifies the increment for the elements of y. incy
 *        must not be zero.
 *
 * Output
 * ------
 * y      updated according to y = alpha*op(A)*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/dgbmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDgbmvNative
    (JNIEnv *env, jclass cls, jchar trans, jint m, jint n, jint kl, jint ku, jdouble alpha, jobject A, jint lda, jobject x, jint incx, jdouble beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDgbmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDgbmv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasDgbmv");
        return;
    }
    double* nativeA;
    double* nativeX;
    double* nativeY;

    nativeA = (double*)getPointer(env, A);
    nativeX = (double*)getPointer(env, x);
    nativeY = (double*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasDgbmv(%c, %d, %d, %d, %d, %lf, '%s', %d, '%s', %d, %lf, '%s', %d)\n",
        trans, m, n, kl, ku, alpha, "A", lda, "x", incx, beta, "y", incy);

    cublasDgbmv((char)trans, m, n, kl, ku, alpha, nativeA, lda, nativeX, incx, beta, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasDtbmv (char uplo, char trans, char diag, int n, int k, const double *A,
 *              int lda, double *x, int incx)
 *
 * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
 * or op(A) = transpose(A). x is an n-element double precision vector, and A is
 * an n x n, unit or non-unit, upper or lower triangular band matrix composed
 * of double precision elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix A is an upper or lower triangular band
 *        matrix. If uplo == 'U' or 'u', A is an upper triangular band matrix.
 *        If uplo == 'L' or 'l', A is a lower triangular band matrix.
 * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
 *        't', 'C', or 'c', op(A) = transpose(A)
 * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
 *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
 *        is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
 *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
 *        'l', k specifies the number of sub-diagonals. k must at least be
 *        zero.
 * A      double precision array of dimension (lda, n). If uplo == 'U' or 'u',
 *        the leading (k + 1) x n part of the array A must contain the upper
 *        triangular band matrix, supplied column by column, with the leading
 *        diagonal of the matrix in row (k + 1) of the array, the first
 *        super-diagonal starting at position 2 in row k, and so on. The top
 *        left k x k triangle of the array A is not referenced. If uplo == 'L'
 *        or 'l', the leading (k + 1) x n part of the array A must constain the
 *        lower triangular band matrix, supplied column by column, with the
 *        leading diagonal of the matrix in row 1 of the array, the first
 *        sub-diagonal startingat position 1 in row 2, and so on. The bottom
 *        right k x k triangle of the array is not referenced.
 * lda    is the leading dimension of A. It must be at least (k + 1).
 * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the source vector. On exit, x is overwritten
 *        with the result vector.
 * incx   specifies the storage spacing for elements of x. incx must not be
 *        zero.
 *
 * Output
 * ------
 * x      updated according to x = op(A) * x
 *
 * Reference: http://www.netlib.org/blas/dtbmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n or k < 0, or if incx == 0
 * CUBLAS_STATUS_ALLOC_FAILED     if function cannot allocate enough internal scratch vector memory
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDtbmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jint k, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDtbmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDtbmv");
        return;
    }
    double* nativeA;
    double* nativeX;

    nativeA = (double*)getPointer(env, A);
    nativeX = (double*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasDtbmv(%c, %c, %c, %d, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, k, "A", lda, "x", incx);

    cublasDtbmv((char)uplo, (char)trans, (char)diag, n, k, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasDtpmv (char uplo, char trans, char diag, int n, const double *AP,
 *              double *x, int incx);
 *
 * performs one of the matrix-vector operations x = op(A) * x, where op(A) = A,
 * or op(A) = transpose(A). x is an n element double precision vector, and A
 * is an n x n, unit or non-unit, upper or lower triangular matrix composed
 * of double precision elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix. If uplo == 'U' or 'u', then A is an upper triangular matrix.
 *        If uplo == 'L' or 'l', then A is a lower triangular matrix.
 * trans  specifies op(A). If transa == 'N' or 'n', op(A) = A. If trans == 'T',
 *        't', 'C', or 'c', op(A) = transpose(A)
 * diag   specifies whether or not matrix A is unit triangular. If diag == 'U'
 *        or 'u', A is assumed to be unit triangular. If diag == 'N' or 'n', A
 *        is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero. In the current implementation n must not exceed 4070.
 * AP     double precision array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i <= j, then A[i,j] is stored in AP[i+(j*(j+1)/2)]. If
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the source vector. On exit, x is overwritten
 *        with the result vector.
 * incx   specifies the storage spacing for elements of x. incx must not be
 *        zero.
 *
 * Output
 * ------
 * x      updated according to x = op(A) * x,
 *
 * Reference: http://www.netlib.org/blas/dtpmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or n < 0
 * CUBLAS_STATUS_ALLOC_FAILED     if function cannot allocate enough internal scratch vector memory
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDtpmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject AP, jobject x, jint incx)
{
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasDtpmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDtpmv");
        return;
    }
    double* nativeAP;
    double* nativeX;

    nativeAP = (double*)getPointer(env, AP);
    nativeX = (double*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasDtpmv(%c, %c, %c, %d, '%s', '%s', %d)\n",
        uplo, trans, diag, n, "AP", "x", incx);

    cublasDtpmv((char)uplo, (char)trans, (char)diag, n, nativeAP, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasDtpsv (char uplo, char trans, char diag, int n, const double *AP,
 *              double *X, int incx)
 *
 * solves one of the systems of equations op(A)*x = b, where op(A) is either
 * op(A) = A or op(A) = transpose(A). b and x are n element vectors, and A is
 * an n x n unit or non-unit, upper or lower triangular matrix. No test for
 * singularity or near-singularity is included in this routine. Such tests
 * must be performed before calling this routine.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix is an upper or lower triangular matrix
 *        as follows: If uplo == 'U' or 'u', A is an upper triangluar matrix.
 *        If uplo == 'L' or 'l', A is a lower triangular matrix.
 * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
 *        't', 'C', or 'c', op(A) = transpose(A).
 * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
 *        assumed to be unit triangular; thas is, diagonal elements are not
 *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
 *        assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * AP     double precision array with at least ((n*(n+1))/2) elements. If uplo
 *        == 'U' or 'u', the array AP contains the upper triangular matrix A,
 *        packed sequentially, column by column; that is, if i <= j, then
 *        A[i,j] is stored is AP[i+(j*(j+1)/2)]. If uplo == 'L' or 'L', the
 *        array AP contains the lower triangular matrix A, packed sequentially,
 *        column by column; that is, if i >= j, then A[i,j] is stored in
 *        AP[i+((2*n-j+1)*j)/2]. When diag = 'U' or 'u', the diagonal elements
 *        of A are not referenced and are assumed to be unity.
 * x      double precision array of length at least (1+(n-1)*abs(incx)).
 * incx   storage spacing between elements of x. It must not be zero.
 *
 * Output
 * ------
 * x      updated to contain the solution vector x that solves op(A) * x = b.
 *
 * Reference: http://www.netlib.org/blas/dtpsv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0 or n > 2035
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDtpsvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject AP, jobject x, jint incx)
{
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasDtpsv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDtpsv");
        return;
    }
    double* nativeAP;
    double* nativeX;

    nativeAP = (double*)getPointer(env, AP);
    nativeX = (double*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasDtpsv(%c, %c, %c, %d, '%s', '%s', %d)\n",
        uplo, trans, diag, n, "AP", "x", incx);

    cublasDtpsv((char)uplo, (char)trans, (char)diag, n, nativeAP, nativeX, incx);
}




/**
 * <pre>
 * void cublasDtbsv (char uplo, char trans, char diag, int n, int k,
 *                   const double *A, int lda, double *X, int incx)
 *
 * solves one of the systems of equations op(A)*x = b, where op(A) is either
 * op(A) = A or op(A) = transpose(A). b and x are n element vectors, and A is
 * an n x n unit or non-unit, upper or lower triangular band matrix with k + 1
 * diagonals. No test for singularity or near-singularity is included in this
 * function. Such tests must be performed before calling this function.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix is an upper or lower triangular band
 *        matrix as follows: If uplo == 'U' or 'u', A is an upper triangular
 *        band matrix. If uplo == 'L' or 'l', A is a lower triangular band
 *        matrix.
 * trans  specifies op(A). If trans == 'N' or 'n', op(A) = A. If trans == 'T',
 *        't', 'C', or 'c', op(A) = transpose(A).
 * diag   specifies whether A is unit triangular. If diag == 'U' or 'u', A is
 *        assumed to be unit triangular; thas is, diagonal elements are not
 *        read and are assumed to be unity. If diag == 'N' or 'n', A is not
 *        assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. n must be
 *        at least zero.
 * k      specifies the number of super- or sub-diagonals. If uplo == 'U' or
 *        'u', k specifies the number of super-diagonals. If uplo == 'L' or
 *        'l', k specifies the number of sub-diagonals. k must at least be
 *        zero.
 * A      double precision array of dimension (lda, n). If uplo == 'U' or 'u',
 *        the leading (k + 1) x n part of the array A must contain the upper
 *        triangular band matrix, supplied column by column, with the leading
 *        diagonal of the matrix in row (k + 1) of the array, the first super-
 *        diagonal starting at position 2 in row k, and so on. The top left
 *        k x k triangle of the array A is not referenced. If uplo == 'L' or
 *        'l', the leading (k + 1) x n part of the array A must constain the
 *        lower triangular band matrix, supplied column by column, with the
 *        leading diagonal of the matrix in row 1 of the array, the first
 *        sub-diagonal starting at position 1 in row 2, and so on. The bottom
 *        right k x k triangle of the array is not referenced.
 * x      double precision array of length at least (1+(n-1)*abs(incx)).
 * incx   storage spacing between elements of x. It must not be zero.
 *
 * Output
 * ------
 * x      updated to contain the solution vector x that solves op(A) * x = b.
 *
 * Reference: http://www.netlib.org/blas/dtbsv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0, n < 0 or n > 2035
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDtbsvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jint k, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDtbsv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDtbsv");
        return;
    }
    double* nativeA;
    double* nativeX;

    nativeA = (double*)getPointer(env, A);
    nativeX = (double*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasDtbsv(%c, %c, %c, %d, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, k, "A", lda, "x", incx);

    cublasDtbsv((char)uplo, (char)trans, (char)diag, n, k, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasDsymv (char uplo, int n, double alpha, const double *A, int lda,
 *              const double *x, int incx, double beta, double *y, int incy)
 *
 * performs the matrix-vector operation
 *
 *     y = alpha*A*x + beta*y
 *
 * Alpha and beta are double precision scalars, and x and y are double
 * precision vectors, each with n elements. A is a symmetric n x n matrix
 * consisting of double precision elements that is stored in either upper or
 * lower storage mode.
 *
 * Input
 * -----
 * uplo   specifies whether the upper or lower triangular part of the array A
 *        is to be referenced. If uplo == 'U' or 'u', the symmetric matrix A
 *        is stored in upper storage mode, i.e. only the upper triangular part
 *        of A is to be referenced while the lower triangular part of A is to
 *        be inferred. If uplo == 'L' or 'l', the symmetric matrix A is stored
 *        in lower storage mode, i.e. only the lower triangular part of A is
 *        to be referenced while the upper triangular part of A is to be
 *        inferred.
 * n      specifies the number of rows and the number of columns of the
 *        symmetric matrix A. n must be at least zero.
 * alpha  double precision scalar multiplier applied to A*x.
 * A      double precision array of dimensions (lda, n). If uplo == 'U' or 'u',
 *        the leading n x n upper triangular part of the array A must contain
 *        the upper triangular part of the symmetric matrix and the strictly
 *        lower triangular part of A is not referenced. If uplo == 'L' or 'l',
 *        the leading n x n lower triangular part of the array A must contain
 *        the lower triangular part of the symmetric matrix and the strictly
 *        upper triangular part of A is not referenced.
 * lda    leading dimension of A. It must be at least max (1, n).
 * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * beta   double precision scalar multiplier applied to vector y.
 * y      double precision array of length at least (1 + (n - 1) * abs(incy)).
 *        If beta is zero, y is not read.
 * incy   storage spacing between elements of y. incy must not be zero.
 *
 * Output
 * ------
 * y      updated according to y = alpha*A*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/dsymv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDsymvNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jdouble alpha, jobject A, jint lda, jobject x, jint incx, jdouble beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDsymv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDsymv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasDsymv");
        return;
    }
    double* nativeA;
    double* nativeX;
    double* nativeY;

    nativeA = (double*)getPointer(env, A);
    nativeX = (double*)getPointer(env, x);
    nativeY = (double*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasDsymv(%c, %d, %lf, '%s', %d, '%s', %d, %lf, '%s', %d)\n",
        uplo, n, alpha, "A", lda, "x", incx, beta, "y", incy);

    cublasDsymv((char)uplo, n, alpha, nativeA, lda, nativeX, incx, beta, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasDsbmv (char uplo, int n, int k, double alpha, const double *A, int lda,
 *              const double *x, int incx, double beta, double *y, int incy)
 *
 * performs the matrix-vector operation
 *
 *     y := alpha*A*x + beta*y
 *
 * alpha and beta are double precision scalars. x and y are double precision
 * vectors with n elements. A is an n by n symmetric band matrix consisting
 * of double precision elements, with k super-diagonals and the same number
 * of subdiagonals.
 *
 * Input
 * -----
 * uplo   specifies whether the upper or lower triangular part of the symmetric
 *        band matrix A is being supplied. If uplo == 'U' or 'u', the upper
 *        triangular part is being supplied. If uplo == 'L' or 'l', the lower
 *        triangular part is being supplied.
 * n      specifies the number of rows and the number of columns of the
 *        symmetric matrix A. n must be at least zero.
 * k      specifies the number of super-diagonals of matrix A. Since the matrix
 *        is symmetric, this is also the number of sub-diagonals. k must be at
 *        least zero.
 * alpha  double precision scalar multiplier applied to A*x.
 * A      double precision array of dimensions (lda, n). When uplo == 'U' or
 *        'u', the leading (k + 1) x n part of array A must contain the upper
 *        triangular band of the symmetric matrix, supplied column by column,
 *        with the leading diagonal of the matrix in row (k+1) of the array,
 *        the first super-diagonal starting at position 2 in row k, and so on.
 *        The top left k x k triangle of the array A is not referenced. When
 *        uplo == 'L' or 'l', the leading (k + 1) x n part of the array A must
 *        contain the lower triangular band part of the symmetric matrix,
 *        supplied column by column, with the leading diagonal of the matrix in
 *        row 1 of the array, the first sub-diagonal starting at position 1 in
 *        row 2, and so on. The bottom right k x k triangle of the array A is
 *        not referenced.
 * lda    leading dimension of A. lda must be at least (k + 1).
 * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * beta   double precision scalar multiplier applied to vector y. If beta is
 *        zero, y is not read.
 * y      double precision array of length at least (1 + (n - 1) * abs(incy)).
 *        If beta is zero, y is not read.
 * incy   storage spacing between elements of y. incy must not be zero.
 *
 * Output
 * ------
 * y      updated according to alpha*A*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/dsbmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if k or n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDsbmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jint k, jdouble alpha, jobject A, jint lda, jobject x, jint incx, jdouble beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDsbmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDsbmv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasDsbmv");
        return;
    }
    double* nativeA;
    double* nativeX;
    double* nativeY;

    nativeA = (double*)getPointer(env, A);
    nativeX = (double*)getPointer(env, x);
    nativeY = (double*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasDsbmv(%c, %d, %d, %lf, '%s', %d, '%s', %d, %lf, '%s', %d)\n",
        uplo, n, k, alpha, "A", lda, "x", incx, beta, "y", incy);

    cublasDsbmv((char)uplo, n, k, alpha, nativeA, lda, nativeX, incx, beta, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasDspmv (char uplo, int n, double alpha, const double *AP, const double *x,
 *              int incx, double beta, double *y, int incy)
 *
 * performs the matrix-vector operation
 *
 *    y = alpha * A * x + beta * y
 *
 * Alpha and beta are double precision scalars, and x and y are double
 * precision vectors with n elements. A is a symmetric n x n matrix
 * consisting of double precision elements that is supplied in packed form.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array AP. If uplo == 'U' or 'u', then the upper
 *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then
 *        the lower triangular part of A is supplied in AP.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  double precision scalar multiplier applied to A*x.
 * AP     double precision array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part
 *        of the symmetric matrix A, packed sequentially, column by column;
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * beta   double precision scalar multiplier applied to vector y;
 * y      double precision array of length at least (1 + (n - 1) * abs(incy)).
 *        If beta is zero, y is not read.
 * incy   storage spacing between elements of y. incy must not be zero.
 *
 * Output
 * ------
 * y      updated according to y = alpha*A*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/dspmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDspmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jdouble alpha, jobject AP, jobject x, jint incx, jdouble beta, jobject y, jint incy)
{
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasDspmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasDspmv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasDspmv");
        return;
    }
    double* nativeAP;
    double* nativeX;
    double* nativeY;

    nativeAP = (double*)getPointer(env, AP);
    nativeX = (double*)getPointer(env, x);
    nativeY = (double*)getPointer(env, y);

    Logger::log(LOG_TRACE, "Executing cublasDspmv(%c, %d, %lf, '%s', '%s', %d, %lf, '%s', %d)\n",
        uplo, n, alpha, "AP", "x", incx, beta, "y", incy);

    cublasDspmv((char)uplo, n, alpha, nativeAP, nativeX, incx, beta, nativeY, incy);
}




/**
 * <pre>
 * void
 * cublasDgemm (char transa, char transb, int m, int n, int k, double alpha,
 *              const double *A, int lda, const double *B, int ldb,
 *              double beta, double *C, int ldc)
 *
 * computes the product of matrix A and matrix B, multiplies the result
 * by scalar alpha, and adds the sum to the product of matrix C and
 * scalar beta. It performs one of the matrix-matrix operations:
 *
 * C = alpha * op(A) * op(B) + beta * C,
 * where op(X) = X or op(X) = transpose(X),
 *
 * and alpha and beta are double-precision scalars. A, B and C are matrices
 * consisting of double-precision elements, with op(A) an m x k matrix,
 * op(B) a k x n matrix, and C an m x n matrix. Matrices A, B, and C are
 * stored in column-major format, and lda, ldb, and ldc are the leading
 * dimensions of the two-dimensional arrays containing A, B, and C.
 *
 * Input
 * -----
 * transa specifies op(A). If transa == 'N' or 'n', op(A) = A.
 *        If transa == 'T', 't', 'C', or 'c', op(A) = transpose(A).
 * transb specifies op(B). If transb == 'N' or 'n', op(B) = B.
 *        If transb == 'T', 't', 'C', or 'c', op(B) = transpose(B).
 * m      number of rows of matrix op(A) and rows of matrix C; m must be at
 *        least zero.
 * n      number of columns of matrix op(B) and number of columns of C;
 *        n must be at least zero.
 * k      number of columns of matrix op(A) and number of rows of op(B);
 *        k must be at least zero.
 * alpha  double-precision scalar multiplier applied to op(A) * op(B).
 * A      double-precision array of dimensions (lda, k) if transa == 'N' or
 *        'n', and of dimensions (lda, m) otherwise. If transa == 'N' or
 *        'n' lda must be at least max(1, m), otherwise lda must be at
 *        least max(1, k).
 * lda    leading dimension of two-dimensional array used to store matrix A.
 * B      double-precision array of dimensions (ldb, n) if transb == 'N' or
 *        'n', and of dimensions (ldb, k) otherwise. If transb == 'N' or
 *        'n' ldb must be at least max (1, k), otherwise ldb must be at
 *        least max(1, n).
 * ldb    leading dimension of two-dimensional array used to store matrix B.
 * beta   double-precision scalar multiplier applied to C. If zero, C does not
 *        have to be a valid input
 * C      double-precision array of dimensions (ldc, n); ldc must be at least
 *        max(1, m).
 * ldc    leading dimension of two-dimensional array used to store matrix C.
 *
 * Output
 * ------
 * C      updated based on C = alpha * op(A)*op(B) + beta * C.
 *
 * Reference: http://www.netlib.org/blas/sgemm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS was not initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m < 0, n < 0, or k < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDgemmNative
    (JNIEnv *env, jclass cls, jchar transa, jchar transb, jint m, jint n, jint k, jdouble alpha, jobject A, jint lda, jobject B, jint ldb, jdouble beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDgemm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasDgemm");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasDgemm");
        return;
    }
    double* nativeA;
    double* nativeB;
    double* nativeC;

    nativeA = (double*)getPointer(env, A);
    nativeB = (double*)getPointer(env, B);
    nativeC = (double*)getPointer(env, C);

    Logger::log(LOG_TRACE, "Executing cublasDgemm(%c, %c, %d, %d, %d, %lf, '%s', %d, '%s', %d, %lf, '%s', %d)\n",
        transa, transb, m, n, k, alpha, "A", lda, "B", ldb, beta, "C", ldc);

    cublasDgemm((char)transa, (char)transb, m, n, k, alpha, nativeA, lda, nativeB, ldb, beta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasDtrsm (char side, char uplo, char transa, char diag, int m, int n,
 *              double alpha, const double *A, int lda, double *B, int ldb)
 *
 * solves one of the matrix equations
 *
 *    op(A) * X = alpha * B,   or   X * op(A) = alpha * B,
 *
 * where alpha is a double precision scalar, and X and B are m x n matrices
 * that are composed of double precision elements. A is a unit or non-unit,
 * upper or lower triangular matrix, and op(A) is one of
 *
 *    op(A) = A  or  op(A) = transpose(A)
 *
 * The result matrix X overwrites input matrix B; that is, on exit the result
 * is stored in B. Matrices A and B are stored in column major format, and
 * lda and ldb are the leading dimensions of the two-dimensonials arrays that
 * contain A and B, respectively.
 *
 * Input
 * -----
 * side   specifies whether op(A) appears on the left or right of X as
 *        follows: side = 'L' or 'l' indicates solve op(A) * X = alpha * B.
 *        side = 'R' or 'r' indicates solve X * op(A) = alpha * B.
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix as follows: uplo = 'U' or 'u' indicates A is an upper
 *        triangular matrix. uplo = 'L' or 'l' indicates A is a lower
 *        triangular matrix.
 * transa specifies the form of op(A) to be used in matrix multiplication
 *        as follows: If transa = 'N' or 'N', then op(A) = A. If transa =
 *        'T', 't', 'C', or 'c', then op(A) = transpose(A).
 * diag   specifies whether or not A is a unit triangular matrix like so:
 *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
 *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
 * m      specifies the number of rows of B. m must be at least zero.
 * n      specifies the number of columns of B. n must be at least zero.
 * alpha  is a double precision scalar to be multiplied with B. When alpha is
 *        zero, then A is not referenced and B need not be set before entry.
 * A      is a double precision array of dimensions (lda, k), where k is
 *        m when side = 'L' or 'l', and is n when side = 'R' or 'r'. If
 *        uplo = 'U' or 'u', the leading k x k upper triangular part of
 *        the array A must contain the upper triangular matrix and the
 *        strictly lower triangular matrix of A is not referenced. When
 *        uplo = 'L' or 'l', the leading k x k lower triangular part of
 *        the array A must contain the lower triangular matrix and the
 *        strictly upper triangular part of A is not referenced. Note that
 *        when diag = 'U' or 'u', the diagonal elements of A are not
 *        referenced, and are assumed to be unity.
 * lda    is the leading dimension of the two dimensional array containing A.
 *        When side = 'L' or 'l' then lda must be at least max(1, m), when
 *        side = 'R' or 'r' then lda must be at least max(1, n).
 * B      is a double precision array of dimensions (ldb, n). ldb must be
 *        at least max (1,m). The leading m x n part of the array B must
 *        contain the right-hand side matrix B. On exit B is overwritten
 *        by the solution matrix X.
 * ldb    is the leading dimension of the two dimensional array containing B.
 *        ldb must be at least max(1, m).
 *
 * Output
 * ------
 * B      contains the solution matrix X satisfying op(A) * X = alpha * B,
 *        or X * op(A) = alpha * B
 *
 * Reference: http://www.netlib.org/blas/dtrsm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDtrsmNative
    (JNIEnv *env, jclass cls, jchar side, jchar uplo, jchar transa, jchar diag, jint m, jint n, jdouble alpha, jobject A, jint lda, jobject B, jint ldb)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDtrsm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasDtrsm");
        return;
    }
    double* nativeA;
    double* nativeB;

    nativeA = (double*)getPointer(env, A);
    nativeB = (double*)getPointer(env, B);

    Logger::log(LOG_TRACE, "Executing cublasDtrsm(%c, %c, %c, %c, %d, %d, %lf, '%s', %d, '%s', %d)\n",
        side, uplo, transa, diag, m, n, alpha, "A", lda, "B", ldb);

    cublasDtrsm((char)side, (char)uplo, (char)transa, (char)diag, m, n, alpha, nativeA, lda, nativeB, ldb);
}




/**
 * <pre>
 * void
 * cublasZtrsm (char side, char uplo, char transa, char diag, int m, int n,
 *              cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
 *              cuDoubleComplex *B, int ldb)
 *
 * solves one of the matrix equations
 *
 *    op(A) * X = alpha * B,   or   X * op(A) = alpha * B,
 *
 * where alpha is a double precision complex scalar, and X and B are m x n matrices
 * that are composed of double precision complex elements. A is a unit or non-unit,
 * upper or lower triangular matrix, and op(A) is one of
 *
 *    op(A) = A  or  op(A) = transpose(A)  or  op( A ) = conj( A' ).
 *
 * The result matrix X overwrites input matrix B; that is, on exit the result
 * is stored in B. Matrices A and B are stored in column major format, and
 * lda and ldb are the leading dimensions of the two-dimensonials arrays that
 * contain A and B, respectively.
 *
 * Input
 * -----
 * side   specifies whether op(A) appears on the left or right of X as
 *        follows: side = 'L' or 'l' indicates solve op(A) * X = alpha * B.
 *        side = 'R' or 'r' indicates solve X * op(A) = alpha * B.
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix as follows: uplo = 'U' or 'u' indicates A is an upper
 *        triangular matrix. uplo = 'L' or 'l' indicates A is a lower
 *        triangular matrix.
 * transa specifies the form of op(A) to be used in matrix multiplication
 *        as follows: If transa = 'N' or 'N', then op(A) = A. If transa =
 *        'T', 't', 'C', or 'c', then op(A) = transpose(A).
 * diag   specifies whether or not A is a unit triangular matrix like so:
 *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
 *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
 * m      specifies the number of rows of B. m must be at least zero.
 * n      specifies the number of columns of B. n must be at least zero.
 * alpha  is a double precision complex scalar to be multiplied with B. When alpha is
 *        zero, then A is not referenced and B need not be set before entry.
 * A      is a double precision complex array of dimensions (lda, k), where k is
 *        m when side = 'L' or 'l', and is n when side = 'R' or 'r'. If
 *        uplo = 'U' or 'u', the leading k x k upper triangular part of
 *        the array A must contain the upper triangular matrix and the
 *        strictly lower triangular matrix of A is not referenced. When
 *        uplo = 'L' or 'l', the leading k x k lower triangular part of
 *        the array A must contain the lower triangular matrix and the
 *        strictly upper triangular part of A is not referenced. Note that
 *        when diag = 'U' or 'u', the diagonal elements of A are not
 *        referenced, and are assumed to be unity.
 * lda    is the leading dimension of the two dimensional array containing A.
 *        When side = 'L' or 'l' then lda must be at least max(1, m), when
 *        side = 'R' or 'r' then lda must be at least max(1, n).
 * B      is a double precision complex array of dimensions (ldb, n). ldb must be
 *        at least max (1,m). The leading m x n part of the array B must
 *        contain the right-hand side matrix B. On exit B is overwritten
 *        by the solution matrix X.
 * ldb    is the leading dimension of the two dimensional array containing B.
 *        ldb must be at least max(1, m).
 *
 * Output
 * ------
 * B      contains the solution matrix X satisfying op(A) * X = alpha * B,
 *        or X * op(A) = alpha * B
 *
 * Reference: http://www.netlib.org/blas/ztrsm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZtrsmNative
    (JNIEnv *env, jclass cls, jchar side, jchar uplo, jchar transa, jchar diag, jint m, jint n, jobject alpha, jobject A, jint lda, jobject B, jint ldb)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZtrsm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasZtrsm");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeB;
    cuDoubleComplex dobuleComplexAlpha;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeB = (cuDoubleComplex*)getPointer(env, B);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZtrsm(%c, %c, %c, %c, %d, %d, [%lf,%lf], '%s', %d, '%s', %d)\n",
        side, uplo, transa, diag, m, n, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "A", lda, "B", ldb);

    cublasZtrsm((char)side, (char)uplo, (char)transa, (char)diag, m, n, dobuleComplexAlpha, nativeA, lda, nativeB, ldb);
}




/**
 * <pre>
 * void
 * cublasDtrmm (char side, char uplo, char transa, char diag, int m, int n,
 *              double alpha, const double *A, int lda, const double *B, int ldb)
 *
 * performs one of the matrix-matrix operations
 *
 *   B = alpha * op(A) * B,  or  B = alpha * B * op(A)
 *
 * where alpha is a double-precision scalar, B is an m x n matrix composed
 * of double precision elements, and A is a unit or non-unit, upper or lower,
 * triangular matrix composed of double precision elements. op(A) is one of
 *
 *   op(A) = A  or  op(A) = transpose(A)
 *
 * Matrices A and B are stored in column major format, and lda and ldb are
 * the leading dimensions of the two-dimensonials arrays that contain A and
 * B, respectively.
 *
 * Input
 * -----
 * side   specifies whether op(A) multiplies B from the left or right.
 *        If side = 'L' or 'l', then B = alpha * op(A) * B. If side =
 *        'R' or 'r', then B = alpha * B * op(A).
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix. If uplo = 'U' or 'u', A is an upper triangular matrix.
 *        If uplo = 'L' or 'l', A is a lower triangular matrix.
 * transa specifies the form of op(A) to be used in the matrix
 *        multiplication. If transa = 'N' or 'n', then op(A) = A. If
 *        transa = 'T', 't', 'C', or 'c', then op(A) = transpose(A).
 * diag   specifies whether or not A is unit triangular. If diag = 'U'
 *        or 'u', A is assumed to be unit triangular. If diag = 'N' or
 *        'n', A is not assumed to be unit triangular.
 * m      the number of rows of matrix B. m must be at least zero.
 * n      the number of columns of matrix B. n must be at least zero.
 * alpha  double precision scalar multiplier applied to op(A)*B, or
 *        B*op(A), respectively. If alpha is zero no accesses are made
 *        to matrix A, and no read accesses are made to matrix B.
 * A      double precision array of dimensions (lda, k). k = m if side =
 *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
 *        the leading k x k upper triangular part of the array A must
 *        contain the upper triangular matrix, and the strictly lower
 *        triangular part of A is not referenced. If uplo = 'L' or 'l'
 *        the leading k x k lower triangular part of the array A must
 *        contain the lower triangular matrix, and the strictly upper
 *        triangular part of A is not referenced. When diag = 'U' or 'u'
 *        the diagonal elements of A are no referenced and are assumed
 *        to be unity.
 * lda    leading dimension of A. When side = 'L' or 'l', it must be at
 *        least max(1,m) and at least max(1,n) otherwise
 * B      double precision array of dimensions (ldb, n). On entry, the
 *        leading m x n part of the array contains the matrix B. It is
 *        overwritten with the transformed matrix on exit.
 * ldb    leading dimension of B. It must be at least max (1, m).
 *
 * Output
 * ------
 * B      updated according to B = alpha * op(A) * B  or B = alpha * B * op(A)
 *
 * Reference: http://www.netlib.org/blas/dtrmm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDtrmmNative
    (JNIEnv *env, jclass cls, jchar side, jchar uplo, jchar transa, jchar diag, jint m, jint n, jdouble alpha, jobject A, jint lda, jobject B, jint ldb)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDtrmm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasDtrmm");
        return;
    }
    double* nativeA;
    double* nativeB;

    nativeA = (double*)getPointer(env, A);
    nativeB = (double*)getPointer(env, B);

    Logger::log(LOG_TRACE, "Executing cublasDtrmm(%c, %c, %c, %c, %d, %d, %lf, '%s', %d, '%s', %d)\n",
        side, uplo, transa, diag, m, n, alpha, "A", lda, "B", ldb);

    cublasDtrmm((char)side, (char)uplo, (char)transa, (char)diag, m, n, alpha, nativeA, lda, nativeB, ldb);
}




/**
 * <pre>
 * void
 * cublasDsymm (char side, char uplo, int m, int n, double alpha,
 *              const double *A, int lda, const double *B, int ldb,
 *              double beta, double *C, int ldc);
 *
 * performs one of the matrix-matrix operations
 *
 *   C = alpha * A * B + beta * C, or
 *   C = alpha * B * A + beta * C,
 *
 * where alpha and beta are double precision scalars, A is a symmetric matrix
 * consisting of double precision elements and stored in either lower or upper
 * storage mode, and B and C are m x n matrices consisting of double precision
 * elements.
 *
 * Input
 * -----
 * side   specifies whether the symmetric matrix A appears on the left side
 *        hand side or right hand side of matrix B, as follows. If side == 'L'
 *        or 'l', then C = alpha * A * B + beta * C. If side = 'R' or 'r',
 *        then C = alpha * B * A + beta * C.
 * uplo   specifies whether the symmetric matrix A is stored in upper or lower
 *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the symmetric matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the symmetric matrix is to be referenced,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * m      specifies the number of rows of the matrix C, and the number of rows
 *        of matrix B. It also specifies the dimensions of symmetric matrix A
 *        when side == 'L' or 'l'. m must be at least zero.
 * n      specifies the number of columns of the matrix C, and the number of
 *        columns of matrix B. It also specifies the dimensions of symmetric
 *        matrix A when side == 'R' or 'r'. n must be at least zero.
 * alpha  double precision scalar multiplier applied to A * B, or B * A
 * A      double precision array of dimensions (lda, ka), where ka is m when
 *        side == 'L' or 'l' and is n otherwise. If side == 'L' or 'l' the
 *        leading m x m part of array A must contain the symmetric matrix,
 *        such that when uplo == 'U' or 'u', the leading m x m part stores the
 *        upper triangular part of the symmetric matrix, and the strictly lower
 *        triangular part of A is not referenced, and when uplo == 'U' or 'u',
 *        the leading m x m part stores the lower triangular part of the
 *        symmetric matrix and the strictly upper triangular part is not
 *        referenced. If side == 'R' or 'r' the leading n x n part of array A
 *        must contain the symmetric matrix, such that when uplo == 'U' or 'u',
 *        the leading n x n part stores the upper triangular part of the
 *        symmetric matrix and the strictly lower triangular part of A is not
 *        referenced, and when uplo == 'U' or 'u', the leading n x n part
 *        stores the lower triangular part of the symmetric matrix and the
 *        strictly upper triangular part is not referenced.
 * lda    leading dimension of A. When side == 'L' or 'l', it must be at least
 *        max(1, m) and at least max(1, n) otherwise.
 * B      double precision array of dimensions (ldb, n). On entry, the leading
 *        m x n part of the array contains the matrix B.
 * ldb    leading dimension of B. It must be at least max (1, m).
 * beta   double precision scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input
 * C      double precision array of dimensions (ldc, n)
 * ldc    leading dimension of C. Must be at least max(1, m)
 *
 * Output
 * ------
 * C      updated according to C = alpha * A * B + beta * C, or C = alpha *
 *        B * A + beta * C
 *
 * Reference: http://www.netlib.org/blas/dsymm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDsymmNative
    (JNIEnv *env, jclass cls, jchar side, jchar uplo, jint m, jint n, jdouble alpha, jobject A, jint lda, jobject B, jint ldb, jdouble beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDsymm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasDsymm");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasDsymm");
        return;
    }
    double* nativeA;
    double* nativeB;
    double* nativeC;

    nativeA = (double*)getPointer(env, A);
    nativeB = (double*)getPointer(env, B);
    nativeC = (double*)getPointer(env, C);

    Logger::log(LOG_TRACE, "Executing cublasDsymm(%c, %c, %d, %d, %lf, '%s', %d, '%s', %d, %lf, '%s', %d)\n",
        side, uplo, m, n, alpha, "A", lda, "B", ldb, beta, "C", ldc);

    cublasDsymm((char)side, (char)uplo, m, n, alpha, nativeA, lda, nativeB, ldb, beta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasZsymm (char side, char uplo, int m, int n, cuDoubleComplex alpha,
 *              const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
 *              cuDoubleComplex beta, cuDoubleComplex *C, int ldc);
 *
 * performs one of the matrix-matrix operations
 *
 *   C = alpha * A * B + beta * C, or
 *   C = alpha * B * A + beta * C,
 *
 * where alpha and beta are double precision complex scalars, A is a symmetric matrix
 * consisting of double precision complex elements and stored in either lower or upper
 * storage mode, and B and C are m x n matrices consisting of double precision
 * complex elements.
 *
 * Input
 * -----
 * side   specifies whether the symmetric matrix A appears on the left side
 *        hand side or right hand side of matrix B, as follows. If side == 'L'
 *        or 'l', then C = alpha * A * B + beta * C. If side = 'R' or 'r',
 *        then C = alpha * B * A + beta * C.
 * uplo   specifies whether the symmetric matrix A is stored in upper or lower
 *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the symmetric matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the symmetric matrix is to be referenced,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * m      specifies the number of rows of the matrix C, and the number of rows
 *        of matrix B. It also specifies the dimensions of symmetric matrix A
 *        when side == 'L' or 'l'. m must be at least zero.
 * n      specifies the number of columns of the matrix C, and the number of
 *        columns of matrix B. It also specifies the dimensions of symmetric
 *        matrix A when side == 'R' or 'r'. n must be at least zero.
 * alpha  double precision scalar multiplier applied to A * B, or B * A
 * A      double precision array of dimensions (lda, ka), where ka is m when
 *        side == 'L' or 'l' and is n otherwise. If side == 'L' or 'l' the
 *        leading m x m part of array A must contain the symmetric matrix,
 *        such that when uplo == 'U' or 'u', the leading m x m part stores the
 *        upper triangular part of the symmetric matrix, and the strictly lower
 *        triangular part of A is not referenced, and when uplo == 'U' or 'u',
 *        the leading m x m part stores the lower triangular part of the
 *        symmetric matrix and the strictly upper triangular part is not
 *        referenced. If side == 'R' or 'r' the leading n x n part of array A
 *        must contain the symmetric matrix, such that when uplo == 'U' or 'u',
 *        the leading n x n part stores the upper triangular part of the
 *        symmetric matrix and the strictly lower triangular part of A is not
 *        referenced, and when uplo == 'U' or 'u', the leading n x n part
 *        stores the lower triangular part of the symmetric matrix and the
 *        strictly upper triangular part is not referenced.
 * lda    leading dimension of A. When side == 'L' or 'l', it must be at least
 *        max(1, m) and at least max(1, n) otherwise.
 * B      double precision array of dimensions (ldb, n). On entry, the leading
 *        m x n part of the array contains the matrix B.
 * ldb    leading dimension of B. It must be at least max (1, m).
 * beta   double precision scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input
 * C      double precision array of dimensions (ldc, n)
 * ldc    leading dimension of C. Must be at least max(1, m)
 *
 * Output
 * ------
 * C      updated according to C = alpha * A * B + beta * C, or C = alpha *
 *        B * A + beta * C
 *
 * Reference: http://www.netlib.org/blas/zsymm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZsymmNative
    (JNIEnv *env, jclass cls, jchar side, jchar uplo, jint m, jint n, jobject alpha, jobject A, jint lda, jobject B, jint ldb, jobject beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZsymm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasZsymm");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasZsymm");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeB;
    cuDoubleComplex* nativeC;
    cuDoubleComplex dobuleComplexAlpha;
    cuDoubleComplex dobuleComplexBeta;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeB = (cuDoubleComplex*)getPointer(env, B);
    nativeC = (cuDoubleComplex*)getPointer(env, C);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    dobuleComplexBeta.x = env->GetDoubleField(beta, cuDoubleComplex_x);
    dobuleComplexBeta.y = env->GetDoubleField(beta, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZsymm(%c, %c, %d, %d, [%lf,%lf], '%s', %d, '%s', %d, [%lf,%lf], '%s', %d)\n",
        side, uplo, m, n, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "A", lda, "B", ldb, dobuleComplexBeta.x, dobuleComplexBeta.y, "C", ldc);

    cublasZsymm((char)side, (char)uplo, m, n, dobuleComplexAlpha, nativeA, lda, nativeB, ldb, dobuleComplexBeta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasDsyrk (char uplo, char trans, int n, int k, double alpha,
 *              const double *A, int lda, double beta, double *C, int ldc)
 *
 * performs one of the symmetric rank k operations
 *
 *   C = alpha * A * transpose(A) + beta * C, or
 *   C = alpha * transpose(A) * A + beta * C.
 *
 * Alpha and beta are double precision scalars. C is an n x n symmetric matrix
 * consisting of double precision elements and stored in either lower or
 * upper storage mode. A is a matrix consisting of double precision elements
 * with dimension of n x k in the first case, and k x n in the second case.
 *
 * Input
 * -----
 * uplo   specifies whether the symmetric matrix C is stored in upper or lower
 *        storage mode as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the symmetric matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the symmetric matrix is to be referenced,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * trans  specifies the operation to be performed. If trans == 'N' or 'n', C =
 *        alpha * transpose(A) + beta * C. If trans == 'T', 't', 'C', or 'c',
 *        C = transpose(A) * A + beta * C.
 * n      specifies the number of rows and the number columns of matrix C. If
 *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
 *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
 *        n must be at least zero.
 * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
 *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
 *        matrix A. k must be at least zero.
 * alpha  double precision scalar multiplier applied to A * transpose(A) or
 *        transpose(A) * A.
 * A      double precision array of dimensions (lda, ka), where ka is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array A must contain the matrix A,
 *        otherwise the leading k x n part of the array must contains the
 *        matrix A.
 * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
 *        least max(1, n). Otherwise lda must be at least max(1, k).
 * beta   double precision scalar multiplier applied to C. If beta izs zero, C
 *        does not have to be a valid input
 * C      double precision array of dimensions (ldc, n). If uplo = 'U' or 'u',
 *        the leading n x n triangular part of the array C must contain the
 *        upper triangular part of the symmetric matrix C and the strictly
 *        lower triangular part of C is not referenced. On exit, the upper
 *        triangular part of C is overwritten by the upper triangular part of
 *        the updated matrix. If uplo = 'L' or 'l', the leading n x n
 *        triangular part of the array C must contain the lower triangular part
 *        of the symmetric matrix C and the strictly upper triangular part of C
 *        is not referenced. On exit, the lower triangular part of C is
 *        overwritten by the lower triangular part of the updated matrix.
 * ldc    leading dimension of C. It must be at least max(1, n).
 *
 * Output
 * ------
 * C      updated according to C = alpha * A * transpose(A) + beta * C, or C =
 *        alpha * transpose(A) * A + beta * C
 *
 * Reference: http://www.netlib.org/blas/dsyrk.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDsyrkNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jint n, jint k, jdouble alpha, jobject A, jint lda, jdouble beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDsyrk");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasDsyrk");
        return;
    }
    double* nativeA;
    double* nativeC;

    nativeA = (double*)getPointer(env, A);
    nativeC = (double*)getPointer(env, C);

    Logger::log(LOG_TRACE, "Executing cublasDsyrk(%c, %c, %d, %d, %lf, '%s', %d, %lf, '%s', %d)\n",
        uplo, trans, n, k, alpha, "A", lda, beta, "C", ldc);

    cublasDsyrk((char)uplo, (char)trans, n, k, alpha, nativeA, lda, beta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasZsyrk (char uplo, char trans, int n, int k, cuDoubleComplex alpha,
 *              const cuDoubleComplex *A, int lda, cuDoubleComplex beta, cuDoubleComplex *C, int ldc)
 *
 * performs one of the symmetric rank k operations
 *
 *   C = alpha * A * transpose(A) + beta * C, or
 *   C = alpha * transpose(A) * A + beta * C.
 *
 * Alpha and beta are double precision complex scalars. C is an n x n symmetric matrix
 * consisting of double precision complex elements and stored in either lower or
 * upper storage mode. A is a matrix consisting of double precision complex elements
 * with dimension of n x k in the first case, and k x n in the second case.
 *
 * Input
 * -----
 * uplo   specifies whether the symmetric matrix C is stored in upper or lower
 *        storage mode as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the symmetric matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the symmetric matrix is to be referenced,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * trans  specifies the operation to be performed. If trans == 'N' or 'n', C =
 *        alpha * transpose(A) + beta * C. If trans == 'T', 't', 'C', or 'c',
 *        C = transpose(A) * A + beta * C.
 * n      specifies the number of rows and the number columns of matrix C. If
 *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
 *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
 *        n must be at least zero.
 * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
 *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
 *        matrix A. k must be at least zero.
 * alpha  double precision complex scalar multiplier applied to A * transpose(A) or
 *        transpose(A) * A.
 * A      double precision complex array of dimensions (lda, ka), where ka is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array A must contain the matrix A,
 *        otherwise the leading k x n part of the array must contains the
 *        matrix A.
 * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
 *        least max(1, n). Otherwise lda must be at least max(1, k).
 * beta   double precision complex scalar multiplier applied to C. If beta izs zero, C
 *        does not have to be a valid input
 * C      double precision complex array of dimensions (ldc, n). If uplo = 'U' or 'u',
 *        the leading n x n triangular part of the array C must contain the
 *        upper triangular part of the symmetric matrix C and the strictly
 *        lower triangular part of C is not referenced. On exit, the upper
 *        triangular part of C is overwritten by the upper triangular part of
 *        the updated matrix. If uplo = 'L' or 'l', the leading n x n
 *        triangular part of the array C must contain the lower triangular part
 *        of the symmetric matrix C and the strictly upper triangular part of C
 *        is not referenced. On exit, the lower triangular part of C is
 *        overwritten by the lower triangular part of the updated matrix.
 * ldc    leading dimension of C. It must be at least max(1, n).
 *
 * Output
 * ------
 * C      updated according to C = alpha * A * transpose(A) + beta * C, or C =
 *        alpha * transpose(A) * A + beta * C
 *
 * Reference: http://www.netlib.org/blas/zsyrk.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZsyrkNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jint n, jint k, jobject alpha, jobject A, jint lda, jobject beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZsyrk");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasZsyrk");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeC;
    cuDoubleComplex dobuleComplexAlpha;
    cuDoubleComplex dobuleComplexBeta;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeC = (cuDoubleComplex*)getPointer(env, C);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    dobuleComplexBeta.x = env->GetDoubleField(beta, cuDoubleComplex_x);
    dobuleComplexBeta.y = env->GetDoubleField(beta, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZsyrk(%c, %c, %d, %d, [%lf,%lf], '%s', %d, [%lf,%lf], '%s', %d)\n",
        uplo, trans, n, k, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "A", lda, dobuleComplexBeta.x, dobuleComplexBeta.y, "C", ldc);

    cublasZsyrk((char)uplo, (char)trans, n, k, dobuleComplexAlpha, nativeA, lda, dobuleComplexBeta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasZsyr2k (char uplo, char trans, int n, int k, cuDoubleComplex alpha,
 *               const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
 *               cuDoubleComplex beta, cuDoubleComplex *C, int ldc)
 *
 * performs one of the symmetric rank 2k operations
 *
 *    C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C, or
 *    C = alpha * transpose(A) * B + alpha * transpose(B) * A + beta * C.
 *
 * Alpha and beta are double precision complex scalars. C is an n x n symmetric matrix
 * consisting of double precision complex elements and stored in either lower or upper
 * storage mode. A and B are matrices consisting of double precision complex elements
 * with dimension of n x k in the first case, and k x n in the second case.
 *
 * Input
 * -----
 * uplo   specifies whether the symmetric matrix C is stored in upper or lower
 *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the symmetric matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the symmetric matrix is to be references,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * trans  specifies the operation to be performed. If trans == 'N' or 'n',
 *        C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C,
 *        If trans == 'T', 't', 'C', or 'c', C = alpha * transpose(A) * B +
 *        alpha * transpose(B) * A + beta * C.
 * n      specifies the number of rows and the number columns of matrix C. If
 *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
 *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
 *        n must be at least zero.
 * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
 *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
 *        matrix A. k must be at least zero.
 * alpha  double precision scalar multiplier.
 * A      double precision array of dimensions (lda, ka), where ka is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array A must contain the matrix A,
 *        otherwise the leading k x n part of the array must contain the matrix
 *        A.
 * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
 *        least max(1, n). Otherwise lda must be at least max(1,k).
 * B      double precision array of dimensions (lda, kb), where kb is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array B must contain the matrix B,
 *        otherwise the leading k x n part of the array must contain the matrix
 *        B.
 * ldb    leading dimension of N. When trans == 'N' or 'n' then ldb must be at
 *        least max(1, n). Otherwise ldb must be at least max(1, k).
 * beta   double precision scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input.
 * C      double precision array of dimensions (ldc, n). If uplo == 'U' or 'u',
 *        the leading n x n triangular part of the array C must contain the
 *        upper triangular part of the symmetric matrix C and the strictly
 *        lower triangular part of C is not referenced. On exit, the upper
 *        triangular part of C is overwritten by the upper triangular part of
 *        the updated matrix. If uplo == 'L' or 'l', the leading n x n
 *        triangular part of the array C must contain the lower triangular part
 *        of the symmetric matrix C and the strictly upper triangular part of C
 *        is not referenced. On exit, the lower triangular part of C is
 *        overwritten by the lower triangular part of the updated matrix.
 * ldc    leading dimension of C. Must be at least max(1, n).
 *
 * Output
 * ------
 * C      updated according to alpha*A*transpose(B) + alpha*B*transpose(A) +
 *        beta*C or alpha*transpose(A)*B + alpha*transpose(B)*A + beta*C
 *
 * Reference:   http://www.netlib.org/blas/zsyr2k.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZsyr2kNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jint n, jint k, jobject alpha, jobject A, jint lda, jobject B, jint ldb, jobject beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZsyr2k");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasZsyr2k");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasZsyr2k");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeB;
    cuDoubleComplex* nativeC;
    cuDoubleComplex dobuleComplexAlpha;
    cuDoubleComplex dobuleComplexBeta;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeB = (cuDoubleComplex*)getPointer(env, B);
    nativeC = (cuDoubleComplex*)getPointer(env, C);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    dobuleComplexBeta.x = env->GetDoubleField(beta, cuDoubleComplex_x);
    dobuleComplexBeta.y = env->GetDoubleField(beta, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZsyr2k(%c, %c, %d, %d, [%lf,%lf], '%s', %d, '%s', %d, [%lf,%lf], '%s', %d)\n",
        uplo, trans, n, k, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "A", lda, "B", ldb, dobuleComplexBeta.x, dobuleComplexBeta.y, "C", ldc);

    cublasZsyr2k((char)uplo, (char)trans, n, k, dobuleComplexAlpha, nativeA, lda, nativeB, ldb, dobuleComplexBeta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasZher2k (char uplo, char trans, int n, int k, cuDoubleComplex alpha,
 *               const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
 *               double beta, cuDoubleComplex *C, int ldc)
 *
 * performs one of the hermitian rank 2k operations
 *
 *    C =   alpha * A * conjugate(transpose(B))
 *        + conjugate(alpha) * B * conjugate(transpose(A))
 *        + beta * C ,
 *    or
 *    C =  alpha * conjugate(transpose(A)) * B
 *       + conjugate(alpha) * conjugate(transpose(B)) * A
 *       + beta * C.
 *
 * Alpha is double precision complex scalar whereas Beta is a double precision real scalar.
 * C is an n x n hermitian matrix consisting of double precision complex elements and
 * stored in either lower or upper storage mode. A and B are matrices consisting of
 * double precision complex elements with dimension of n x k in the first case,
 * and k x n in the second case.
 *
 * Input
 * -----
 * uplo   specifies whether the hermitian matrix C is stored in upper or lower
 *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the hermitian matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the hermitian matrix is to be references,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * trans  specifies the operation to be performed. If trans == 'N' or 'n',
 *        C =   alpha * A * conjugate(transpose(B))
 *            + conjugate(alpha) * B * conjugate(transpose(A))
 *            + beta * C .
 *        If trans == 'T', 't', 'C', or 'c',
 *        C =  alpha * conjugate(transpose(A)) * B
 *          + conjugate(alpha) * conjugate(transpose(B)) * A
 *          + beta * C.
 * n      specifies the number of rows and the number columns of matrix C. If
 *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
 *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
 *        n must be at least zero.
 * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
 *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
 *        matrix A. k must be at least zero.
 * alpha  double precision scalar multiplier.
 * A      double precision array of dimensions (lda, ka), where ka is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array A must contain the matrix A,
 *        otherwise the leading k x n part of the array must contain the matrix
 *        A.
 * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
 *        least max(1, n). Otherwise lda must be at least max(1,k).
 * B      double precision array of dimensions (lda, kb), where kb is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array B must contain the matrix B,
 *        otherwise the leading k x n part of the array must contain the matrix
 *        B.
 * ldb    leading dimension of N. When trans == 'N' or 'n' then ldb must be at
 *        least max(1, n). Otherwise ldb must be at least max(1, k).
 * beta   double precision scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input.
 * C      double precision array of dimensions (ldc, n). If uplo == 'U' or 'u',
 *        the leading n x n triangular part of the array C must contain the
 *        upper triangular part of the hermitian matrix C and the strictly
 *        lower triangular part of C is not referenced. On exit, the upper
 *        triangular part of C is overwritten by the upper triangular part of
 *        the updated matrix. If uplo == 'L' or 'l', the leading n x n
 *        triangular part of the array C must contain the lower triangular part
 *        of the hermitian matrix C and the strictly upper triangular part of C
 *        is not referenced. On exit, the lower triangular part of C is
 *        overwritten by the lower triangular part of the updated matrix.
 *        The imaginary parts of the diagonal elements need
 *        not be set,  they are assumed to be zero,  and on exit they
 *        are set to zero.
 * ldc    leading dimension of C. Must be at least max(1, n).
 *
 * Output
 * ------
 * C      updated according to alpha*A*conjugate(transpose(B)) +
 *        + conjugate(alpha)*B*conjugate(transpose(A)) + beta*C or
 *        alpha*conjugate(transpose(A))*B + conjugate(alpha)*conjugate(transpose(B))*A
 *        + beta*C.
 *
 * Reference:   http://www.netlib.org/blas/zher2k.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZher2kNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jint n, jint k, jobject alpha, jobject A, jint lda, jobject B, jint ldb, jdouble beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZher2k");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasZher2k");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasZher2k");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeB;
    cuDoubleComplex* nativeC;
    cuDoubleComplex dobuleComplexAlpha;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeB = (cuDoubleComplex*)getPointer(env, B);
    nativeC = (cuDoubleComplex*)getPointer(env, C);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZher2k(%c, %c, %d, %d, [%lf,%lf], '%s', %d, '%s', %d, %lf, '%s', %d)\n",
        uplo, trans, n, k, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "A", lda, "B", ldb, beta, "C", ldc);

    cublasZher2k((char)uplo, (char)trans, n, k, dobuleComplexAlpha, nativeA, lda, nativeB, ldb, beta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasZher (char uplo, int n, double alpha, const cuDoubleComplex *x, int incx,
 *             cuDoubleComplex *A, int lda)
 *
 * performs the hermitian rank 1 operation
 *
 *    A = alpha * x * conjugate(transpose(x)) + A,
 *
 * where alpha is a double precision real scalar, x is an n element double
 * precision complex vector and A is an n x n hermitian matrix consisting of
 * double precision complex elements. Matrix A is stored in column major format,
 * and lda is the leading dimension of the two-dimensional array
 * containing A.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or
 *        the lower triangular part of array A. If uplo = 'U' or 'u',
 *        then only the upper triangular part of A may be referenced.
 *        If uplo = 'L' or 'l', then only the lower triangular part of
 *        A may be referenced.
 * n      specifies the number of rows and columns of the matrix A. It
 *        must be at least 0.
 * alpha  double precision real scalar multiplier applied to
 *        x * conjugate(transpose(x))
 * x      double precision complex array of length at least (1 + (n - 1) * abs(incx))
 * incx   specifies the storage spacing between elements of x. incx must
 *        not be zero.
 * A      double precision complex array of dimensions (lda, n). If uplo = 'U' or
 *        'u', then A must contain the upper triangular part of a hermitian
 *        matrix, and the strictly lower triangular part is not referenced.
 *        If uplo = 'L' or 'l', then A contains the lower triangular part
 *        of a hermitian matrix, and the strictly upper triangular part is
 *        not referenced. The imaginary parts of the diagonal elements need
 *        not be set, they are assumed to be zero, and on exit they
 *        are set to zero.
 * lda    leading dimension of the two-dimensional array containing A. lda
 *        must be at least max(1, n).
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * conjugate(transpose(x)) + A
 *
 * Reference: http://www.netlib.org/blas/zher.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZherNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jdouble alpha, jobject x, jint incx, jobject A, jint lda)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZher");
        return;
    }
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZher");
        return;
    }
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeA;

    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeA = (cuDoubleComplex*)getPointer(env, A);

    Logger::log(LOG_TRACE, "Executing cublasZher(%c, %d, %lf, '%s', %d, '%s', %d)\n",
        uplo, n, alpha, "x", incx, "A", lda);

    cublasZher((char)uplo, n, alpha, nativeX, incx, nativeA, lda);
}




/**
 * <pre>
 * void
 * cublasZhpr (char uplo, int n, double alpha, const cuDoubleComplex *x, int incx,
 *             cuDoubleComplex *AP)
 *
 * performs the hermitian rank 1 operation
 *
 *    A = alpha * x * conjugate(transpose(x)) + A,
 *
 * where alpha is a double precision real scalar and x is an n element double
 * precision complex vector. A is a hermitian n x n matrix consisting of double
 * precision complex elements that is supplied in packed form.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array AP. If uplo == 'U' or 'u', then the upper
 *        triangular part of A is supplied in AP. If uplo == 'L' or 'l', then
 *        the lower triangular part of A is supplied in AP.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  double precision real scalar multiplier applied to x * conjugate(transpose(x)).
 * x      double precision array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * AP     double precision complex array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part
 *        of the hermitian matrix A, packed sequentially, column by column;
 *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part
 *        of the hermitian matrix A, packed sequentially, column by column;
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 *        The imaginary parts of the diagonal elements need not be set, they
 *        are assumed to be zero, and on exit they are set to zero.
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * conjugate(transpose(x)) + A
 *
 * Reference: http://www.netlib.org/blas/zhpr.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, or incx == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZhprNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jdouble alpha, jobject x, jint incx, jobject AP)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZhpr");
        return;
    }
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasZhpr");
        return;
    }
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeAP;

    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeAP = (cuDoubleComplex*)getPointer(env, AP);

    Logger::log(LOG_TRACE, "Executing cublasZhpr(%c, %d, %lf, '%s', %d, '%s')\n",
        uplo, n, alpha, "x", incx, "AP");

    cublasZhpr((char)uplo, n, alpha, nativeX, incx, nativeAP);
}




/**
 * <pre>
 * void
 * cublasZhpr2 (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx,
 *              const cuDoubleComplex *y, int incy, cuDoubleComplex *AP)
 *
 * performs the hermitian rank 2 operation
 *
 *    A = alpha*x*conjugate(transpose(y)) + conjugate(alpha)*y*conjugate(transpose(x)) + A,
 *
 * where alpha is a double precision complex scalar, and x and y are n element double
 * precision complex vectors. A is a hermitian n x n matrix consisting of double
 * precision complex elements that is supplied in packed form.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array A. If uplo == 'U' or 'u', then only the
 *        upper triangular part of A may be referenced and the lower triangular
 *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
 *        triangular part of A may be referenced and the upper triangular part
 *        of A is inferred.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  double precision complex scalar multiplier applied to x * conjugate(transpose(y)) +
 *        y * conjugate(transpose(x)).
 * x      double precision complex array of length at least (1 + (n - 1) * abs (incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * y      double precision complex array of length at least (1 + (n - 1) * abs (incy)).
 * incy   storage spacing between elements of y. incy must not be zero.
 * AP     double precision complex array with at least ((n * (n + 1)) / 2) elements. If
 *        uplo == 'U' or 'u', the array AP contains the upper triangular part
 *        of the hermitian matrix A, packed sequentially, column by column;
 *        that is, if i <= j, then A[i,j] is stored is AP[i+(j*(j+1)/2)]. If
 *        uplo == 'L' or 'L', the array AP contains the lower triangular part
 *        of the hermitian matrix A, packed sequentially, column by column;
 *        that is, if i >= j, then A[i,j] is stored in AP[i+((2*n-j+1)*j)/2].
 *        The imaginary parts of the diagonal elements need not be set, they
 *        are assumed to be zero, and on exit they are set to zero.
 *
 * Output
 * ------
 * A      updated according to A = alpha*x*conjugate(transpose(y))
 *                               + conjugate(alpha)*y*conjugate(transpose(x))+A
 *
 * Reference: http://www.netlib.org/blas/zhpr2.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZhpr2Native
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jobject alpha, jobject x, jint incx, jobject y, jint incy, jobject AP)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZhpr2");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZhpr2");
        return;
    }
    if (AP == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'AP' is null for cublasZhpr2");
        return;
    }
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;
    cuDoubleComplex* nativeAP;
    cuDoubleComplex dobuleComplexAlpha;

    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);
    nativeAP = (cuDoubleComplex*)getPointer(env, AP);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZhpr2(%c, %d, [%lf,%lf], '%s', %d, '%s', %d, '%s')\n",
        uplo, n, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "x", incx, "y", incy, "AP");

    cublasZhpr2((char)uplo, n, dobuleComplexAlpha, nativeX, incx, nativeY, incy, nativeAP);
}




/**
 * <pre>
 * void cublasZher2 (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx,
 *                   const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)
 *
 * performs the hermitian rank 2 operation
 *
 *    A = alpha*x*conjugate(transpose(y)) + conjugate(alpha)*y*conjugate(transpose(x)) + A,
 *
 * where alpha is a double precision complex scalar, x and y are n element double
 * precision complex vector and A is an n by n hermitian matrix consisting of double
 * precision complex elements.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the lower
 *        triangular part of array A. If uplo == 'U' or 'u', then only the
 *        upper triangular part of A may be referenced and the lower triangular
 *        part of A is inferred. If uplo == 'L' or 'l', then only the lower
 *        triangular part of A may be referenced and the upper triangular part
 *        of A is inferred.
 * n      specifies the number of rows and columns of the matrix A. It must be
 *        at least zero.
 * alpha  double precision complex scalar multiplier applied to x * conjugate(transpose(y)) +
 *        y * conjugate(transpose(x)).
 * x      double precision array of length at least (1 + (n - 1) * abs (incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * y      double precision array of length at least (1 + (n - 1) * abs (incy)).
 * incy   storage spacing between elements of y. incy must not be zero.
 * A      double precision complex array of dimensions (lda, n). If uplo == 'U' or 'u',
 *        then A must contains the upper triangular part of a hermitian matrix,
 *        and the strictly lower triangular parts is not referenced. If uplo ==
 *        'L' or 'l', then A contains the lower triangular part of a hermitian
 *        matrix, and the strictly upper triangular part is not referenced.
 *        The imaginary parts of the diagonal elements need not be set,
 *        they are assumed to be zero, and on exit they are set to zero.
 *
 * lda    leading dimension of A. It must be at least max(1, n).
 *
 * Output
 * ------
 * A      updated according to A = alpha*x*conjugate(transpose(y))
 *                               + conjugate(alpha)*y*conjugate(transpose(x))+A
 *
 * Reference: http://www.netlib.org/blas/zher2.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZher2Native
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jobject alpha, jobject x, jint incx, jobject y, jint incy, jobject A, jint lda)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZher2");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZher2");
        return;
    }
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZher2");
        return;
    }
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;
    cuDoubleComplex* nativeA;
    cuDoubleComplex dobuleComplexAlpha;

    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);
    nativeA = (cuDoubleComplex*)getPointer(env, A);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZher2(%c, %d, [%lf,%lf], '%s', %d, '%s', %d, '%s', %d)\n",
        uplo, n, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "x", incx, "y", incy, "A", lda);

    cublasZher2((char)uplo, n, dobuleComplexAlpha, nativeX, incx, nativeY, incy, nativeA, lda);
}




/**
 * <pre>
 * void
 * cublasDsyr2k (char uplo, char trans, int n, int k, double alpha,
 *               const double *A, int lda, const double *B, int ldb,
 *               double beta, double *C, int ldc)
 *
 * performs one of the symmetric rank 2k operations
 *
 *    C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C, or
 *    C = alpha * transpose(A) * B + alpha * transpose(B) * A + beta * C.
 *
 * Alpha and beta are double precision scalars. C is an n x n symmetric matrix
 * consisting of double precision elements and stored in either lower or upper
 * storage mode. A and B are matrices consisting of double precision elements
 * with dimension of n x k in the first case, and k x n in the second case.
 *
 * Input
 * -----
 * uplo   specifies whether the symmetric matrix C is stored in upper or lower
 *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the symmetric matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the symmetric matrix is to be references,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * trans  specifies the operation to be performed. If trans == 'N' or 'n',
 *        C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C,
 *        If trans == 'T', 't', 'C', or 'c', C = alpha * transpose(A) * B +
 *        alpha * transpose(B) * A + beta * C.
 * n      specifies the number of rows and the number columns of matrix C. If
 *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
 *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
 *        n must be at least zero.
 * k      If trans == 'N' or 'n', k specifies the number of rows of matrix A.
 *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
 *        matrix A. k must be at least zero.
 * alpha  double precision scalar multiplier.
 * A      double precision array of dimensions (lda, ka), where ka is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array A must contain the matrix A,
 *        otherwise the leading k x n part of the array must contain the matrix
 *        A.
 * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
 *        least max(1, n). Otherwise lda must be at least max(1,k).
 * B      double precision array of dimensions (lda, kb), where kb is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array B must contain the matrix B,
 *        otherwise the leading k x n part of the array must contain the matrix
 *        B.
 * ldb    leading dimension of N. When trans == 'N' or 'n' then ldb must be at
 *        least max(1, n). Otherwise ldb must be at least max(1, k).
 * beta   double precision scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input.
 * C      double precision array of dimensions (ldc, n). If uplo == 'U' or 'u',
 *        the leading n x n triangular part of the array C must contain the
 *        upper triangular part of the symmetric matrix C and the strictly
 *        lower triangular part of C is not referenced. On exit, the upper
 *        triangular part of C is overwritten by the upper triangular part of
 *        the updated matrix. If uplo == 'L' or 'l', the leading n x n
 *        triangular part of the array C must contain the lower triangular part
 *        of the symmetric matrix C and the strictly upper triangular part of C
 *        is not referenced. On exit, the lower triangular part of C is
 *        overwritten by the lower triangular part of the updated matrix.
 * ldc    leading dimension of C. Must be at least max(1, n).
 *
 * Output
 * ------
 * C      updated according to alpha*A*transpose(B) + alpha*B*transpose(A) +
 *        beta*C or alpha*transpose(A)*B + alpha*transpose(B)*A + beta*C
 *
 * Reference:   http://www.netlib.org/blas/dsyr2k.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasDsyr2kNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jint n, jint k, jdouble alpha, jobject A, jint lda, jobject B, jint ldb, jdouble beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasDsyr2k");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasDsyr2k");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasDsyr2k");
        return;
    }
    double* nativeA;
    double* nativeB;
    double* nativeC;

    nativeA = (double*)getPointer(env, A);
    nativeB = (double*)getPointer(env, B);
    nativeC = (double*)getPointer(env, C);

    Logger::log(LOG_TRACE, "Executing cublasDsyr2k(%c, %c, %d, %d, %lf, '%s', %d, '%s', %d, %lf, '%s', %d)\n",
        uplo, trans, n, k, alpha, "A", lda, "B", ldb, beta, "C", ldc);

    cublasDsyr2k((char)uplo, (char)trans, n, k, alpha, nativeA, lda, nativeB, ldb, beta, nativeC, ldc);
}




/**
 * <pre>
 * void cublasZgemm (char transa, char transb, int m, int n, int k,
 *                   cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
 *                   const cuDoubleComplex *B, int ldb, cuDoubleComplex beta,
 *                   cuDoubleComplex *C, int ldc)
 *
 * zgemm performs one of the matrix-matrix operations
 *
 *    C = alpha * op(A) * op(B) + beta*C,
 *
 * where op(X) is one of
 *
 *    op(X) = X   or   op(X) = transpose  or  op(X) = conjg(transpose(X))
 *
 * alpha and beta are double-complex scalars, and A, B and C are matrices
 * consisting of double-complex elements, with op(A) an m x k matrix, op(B)
 * a k x n matrix and C an m x n matrix.
 *
 * Input
 * -----
 * transa specifies op(A). If transa == 'N' or 'n', op(A) = A. If transa ==
 *        'T' or 't', op(A) = transpose(A). If transa == 'C' or 'c', op(A) =
 *        conjg(transpose(A)).
 * transb specifies op(B). If transa == 'N' or 'n', op(B) = B. If transb ==
 *        'T' or 't', op(B) = transpose(B). If transb == 'C' or 'c', op(B) =
 *        conjg(transpose(B)).
 * m      number of rows of matrix op(A) and rows of matrix C. It must be at
 *        least zero.
 * n      number of columns of matrix op(B) and number of columns of C. It
 *        must be at least zero.
 * k      number of columns of matrix op(A) and number of rows of op(B). It
 *        must be at least zero.
 * alpha  double-complex scalar multiplier applied to op(A)op(B)
 * A      double-complex array of dimensions (lda, k) if transa ==  'N' or
 *        'n'), and of dimensions (lda, m) otherwise.
 * lda    leading dimension of A. When transa == 'N' or 'n', it must be at
 *        least max(1, m) and at least max(1, k) otherwise.
 * B      double-complex array of dimensions (ldb, n) if transb == 'N' or 'n',
 *        and of dimensions (ldb, k) otherwise
 * ldb    leading dimension of B. When transb == 'N' or 'n', it must be at
 *        least max(1, k) and at least max(1, n) otherwise.
 * beta   double-complex scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input.
 * C      double precision array of dimensions (ldc, n)
 * ldc    leading dimension of C. Must be at least max(1, m).
 *
 * Output
 * ------
 * C      updated according to C = alpha*op(A)*op(B) + beta*C
 *
 * Reference: http://www.netlib.org/blas/zgemm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if any of m, n, or k are < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZgemmNative
    (JNIEnv *env, jclass cls, jchar transa, jchar transb, jint m, jint n, jint k, jobject alpha, jobject A, jint lda, jobject B, jint ldb, jobject beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZgemm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasZgemm");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasZgemm");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeB;
    cuDoubleComplex* nativeC;
    cuDoubleComplex dobuleComplexAlpha;
    cuDoubleComplex dobuleComplexBeta;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeB = (cuDoubleComplex*)getPointer(env, B);
    nativeC = (cuDoubleComplex*)getPointer(env, C);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    dobuleComplexBeta.x = env->GetDoubleField(beta, cuDoubleComplex_x);
    dobuleComplexBeta.y = env->GetDoubleField(beta, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZgemm(%c, %c, %d, %d, %d, [%lf,%lf], '%s', %d, '%s', %d, [%lf,%lf], '%s', %d)\n",
        transa, transb, m, n, k, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "A", lda, "B", ldb, dobuleComplexBeta.x, dobuleComplexBeta.y, "C", ldc);

    cublasZgemm((char)transa, (char)transb, m, n, k, dobuleComplexAlpha, nativeA, lda, nativeB, ldb, dobuleComplexBeta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasZtrmm (char side, char uplo, char transa, char diag, int m, int n,
 *              cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
 *              int ldb)
 *
 * performs one of the matrix-matrix operations
 *
 *   B = alpha * op(A) * B,  or  B = alpha * B * op(A)
 *
 * where alpha is a double-precision complex scalar, B is an m x n matrix composed
 * of double precision complex elements, and A is a unit or non-unit, upper or lower,
 * triangular matrix composed of double precision complex elements. op(A) is one of
 *
 *   op(A) = A  , op(A) = transpose(A) or op(A) = conjugate(transpose(A))
 *
 * Matrices A and B are stored in column major format, and lda and ldb are
 * the leading dimensions of the two-dimensonials arrays that contain A and
 * B, respectively.
 *
 * Input
 * -----
 * side   specifies whether op(A) multiplies B from the left or right.
 *        If side = 'L' or 'l', then B = alpha * op(A) * B. If side =
 *        'R' or 'r', then B = alpha * B * op(A).
 * uplo   specifies whether the matrix A is an upper or lower triangular
 *        matrix. If uplo = 'U' or 'u', A is an upper triangular matrix.
 *        If uplo = 'L' or 'l', A is a lower triangular matrix.
 * transa specifies the form of op(A) to be used in the matrix
 *        multiplication. If transa = 'N' or 'n', then op(A) = A. If
 *        transa = 'T' or 't', then op(A) = transpose(A).
 *        If transa = 'C' or 'c', then op(A) = conjugate(transpose(A)).
 * diag   specifies whether or not A is unit triangular. If diag = 'U'
 *        or 'u', A is assumed to be unit triangular. If diag = 'N' or
 *        'n', A is not assumed to be unit triangular.
 * m      the number of rows of matrix B. m must be at least zero.
 * n      the number of columns of matrix B. n must be at least zero.
 * alpha  double precision complex scalar multiplier applied to op(A)*B, or
 *        B*op(A), respectively. If alpha is zero no accesses are made
 *        to matrix A, and no read accesses are made to matrix B.
 * A      double precision complex array of dimensions (lda, k). k = m if side =
 *        'L' or 'l', k = n if side = 'R' or 'r'. If uplo = 'U' or 'u'
 *        the leading k x k upper triangular part of the array A must
 *        contain the upper triangular matrix, and the strictly lower
 *        triangular part of A is not referenced. If uplo = 'L' or 'l'
 *        the leading k x k lower triangular part of the array A must
 *        contain the lower triangular matrix, and the strictly upper
 *        triangular part of A is not referenced. When diag = 'U' or 'u'
 *        the diagonal elements of A are no referenced and are assumed
 *        to be unity.
 * lda    leading dimension of A. When side = 'L' or 'l', it must be at
 *        least max(1,m) and at least max(1,n) otherwise
 * B      double precision complex array of dimensions (ldb, n). On entry, the
 *        leading m x n part of the array contains the matrix B. It is
 *        overwritten with the transformed matrix on exit.
 * ldb    leading dimension of B. It must be at least max (1, m).
 *
 * Output
 * ------
 * B      updated according to B = alpha * op(A) * B  or B = alpha * B * op(A)
 *
 * Reference: http://www.netlib.org/blas/ztrmm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZtrmmNative
    (JNIEnv *env, jclass cls, jchar side, jchar uplo, jchar transa, jchar diag, jint m, jint n, jobject alpha, jobject A, jint lda, jobject B, jint ldb)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZtrmm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasZtrmm");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeB;
    cuDoubleComplex dobuleComplexAlpha;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeB = (cuDoubleComplex*)getPointer(env, B);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZtrmm(%c, %c, %c, %c, %d, %d, [%lf,%lf], '%s', %d, '%s', %d)\n",
        side, uplo, transa, diag, m, n, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "A", lda, "B", ldb);

    cublasZtrmm((char)side, (char)uplo, (char)transa, (char)diag, m, n, dobuleComplexAlpha, nativeA, lda, nativeB, ldb);
}




/**
 * <pre>
 * cublasZgeru (int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx,
 *             const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)
 *
 * performs the symmetric rank 1 operation
 *
 *    A = alpha * x * transpose(y) + A,
 *
 * where alpha is a double precision complex scalar, x is an m element double
 * precision complex vector, y is an n element double precision complex vector, and A
 * is an m by n matrix consisting of double precision complex elements. Matrix A
 * is stored in column major format, and lda is the leading dimension of
 * the two-dimensional array used to store A.
 *
 * Input
 * -----
 * m      specifies the number of rows of the matrix A. It must be at least
 *        zero.
 * n      specifies the number of columns of the matrix A. It must be at
 *        least zero.
 * alpha  double precision complex scalar multiplier applied to x * transpose(y)
 * x      double precision complex array of length at least (1 + (m - 1) * abs(incx))
 * incx   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 * y      double precision complex array of length at least (1 + (n - 1) * abs(incy))
 * incy   specifies the storage spacing between elements of y. incy must not
 *        be zero.
 * A      double precision complex array of dimensions (lda, n).
 * lda    leading dimension of two-dimensional array used to store matrix A
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * transpose(y) + A
 *
 * Reference: http://www.netlib.org/blas/zgeru.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m < 0, n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZgeruNative
    (JNIEnv *env, jclass cls, jint m, jint n, jobject alpha, jobject x, jint incx, jobject y, jint incy, jobject A, jint lda)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZgeru");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZgeru");
        return;
    }
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZgeru");
        return;
    }
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;
    cuDoubleComplex* nativeA;
    cuDoubleComplex dobuleComplexAlpha;

    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);
    nativeA = (cuDoubleComplex*)getPointer(env, A);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZgeru(%d, %d, [%lf,%lf], '%s', %d, '%s', %d, '%s', %d)\n",
        m, n, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "x", incx, "y", incy, "A", lda);

    cublasZgeru(m, n, dobuleComplexAlpha, nativeX, incx, nativeY, incy, nativeA, lda);
}




/**
 * <pre>
 * cublasZgerc (int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx,
 *             const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)
 *
 * performs the symmetric rank 1 operation
 *
 *    A = alpha * x * conjugate(transpose(y)) + A,
 *
 * where alpha is a double precision complex scalar, x is an m element double
 * precision complex vector, y is an n element double precision complex vector, and A
 * is an m by n matrix consisting of double precision complex elements. Matrix A
 * is stored in column major format, and lda is the leading dimension of
 * the two-dimensional array used to store A.
 *
 * Input
 * -----
 * m      specifies the number of rows of the matrix A. It must be at least
 *        zero.
 * n      specifies the number of columns of the matrix A. It must be at
 *        least zero.
 * alpha  double precision complex scalar multiplier applied to x * conjugate(transpose(y))
 * x      double precision array of length at least (1 + (m - 1) * abs(incx))
 * incx   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 * y      double precision complex array of length at least (1 + (n - 1) * abs(incy))
 * incy   specifies the storage spacing between elements of y. incy must not
 *        be zero.
 * A      double precision complex array of dimensions (lda, n).
 * lda    leading dimension of two-dimensional array used to store matrix A
 *
 * Output
 * ------
 * A      updated according to A = alpha * x * conjugate(transpose(y)) + A
 *
 * Reference: http://www.netlib.org/blas/zgerc.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m < 0, n < 0, incx == 0, incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZgercNative
    (JNIEnv *env, jclass cls, jint m, jint n, jobject alpha, jobject x, jint incx, jobject y, jint incy, jobject A, jint lda)
{
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZgerc");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZgerc");
        return;
    }
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZgerc");
        return;
    }
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;
    cuDoubleComplex* nativeA;
    cuDoubleComplex dobuleComplexAlpha;

    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);
    nativeA = (cuDoubleComplex*)getPointer(env, A);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZgerc(%d, %d, [%lf,%lf], '%s', %d, '%s', %d, '%s', %d)\n",
        m, n, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "x", incx, "y", incy, "A", lda);

    cublasZgerc(m, n, dobuleComplexAlpha, nativeX, incx, nativeY, incy, nativeA, lda);
}




/**
 * <pre>
 * void
 * cublasZherk (char uplo, char trans, int n, int k, double alpha,
 *              const cuDoubleComplex *A, int lda, double beta, cuDoubleComplex *C, int ldc)
 *
 * performs one of the hermitian rank k operations
 *
 *   C = alpha * A * conjugate(transpose(A)) + beta * C, or
 *   C = alpha * conjugate(transpose(A)) * A + beta * C.
 *
 * Alpha and beta are double precision scalars. C is an n x n hermitian matrix
 * consisting of double precision complex elements and stored in either lower or
 * upper storage mode. A is a matrix consisting of double precision complex elements
 * with dimension of n x k in the first case, and k x n in the second case.
 *
 * Input
 * -----
 * uplo   specifies whether the hermitian matrix C is stored in upper or lower
 *        storage mode as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the hermitian matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the hermitian matrix is to be referenced,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * trans  specifies the operation to be performed. If trans == 'N' or 'n', C =
 *        alpha * A * conjugate(transpose(A)) + beta * C. If trans == 'T', 't', 'C', or 'c',
 *        C = alpha * conjugate(transpose(A)) * A + beta * C.
 * n      specifies the number of rows and the number columns of matrix C. If
 *        trans == 'N' or 'n', n specifies the number of rows of matrix A. If
 *        trans == 'T', 't', 'C', or 'c', n specifies the columns of matrix A.
 *        n must be at least zero.
 * k      If trans == 'N' or 'n', k specifies the number of columns of matrix A.
 *        If trans == 'T', 't', 'C', or 'c', k specifies the number of rows of
 *        matrix A. k must be at least zero.
 * alpha  double precision scalar multiplier applied to A * conjugate(transpose(A)) or
 *        conjugate(transpose(A)) * A.
 * A      double precision complex array of dimensions (lda, ka), where ka is k when
 *        trans == 'N' or 'n', and is n otherwise. When trans == 'N' or 'n',
 *        the leading n x k part of array A must contain the matrix A,
 *        otherwise the leading k x n part of the array must contains the
 *        matrix A.
 * lda    leading dimension of A. When trans == 'N' or 'n' then lda must be at
 *        least max(1, n). Otherwise lda must be at least max(1, k).
 * beta   double precision scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input
 * C      double precision complex array of dimensions (ldc, n). If uplo = 'U' or 'u',
 *        the leading n x n triangular part of the array C must contain the
 *        upper triangular part of the hermitian matrix C and the strictly
 *        lower triangular part of C is not referenced. On exit, the upper
 *        triangular part of C is overwritten by the upper triangular part of
 *        the updated matrix. If uplo = 'L' or 'l', the leading n x n
 *        triangular part of the array C must contain the lower triangular part
 *        of the hermitian matrix C and the strictly upper triangular part of C
 *        is not referenced. On exit, the lower triangular part of C is
 *        overwritten by the lower triangular part of the updated matrix.
 *        The imaginary parts of the diagonal elements need
 *        not be set,  they are assumed to be zero,  and on exit they
 *        are set to zero.
 * ldc    leading dimension of C. It must be at least max(1, n).
 *
 * Output
 * ------
 * C      updated according to C = alpha * A * conjugate(transpose(A)) + beta * C, or C =
 *        alpha * conjugate(transpose(A)) * A + beta * C
 *
 * Reference: http://www.netlib.org/blas/zherk.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n < 0 or k < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZherkNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jint n, jint k, jdouble alpha, jobject A, jint lda, jdouble beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZherk");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasZherk");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeC;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeC = (cuDoubleComplex*)getPointer(env, C);

    Logger::log(LOG_TRACE, "Executing cublasZherk(%c, %c, %d, %d, %lf, '%s', %d, %lf, '%s', %d)\n",
        uplo, trans, n, k, alpha, "A", lda, beta, "C", ldc);

    cublasZherk((char)uplo, (char)trans, n, k, alpha, nativeA, lda, beta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasZhemm (char side, char uplo, int m, int n, cuDoubleComplex alpha,
 *              const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
 *              cuDoubleComplex beta, cuDoubleComplex *C, int ldc);
 *
 * performs one of the matrix-matrix operations
 *
 *   C = alpha * A * B + beta * C, or
 *   C = alpha * B * A + beta * C,
 *
 * where alpha and beta are double precision complex scalars, A is a hermitian matrix
 * consisting of double precision complex elements and stored in either lower or upper
 * storage mode, and B and C are m x n matrices consisting of double precision
 * complex elements.
 *
 * Input
 * -----
 * side   specifies whether the hermitian matrix A appears on the left side
 *        hand side or right hand side of matrix B, as follows. If side == 'L'
 *        or 'l', then C = alpha * A * B + beta * C. If side = 'R' or 'r',
 *        then C = alpha * B * A + beta * C.
 * uplo   specifies whether the hermitian matrix A is stored in upper or lower
 *        storage mode, as follows. If uplo == 'U' or 'u', only the upper
 *        triangular part of the hermitian matrix is to be referenced, and the
 *        elements of the strictly lower triangular part are to be infered from
 *        those in the upper triangular part. If uplo == 'L' or 'l', only the
 *        lower triangular part of the hermitian matrix is to be referenced,
 *        and the elements of the strictly upper triangular part are to be
 *        infered from those in the lower triangular part.
 * m      specifies the number of rows of the matrix C, and the number of rows
 *        of matrix B. It also specifies the dimensions of hermitian matrix A
 *        when side == 'L' or 'l'. m must be at least zero.
 * n      specifies the number of columns of the matrix C, and the number of
 *        columns of matrix B. It also specifies the dimensions of hermitian
 *        matrix A when side == 'R' or 'r'. n must be at least zero.
 * alpha  double precision scalar multiplier applied to A * B, or B * A
 * A      double precision complex array of dimensions (lda, ka), where ka is m when
 *        side == 'L' or 'l' and is n otherwise. If side == 'L' or 'l' the
 *        leading m x m part of array A must contain the hermitian matrix,
 *        such that when uplo == 'U' or 'u', the leading m x m part stores the
 *        upper triangular part of the hermitian matrix, and the strictly lower
 *        triangular part of A is not referenced, and when uplo == 'U' or 'u',
 *        the leading m x m part stores the lower triangular part of the
 *        hermitian matrix and the strictly upper triangular part is not
 *        referenced. If side == 'R' or 'r' the leading n x n part of array A
 *        must contain the hermitian matrix, such that when uplo == 'U' or 'u',
 *        the leading n x n part stores the upper triangular part of the
 *        hermitian matrix and the strictly lower triangular part of A is not
 *        referenced, and when uplo == 'U' or 'u', the leading n x n part
 *        stores the lower triangular part of the hermitian matrix and the
 *        strictly upper triangular part is not referenced. The imaginary parts
 *        of the diagonal elements need not be set, they are assumed to be zero.
 *
 * lda    leading dimension of A. When side == 'L' or 'l', it must be at least
 *        max(1, m) and at least max(1, n) otherwise.
 * B      double precision complex array of dimensions (ldb, n). On entry, the leading
 *        m x n part of the array contains the matrix B.
 * ldb    leading dimension of B. It must be at least max (1, m).
 * beta   double precision complex scalar multiplier applied to C. If beta is zero, C
 *        does not have to be a valid input
 * C      double precision complex array of dimensions (ldc, n)
 * ldc    leading dimension of C. Must be at least max(1, m)
 *
 * Output
 * ------
 * C      updated according to C = alpha * A * B + beta * C, or C = alpha *
 *        B * A + beta * C
 *
 * Reference: http://www.netlib.org/blas/zhemm.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if m or n are < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZhemmNative
    (JNIEnv *env, jclass cls, jchar side, jchar uplo, jint m, jint n, jobject alpha, jobject A, jint lda, jobject B, jint ldb, jobject beta, jobject C, jint ldc)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZhemm");
        return;
    }
    if (B == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'B' is null for cublasZhemm");
        return;
    }
    if (C == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'C' is null for cublasZhemm");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeB;
    cuDoubleComplex* nativeC;
    cuDoubleComplex dobuleComplexAlpha;
    cuDoubleComplex dobuleComplexBeta;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeB = (cuDoubleComplex*)getPointer(env, B);
    nativeC = (cuDoubleComplex*)getPointer(env, C);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    dobuleComplexBeta.x = env->GetDoubleField(beta, cuDoubleComplex_x);
    dobuleComplexBeta.y = env->GetDoubleField(beta, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZhemm(%c, %c, %d, %d, [%lf,%lf], '%s', %d, '%s', %d, [%lf,%lf], '%s', %d)\n",
        side, uplo, m, n, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "A", lda, "B", ldb, dobuleComplexBeta.x, dobuleComplexBeta.y, "C", ldc);

    cublasZhemm((char)side, (char)uplo, m, n, dobuleComplexAlpha, nativeA, lda, nativeB, ldb, dobuleComplexBeta, nativeC, ldc);
}




/**
 * <pre>
 * void
 * cublasZtrsv (char uplo, char trans, char diag, int n, const cuDoubleComplex *A,
 *              int lda, cuDoubleComplex *x, int incx)
 *
 * solves a system of equations op(A) * x = b, where op(A) is either A,
 * transpose(A) or conjugate(transpose(A)). b and x are double precision
 * complex vectors consisting of n elements, and A is an n x n matrix
 * composed of a unit or non-unit, upper or lower triangular matrix.
 * Matrix A is stored in column major format, and lda is the leading
 * dimension of the two-dimensional array containing A.
 *
 * No test for singularity or near-singularity is included in this function.
 * Such tests must be performed before calling this function.
 *
 * Input
 * -----
 * uplo   specifies whether the matrix data is stored in the upper or the
 *        lower triangular part of array A. If uplo = 'U' or 'u', then only
 *        the upper triangular part of A may be referenced. If uplo = 'L' or
 *        'l', then only the lower triangular part of A may be referenced.
 * trans  specifies op(A). If transa = 'n' or 'N', op(A) = A. If transa = 't',
 *        'T', 'c', or 'C', op(A) = transpose(A)
 * diag   specifies whether or not A is a unit triangular matrix like so:
 *        if diag = 'U' or 'u', A is assumed to be unit triangular. If
 *        diag = 'N' or 'n', then A is not assumed to be unit triangular.
 * n      specifies the number of rows and columns of the matrix A. It
 *        must be at least 0.
 * A      is a double precision complex array of dimensions (lda, n). If uplo = 'U'
 *        or 'u', then A must contains the upper triangular part of a symmetric
 *        matrix, and the strictly lower triangular parts is not referenced.
 *        If uplo = 'L' or 'l', then A contains the lower triangular part of
 *        a symmetric matrix, and the strictly upper triangular part is not
 *        referenced.
 * lda    is the leading dimension of the two-dimensional array containing A.
 *        lda must be at least max(1, n).
 * x      double precision complex array of length at least (1 + (n - 1) * abs(incx)).
 *        On entry, x contains the n element right-hand side vector b. On exit,
 *        it is overwritten with the solution vector x.
 * incx   specifies the storage spacing between elements of x. incx must not
 *        be zero.
 *
 * Output
 * ------
 * x      updated to contain the solution vector x that solves op(A) * x = b.
 *
 * Reference: http://www.netlib.org/blas/ztrsv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx == 0 or if n < 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZtrsvNative
    (JNIEnv *env, jclass cls, jchar uplo, jchar trans, jchar diag, jint n, jobject A, jint lda, jobject x, jint incx)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZtrsv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZtrsv");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeX;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeX = (cuDoubleComplex*)getPointer(env, x);

    Logger::log(LOG_TRACE, "Executing cublasZtrsv(%c, %c, %c, %d, '%s', %d, '%s', %d)\n",
        uplo, trans, diag, n, "A", lda, "x", incx);

    cublasZtrsv((char)uplo, (char)trans, (char)diag, n, nativeA, lda, nativeX, incx);
}




/**
 * <pre>
 * void
 * cublasZhbmv (char uplo, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
 *              const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy)
 *
 * performs the matrix-vector operation
 *
 *     y := alpha*A*x + beta*y
 *
 * alpha and beta are double precision complex scalars. x and y are double precision
 * complex vectors with n elements. A is an n by n hermitian band matrix consisting
 * of double precision complex elements, with k super-diagonals and the same number
 * of subdiagonals.
 *
 * Input
 * -----
 * uplo   specifies whether the upper or lower triangular part of the hermitian
 *        band matrix A is being supplied. If uplo == 'U' or 'u', the upper
 *        triangular part is being supplied. If uplo == 'L' or 'l', the lower
 *        triangular part is being supplied.
 * n      specifies the number of rows and the number of columns of the
 *        hermitian matrix A. n must be at least zero.
 * k      specifies the number of super-diagonals of matrix A. Since the matrix
 *        is hermitian, this is also the number of sub-diagonals. k must be at
 *        least zero.
 * alpha  double precision complex scalar multiplier applied to A*x.
 * A      double precision complex array of dimensions (lda, n). When uplo == 'U' or
 *        'u', the leading (k + 1) x n part of array A must contain the upper
 *        triangular band of the hermitian matrix, supplied column by column,
 *        with the leading diagonal of the matrix in row (k+1) of the array,
 *        the first super-diagonal starting at position 2 in row k, and so on.
 *        The top left k x k triangle of the array A is not referenced. When
 *        uplo == 'L' or 'l', the leading (k + 1) x n part of the array A must
 *        contain the lower triangular band part of the hermitian matrix,
 *        supplied column by column, with the leading diagonal of the matrix in
 *        row 1 of the array, the first sub-diagonal starting at position 1 in
 *        row 2, and so on. The bottom right k x k triangle of the array A is
 *        not referenced. The imaginary parts of the diagonal elements need
 *        not be set, they are assumed to be zero.
 * lda    leading dimension of A. lda must be at least (k + 1).
 * x      double precision complex array of length at least (1 + (n - 1) * abs(incx)).
 * incx   storage spacing between elements of x. incx must not be zero.
 * beta   double precision complex scalar multiplier applied to vector y. If beta is
 *        zero, y is not read.
 * y      double precision complex array of length at least (1 + (n - 1) * abs(incy)).
 *        If beta is zero, y is not read.
 * incy   storage spacing between elements of y. incy must not be zero.
 *
 * Output
 * ------
 * y      updated according to alpha*A*x + beta*y
 *
 * Reference: http://www.netlib.org/blas/zhbmv.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if k or n < 0, or if incx or incy == 0
 * CUBLAS_STATUS_ARCH_MISMATCH    if invoked on device without DP support
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 * </pre>
 */
JNIEXPORT void JNICALL Java_jcuda_jcublas_JCublas_cublasZhbmvNative
    (JNIEnv *env, jclass cls, jchar uplo, jint n, jint k, jobject alpha, jobject A, jint lda, jobject x, jint incx, jobject beta, jobject y, jint incy)
{
    if (A == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'A' is null for cublasZhbmv");
        return;
    }
    if (x == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'x' is null for cublasZhbmv");
        return;
    }
    if (y == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'y' is null for cublasZhbmv");
        return;
    }
    cuDoubleComplex* nativeA;
    cuDoubleComplex* nativeX;
    cuDoubleComplex* nativeY;
    cuDoubleComplex dobuleComplexAlpha;
    cuDoubleComplex dobuleComplexBeta;

    nativeA = (cuDoubleComplex*)getPointer(env, A);
    nativeX = (cuDoubleComplex*)getPointer(env, x);
    nativeY = (cuDoubleComplex*)getPointer(env, y);

    dobuleComplexAlpha.x = env->GetDoubleField(alpha, cuDoubleComplex_x);
    dobuleComplexAlpha.y = env->GetDoubleField(alpha, cuDoubleComplex_y);
    dobuleComplexBeta.x = env->GetDoubleField(beta, cuDoubleComplex_x);
    dobuleComplexBeta.y = env->GetDoubleField(beta, cuDoubleComplex_y);
    Logger::log(LOG_TRACE, "Executing cublasZhbmv(%c, %d, %d, [%lf,%lf], '%s', %d, '%s', %d, [%lf,%lf], '%s', %d)\n",
        uplo, n, k, dobuleComplexAlpha.x, dobuleComplexAlpha.y, "A", lda, "x", incx, dobuleComplexBeta.x, dobuleComplexBeta.y, "y", incy);

    cublasZhbmv((char)uplo, n, k, dobuleComplexAlpha, nativeA, lda, nativeX, incx, dobuleComplexBeta, nativeY, incy);
}




