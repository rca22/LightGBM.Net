using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace LightGBMNet.Train
{
    /// <summary>
    /// Definition of ReduceScatter funtion
    /// </summary>
    public delegate void ReduceScatterFunction([MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)]byte[] input, int inputSize, int typeSize,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 5)]int[] blockStart, [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 5)]int[] blockLen, int numBlock,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 7)]byte[] output, int outputSize,
        IntPtr reducer);

    /// <summary>
    /// Definition of Allgather funtion
    /// </summary>
    public delegate void AllGatherFunction([MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)]byte[] input, int inputSize,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 4)]int[] blockStart, [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 4)]int[] blockLen, int numBlock,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 6)]byte[] output, int outputSize);

    /// <summary>
    /// Wrapper of the c interfaces of LightGBM.
    /// Refer to https://github.com/Microsoft/LightGBM/blob/master/include/LightGBM/c_api.h to get the details.
    /// </summary>
    internal static class PInvoke
    {
        /// <summary>
        /// Need to specify an assumption on the feature name size.
        /// </summary>
        public static int MAX_PREALLOCATED_STRING_LENGTH = 100;

        internal enum CApiDType : int
        {
            Float32 = 0,
            Float64 = 1,
            Int32   = 2,
            Int64   = 3
        }

        internal enum CApiPredictType : int
        {
            Normal     = 0,
            RawScore   = 1,
            LeafIndex  = 2,
            Contrib    = 3
        }

        internal enum CApiFeatureImportanceType : int
        {
            Split = 0,
            Gain  = 1
        }

        //public enum FieldName : int
        //{
        //    Label,
        //    Weight,
        //    Group,
        //    GroupId
        //}

        //private static string GetFieldNameString(FieldName fld)
        //{
        //    switch(fld)
        //    {
        //        case FieldName.Label:   return "label";
        //        case FieldName.Weight:  return "weight";
        //        case FieldName.Group:   return "group";
        //        case FieldName.GroupId: return "group_id";
        //        default:
        //            throw new ArgumentException("FieldName not recognised", "fld");
        //    }
        //}

        private const string DllName = @"x64\lib_lightgbm";

        // The functions below are presented for simplicity in the order in which they appear in the file
        // https://github.com/Microsoft/LightGBM/blob/master/include/LightGBM/c_api.h

        #region API

        /// <summary>
        /// get string message of the last error all function in this file will return 0 when succeed
        /// and -1 when an error occured
        /// </summary>
        /// <returns>error information</returns>
        [DllImport(DllName, EntryPoint = "LGBM_GetLastError", CallingConvention = CallingConvention.StdCall)]
        public static extern IntPtr GetLastError();

        /// <summary>
        /// load data set from file like the command_line LightGBM
        /// </summary>
        /// <param name="filename">the name of the file</param>
        /// <param name="parameters">additional parameters</param>
        /// <param name="reference">used to align bin mapper with other dataset, nullptr means don't use</param>
        /// <param name="ret">a loaded dataset</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetCreateFromFile", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetCreateFromFile(
            [MarshalAs(UnmanagedType.LPStr)]string filename,
            [MarshalAs(UnmanagedType.LPStr)]string parameters,
            IntPtr reference,
            ref IntPtr ret);

        /// <summary>
        /// create a empty dataset by sampling data.
        /// </summary>
        /// <param name="sampleValuePerColumn">sampled data, grouped by the column.</param>
        /// <param name="sampleIndicesPerColumn">indices of sampled data.</param>
        /// <param name="numCol">number columns</param>
        /// <param name="sampleNonZeroCntPerColumn">Size of each sampling column</param>
        /// <param name="numSampleRow">Number of sampled rows</param>
        /// <param name="numTotalRow">number of total rows</param>
        /// <param name="parameters">additional parameters</param>
        /// <param name="ret">created dataset</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetCreateFromSampledColumn", CallingConvention = CallingConvention.StdCall)]
        public static extern unsafe int DatasetCreateFromSampledColumn(IntPtr sampleValuePerColumn,
            IntPtr sampleIndicesPerColumn,
            int numCol,
            int *sampleNonZeroCntPerColumn,
            int numSampleRow,
            int numTotalRow,
            [MarshalAs(UnmanagedType.LPStr)]string parameters,
            ref IntPtr ret);

        /// <summary>
        /// create a empty dataset by reference Dataset
        /// </summary>
        /// <param name="reference">used to align bin mapper</param>
        /// <param name="numTotalRow">number of total rows</param>
        /// <param name="ret">created dataset</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetCreateByReference", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetCreateByReference(
            IntPtr reference,
            long numTotalRow,
            ref IntPtr ret);

        /// <summary>
        /// push data to existing dataset, if nrow + start_row == num_total_row, will call dataset-&gt;FinishLoad
        /// </summary>
        /// <param name="dataset">handle of dataset</param>
        /// <param name="data">pointer to the data space</param>
        /// <param name="dataType">type of data pointer</param>
        /// <param name="numRow">number of rows</param>
        /// <param name="numCol">number columns</param>
        /// <param name="startRowIdx">row start index</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetPushRows", CallingConvention = CallingConvention.StdCall)]
        private static extern unsafe int DatasetPushRows(IntPtr dataset,
            float *data,
            CApiDType dataType,
            int numRow,
            int numCol,
            int startRowIdx);

        public static unsafe int DatasetPushRows(IntPtr dataset,
            float[] data,
            int numRow,
            int numCol,
            int startRowIdx)
        {
            fixed (float *dataPtr = data)
                return DatasetPushRows(dataset, dataPtr, CApiDType.Float32, numRow, numCol, startRowIdx);
        }

        /// <summary>
        /// push data to existing dataset, if nrow + start_row == num_total_row, will call dataset-&gt;FinishLoad
        /// </summary>
        /// <param name="dataset">handle of dataset</param>
        /// <param name="indPtr">pointer to row headers</param>
        /// <param name="indPtrType">type of indptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64</param>
        /// <param name="indices">findex</param>
        /// <param name="data">fvalue</param>
        /// <param name="dataType">type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64</param>
        /// <param name="nIndPtr">number of rows in the matrix + 1</param>
        /// <param name="numElem">number of nonzero elements in the matrix</param>
        /// <param name="numCol">number of columns</param>
        /// <param name="startRowIdx">row start index</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetPushRowsByCSR", CallingConvention = CallingConvention.StdCall)]
        private static extern unsafe int DatasetPushRowsByCsr(IntPtr dataset,
            int *indPtr,
            CApiDType indPtrType,
            int *indices,
            float *data,
            CApiDType dataType,
            long nIndPtr,
            long numElem,
            long numCol,
            long startRowIdx);

        public static unsafe int DatasetPushRowsByCsr(IntPtr dataset,
            int [] indPtr,
            int [] indices,
            float [] data,
            long nIndPtr,
            long numElem,
            long numCol,
            long startRowIdx)
        {
            fixed (int* indPtr2 = indPtr, indices2 = indices)
            fixed (float* data2 = data)
                return DatasetPushRowsByCsr(dataset,
                indPtr2, CApiDType.Int32,
                indices2, data2, CApiDType.Float32,
                nIndPtr, numElem, numCol, startRowIdx);
        }

        /// <summary>
        /// create a dataset from CSR format
        /// </summary>
        /// <param name="indPtr">pointer to row headers</param>
        /// <param name="indPtrType">type of indptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64</param>
        /// <param name="indices">findex</param>
        /// <param name="data">fvalue</param>
        /// <param name="dataType">type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64</param>
        /// <param name="nIndPtr">number of rows in the matrix + 1</param>
        /// <param name="numElem">number of nonzero elements in the matrix</param>
        /// <param name="numCol">number of columns</param>
        /// <param name="parameters">additional parameters</param>
        /// <param name="reference">reference used to align bin mapper with other dataset, nullptr means don't used</param>
        /// <param name="ret">created dataset</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetCreateFromCSR", CallingConvention = CallingConvention.StdCall)]
        private static extern unsafe int DatasetCreateFromCsr(
            int *indPtr,
            CApiDType indPtrType,
            int *indices,
            float *data,
            CApiDType dataType,
            long nIndPtr,
            long numElem,
            long numCol,
            [MarshalAs(UnmanagedType.LPStr)]string parameters,
            IntPtr reference,
            ref IntPtr ret);

        public static unsafe int DatasetCreateFromCsr(
            int *indPtr,
            int *indices,
            float *data,
            long nIndPtr,
            long numElem,
            long numCol,
            string parameters,
            IntPtr reference,
            ref IntPtr ret)
        {
            return DatasetCreateFromCsr(
                indPtr, CApiDType.Int32,
                indices, data, CApiDType.Float32,
                nIndPtr, numElem, numCol, parameters, reference, ref ret);
        }

        /// <summary>
        /// create a dataset from CSC format
        /// </summary>
        /// <param name="colPtr">pointer to col headers</param>
        /// <param name="colPtrType">type of col_ptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64</param>
        /// <param name="indices">findex</param>
        /// <param name="data">fvalue</param>
        /// <param name="dataType">type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64</param>
        /// <param name="nColPtr">number of cols in the matrix + 1</param>
        /// <param name="nElem">number of nonzero elements in the matrix</param>
        /// <param name="numRow">number of rows</param>
        /// <param name="parameters">additional parameters</param>
        /// <param name="reference">used to align bin mapper with other dataset, nullptr means don't used</param>
        /// <param name="ret">created dataset</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetCreateFromCSC", CallingConvention = CallingConvention.StdCall)]
        private static extern unsafe int DatasetCreateFromCsc(
            int *colPtr,
            CApiDType colPtrType,
            int *indices,
            float *data,
            CApiDType dataType,
            long nColPtr,
            long nElem,
            long numRow,
            [MarshalAs(UnmanagedType.LPStr)]string parameters,
            IntPtr reference,
            ref IntPtr ret);

        public static unsafe int DatasetCreateFromCsc(
            int *colPtr,
            int *indices,
            float *data,
            long nColPtr,
            long nElem,
            long numRow,
            string parameters,
            IntPtr reference,
            ref IntPtr ret)
        {
            return DatasetCreateFromCsc(
                colPtr, CApiDType.Int32,
                indices,
                data, CApiDType.Float32,
                nColPtr, nElem, numRow, parameters, reference, ref ret);
        }

        /// <summary>
        /// create dataset from dense matrix
        /// </summary>
        /// <param name="data">data pointer to the data space</param>
        /// <param name="dataType">type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64</param>
        /// <param name="nRow">number of rows</param>
        /// <param name="nCol">number columns</param>
        /// <param name="isRowMajor">1 for row major, 0 for column major</param>
        /// <param name="parameters">additional parameters</param>
        /// <param name="reference">used to align bin mapper with other dataset, nullptr means don't used</param>
        /// <param name="ret">created dataset</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetCreateFromMat", CallingConvention = CallingConvention.StdCall)]
        private static extern unsafe int DatasetCreateFromMat(
            float *data,
            CApiDType dataType,
            int nRow,
            int nCol,
            int isRowMajor,
            [MarshalAs(UnmanagedType.LPStr)]string parameters,
            IntPtr reference,
            ref IntPtr ret);

        public static unsafe int DatasetCreateFromMat(
            float *data,
            int nRow,
            int nCol,
            bool isRowMajor,
            string parameters,
            IntPtr reference,
            ref IntPtr ret)
        {
            return DatasetCreateFromMat(
                data, CApiDType.Float32,
                nRow, nCol,
                (isRowMajor ? 1 : 0),
                parameters, reference, ref ret);
        }

        /// <summary>
        /// create dataset from array of dense matrices
        /// </summary>
        /// <param name="nMat">Number of matrices</param>
        /// <param name="data">pointer to the data matrices</param>
        /// <param name="dataType">type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64</param>
        /// <param name="nrow">number of rows in each matrix</param>
        /// <param name="ncol">number columns</param>
        /// <param name="isRowMajor">1 for row major, 0 for column major</param>
        /// <param name="parameters">additional parameters</param>
        /// <param name="reference">used to align bin mapper with other dataset, nullptr means don't used</param>
        /// <param name="ret">created dataset</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetCreateFromMats", CallingConvention = CallingConvention.StdCall)]
        private static extern unsafe int DatasetCreateFromMats(
            int nMat,
            float** data,
            CApiDType dataType,
            int* nrow,
            int ncol,
            int isRowMajor,
            [MarshalAs(UnmanagedType.LPStr)]string parameters,
            IntPtr reference,
            ref IntPtr ret);

        public static unsafe int DatasetCreateFromMats(
            int nMat,
            float** data,
            int* nRow,
            int nCol,
            bool isRowMajor,
            string parameters,
            IntPtr reference,
            ref IntPtr ret)
        {
            return DatasetCreateFromMats(
                nMat,
                data,
                CApiDType.Float32,
                nRow,
                nCol,
                (isRowMajor ? 1 : 0),
                parameters,
                reference,
                ref ret);
        }

        /// <summary>
        /// Create subset of a data
        /// </summary>
        /// <param name="handle">handle of full dataset</param>
        /// <param name="usedRowIndices">Indices used in subset</param>
        /// <param name="numUsedRowIndices">len of used_row_indices</param>
        /// <param name="parameters">additional parameters</param>
        /// <param name="ret">subset of data</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetGetSubset", CallingConvention = CallingConvention.StdCall)]
        public static extern unsafe int DatasetGetSubset(
            IntPtr handle,
            int *usedRowIndices,
            int numUsedRowIndices,
            [MarshalAs(UnmanagedType.LPStr)]string parameters,
            ref IntPtr ret);

        /// <summary>
        /// save feature names to Dataset
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="featureNames">feature names</param>
        /// <param name="numFeatureNames">number of feature names</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetSetFeatureNames", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetSetFeatureNames(
            IntPtr handle,
            IntPtr[] featureNames, 
            int numFeatureNames);

        /// <summary>
        /// get feature names of Dataset
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="len">Number of ``char*`` pointers stored at ``out_strs``. If smaller than the max size, only this many strings are copied</param>
        /// <param name="numFeatureNames">number of feature names</param>
        /// <param name="buffer_len"> Size of pre-allocated strings. Content is copied up to ``buffer_len - 1`` and null-terminated</param>
        /// <param name="out_buffer_len"> String sizes required to do the full string copies</param>
        /// <param name="featureNames">feature names, should pre-allocate memory</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetGetFeatureNames", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetGetFeatureNames(
            IntPtr handle,
            int len,
            ref int numFeatureNames,
            long buffer_len,
            ref long out_buffer_len,
            IntPtr[] featureNames
            );

        /// <summary>
        /// free space for dataset
        /// </summary>
        /// <param name="handle"></param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetFree", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetFree(IntPtr handle);

        /// <summary>
        /// save dateset to binary file
        /// </summary>
        /// <param name="handle">a instance of dataset</param>
        /// <param name="fileName">file name</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetSaveBinary", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetSaveBinary(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)] string fileName);

        /// <summary>
        ///  set vector to a content in info
        ///  Note: group and group only work for C_API_DTYPE_INT32 label and weight only work for C_API_DTYPE_FLOAT32
        /// </summary>
        /// <param name="handle">instance of dataset</param>
        /// <param name="field">field name, can be label, weight, group, group_id</param>
        /// <param name="array">pointer to vector</param>
        /// <param name="len">number of element in field_data</param>
        /// <param name="type">C_API_DTYPE_FLOAT32 or C_API_DTYPE_INT32</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetSetField", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetSetField(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)]string field,
            IntPtr array,
            int len,
            CApiDType type);

        /// <summary>
        /// get info vector from dataset
        /// </summary>
        /// <param name="handle">instance of data matrix</param>
        /// <param name="fieldName">field name</param>
        /// <param name="outLen"> used to set result length</param>
        /// <param name="outPtr">pointer to the result</param>
        /// <param name="type">C_API_DTYPE_FLOAT32 or C_API_DTYPE_INT32</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetGetField", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetGetField(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)]string fieldName,
            ref int outLen,
            ref IntPtr outPtr,
            ref CApiDType type);

        /// <summary>
        /// get number of data.
        /// </summary>
        /// <param name="handle">handle to the dataset</param>
        /// <param name="res">The address to hold number of data</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetGetNumData", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetGetNumData(IntPtr handle, ref int res);

        /// <summary>
        /// get number of features
        /// </summary>
        /// <param name="handle">handle to the dataset</param>
        /// <param name="res">The output of number of features</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_DatasetGetNumFeature", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetGetNumFeature(IntPtr handle, ref int res);

        /// <summary>
        /// create an new boosting learner
        /// </summary>
        /// <param name="trainset">training data set</param>
        /// <param name="param">format: 'key1=value1 key2=value2'</param>
        /// <param name="res">handle of created Booster</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterCreate", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterCreate(IntPtr trainset,
            [MarshalAs(UnmanagedType.LPStr)]string param,
            ref IntPtr res);

        /*!
        * \brief load an existing boosting from model file
        * \param filename filename of model
        * \param out_num_iterations number of iterations of this booster
        * \param out handle of created Booster
        * \return 0 when succeed, -1 when failure happens
        */
        [DllImport(DllName, EntryPoint = "LGBM_BoosterCreateFromModelfile", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterCreateFromModelfile(
            [MarshalAs(UnmanagedType.LPStr)]string filename,
            ref int outNumIterations,
            ref IntPtr res);

        /// <summary>
        /// load an existing boosting from string
        /// </summary>
        /// <param name="modelStr">model string</param>
        /// <param name="outNumIterations">number of iterations of this booster</param>
        /// <param name="ret">handle of created Booster</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterLoadModelFromString", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterLoadModelFromString(
            [MarshalAs(UnmanagedType.LPStr)]string modelStr,
            ref int outNumIterations,
            ref IntPtr ret);

        /// <summary>
        /// free obj in handle
        /// </summary>
        /// <param name="handle">handle handle to be freed</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterFree", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterFree(IntPtr handle);

        /// <summary>
        /// Shuffle Models
        /// </summary>
        /// <param name="handle"></param>
        /// <returns></returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterShuffleModels", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterShuffleModels(IntPtr handle);

        /// <summary>
        /// Merge model in two booster to first handle
        /// </summary>
        /// <param name="handle">handle, will merge other handle to this</param>
        /// <param name="otherHandle">other handle</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterMerge", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterMerge(IntPtr handle, IntPtr otherHandle);

        /// <summary>
        /// Add new validation to booster
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="validset">validation data set</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterAddValidData", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterAddValidData(IntPtr handle, IntPtr validset);

        /// <summary>
        /// Reset training data for booster
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="trainData">training data set</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterResetTrainingData", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterResetTrainingData(IntPtr handle, IntPtr trainData);

        /// <summary>
        /// Reset config for current booster
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="parameters">format: 'key1=value1 key2=value2'</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterResetParameter", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterResetParameter(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)]string parameters);

        /// <summary>
        /// Get number of class
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="outLen">number of class</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterGetNumClasses", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterGetNumClasses(IntPtr handle, ref int outLen);

        /// <summary>
        /// update the model in one round
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="isFinished">1 means finised(cannot split any more)</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterUpdateOneIter", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterUpdateOneIter(IntPtr handle, ref int isFinished);

        /// <summary>
        /// Refit the tree model using the new data (online learning)
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="leafPreds"></param>
        /// <param name="nRow">number of rows of leafPreds</param>
        /// <param name="nCol">number of columns of leafPreds</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterRefit", CallingConvention = CallingConvention.StdCall)]
        public static extern unsafe int BoosterRefit(IntPtr handle, int *leafPreds, int nRow, int nCol);

        /// <summary>
        /// update the model, by directly specify gradient and second order gradient, this can be used to support customized loss function
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="grad">gradient statistics</param>
        /// <param name="hess">second order gradient statistics</param>
        /// <param name="isFinished">1 means finised(cannot split any more)</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterUpdateOneIterCustom", CallingConvention = CallingConvention.StdCall)]
        public static extern unsafe int BoosterUpdateOneIterCustom(
            IntPtr handle,
            float *grad,
            float *hess,
            ref int isFinished);

        /// <summary>
        /// Rollback one iteration
        /// </summary>
        /// <param name="handle">handle</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterRollbackOneIter", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterRollbackOneIter(IntPtr handle);

        /// <summary>
        /// Get iteration of current boosting rounds
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="outIteration">iteration of boosting rounds</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterGetCurrentIteration", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterGetCurrentIteration(IntPtr handle, ref int outIteration);

        /// <summary>
        /// Get number of tree per iteration
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="outTreePerIteration">number of tree per iteration</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterNumModelPerIteration", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterNumModelPerIteration(IntPtr handle, ref int outTreePerIteration);

        /// <summary>
        /// Get number of weak sub-models
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="outModels">number of weak sub-models</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterNumberOfTotalModel", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterNumberOfTotalModel(IntPtr handle, ref int outModels);

        /// <summary>
        /// Get number of eval
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="outLen">total number of eval results</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterGetEvalCounts", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterGetEvalCounts(IntPtr handle, ref int outLen);

        /// <summary>
        /// Get name of eval
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="outLen">total number of eval results</param>
        /// <param name="outStrs">names of eval result, need to pre-allocate memory before call this</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterGetEvalNames", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterGetEvalNames(IntPtr handle, int len, ref int outLen, UInt64 bufferLen, ref UInt64 outBufferLen, IntPtr[] outStrs);

        /// <summary>
        /// Get name of features
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="outLen">total number of features</param>
        /// <param name="outStrs">names of features, need to pre-allocate memory before call this</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterGetFeatureNames", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterGetFeatureNames(IntPtr handle, int len, ref int outLen, UInt64 bufferLen, ref UInt64 outBufferLen, IntPtr[] outStrs);

        /// <summary>
        /// Get number of features
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="outLen">total number of features</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterGetNumFeature", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterGetNumFeature(IntPtr handle, ref int outLen);

        /// <summary>
        /// get evaluation for training data and validation data
        /// Note: 1. you should call LGBM_BoosterGetEvalNames first to get the name of evaluation results
        /// 2. should pre-allocate memory for out_results, you can get its length by LGBM_BoosterGetEvalCounts
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="dataIdx">0:training data, 1: 1st valid data, 2:2nd valid data ...</param>
        /// <param name="outLen">len of output result</param>
        /// <param name="outResult">float array contains result</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterGetEval", CallingConvention = CallingConvention.StdCall)]
        public static extern unsafe int BoosterGetEval(IntPtr handle, int dataIdx,
                                 ref int outLen, double* outResult);

        /// <summary>
        /// Get number of predict for inner dataset
        /// this can be used to support customized eval function
        /// Note:  should pre-allocate memory for out_result, its length is equal to num_class* num_data
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="dataIdx">0:training data, 1: 1st valid data, 2:2nd valid data ...</param>
        /// <param name="outLen">len of output result</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterGetNumPredict", CallingConvention = CallingConvention.StdCall)]
        public static extern unsafe int BoosterGetNumPredict(
            IntPtr handle,
            int dataIdx,
            ref long outLen);

        /// <summary>
        /// Get prediction for training data and validation data
        /// this can be used to support customized eval function
        /// Note:  should pre-allocate memory for out_result, its length is equal to num_class* num_data
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="dataIdx">0:training data, 1: 1st valid data, 2:2nd valid data ...</param>
        /// <param name="outLen">len of output result</param>
        /// <param name="outResult">used to set a pointer to array, should allocate memory before call this function</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterGetPredict", CallingConvention = CallingConvention.StdCall)]
        public static extern unsafe int BoosterGetPredict(
            IntPtr handle,
            int dataIdx,
            ref long outLen,
            double* outResult);

        /// <summary>
        /// make prediction for file
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="dataFilename">filename of data file</param>
        /// <param name="dataHasHeader">data file has header or not</param>
        /// <param name="predictType">predict_type</param>
        /// <param name="startIteration">Start index of the iteration to predict</param>
        /// <param name="numIteration">number of iteration for prediction, &lt;= 0 means no limit</param>
        /// <param name="parameter">Other parameters for the parameters, e.g. early stopping for prediction.</param>
        /// <param name="fileName">filename of result file</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterPredictForFile", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterPredictForFile(
            IntPtr handle,
            string dataFilename,
            int dataHasHeader,//really boolean
            CApiPredictType predictType,
            int startIteration,
            int numIteration,
            [MarshalAs(UnmanagedType.LPStr)] string parameter,
            ref string fileName);

        /// <summary>
        /// Get number of prediction
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="numRow">number of rows</param>
        /// <param name="predictType">predict type</param>
        /// <param name="startIteration">Start index of the iteration to predict</param>
        /// <param name="numIteration">number of iteration for prediction, &lt;= 0 means no limit</param>
        /// <param name="outLen">length of prediction</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterCalcNumPredict", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterCalcNumPredict(
            IntPtr handle,
            int numRow,
            CApiPredictType predictType,
            int startIteration,
            int numIteration,
            ref long outLen);

        /// <summary>
        /// make prediction for an new data set
        ///        Note:  should pre-allocate memory for out_result,
        ///               for noraml and raw score: its length is equal to num_class* num_data
        ///               for leaf index, its length is equal to num_class* num_data * num_iteration
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="indPtr">pointer to row headers</param>
        /// <param name="indptrType">type of indptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64</param>
        /// <param name="indices">findex</param>
        /// <param name="data">fvalue</param>
        /// <param name="dataType">type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64</param>
        /// <param name="nIndPtr">number of rows in the matrix + 1</param>
        /// <param name="nElem">number of nonzero elements in the matrix</param>
        /// <param name="numCol">number of columns; when it's set to 0, then guess from data</param>
        /// <param name="predictType">predict type</param>
        /// <param name="startIteration">Start index of the iteration to predict</param>
        /// <param name="numIteration">number of iteration for prediction, &lt;= 0 means no limit</param>
        /// <param name="parameter">Other parameters for the parameters, e.g. early stopping for prediction.</param>
        /// <param name="outLen">len of output result</param>
        /// <param name="outResult">used to set a pointer to array, should allocate memory before call this function</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterPredictForCSR", CallingConvention = CallingConvention.StdCall)]
        private static extern unsafe int BoosterPredictForCsr(
            IntPtr handle,
            int *indPtr,
            CApiDType indptrType,
            int *indices,
            float *data,
            CApiDType dataType,
            long nIndPtr,
            long nElem,
            long numCol,
            CApiPredictType predictType,
            int startIteration,
            int numIteration,
            [MarshalAs(UnmanagedType.LPStr)] string parameter,
            ref long outLen,
            double* outResult);

        public static unsafe int BoosterPredictForCsr(
            IntPtr handle,
            int[] indPtr,
            int[] indices,
            float[] data,
            long nIndPtr,
            long nElem,
            long numCol,
            CApiPredictType predictType,
            int numIteration,
            [MarshalAs(UnmanagedType.LPStr)] string parameter,
            ref long outLen,
            double* outResult)
        {
            fixed (int *indPtr2 = indPtr, indices2 = indices)
            fixed (float* data2 = data)
                return BoosterPredictForCsr(handle,
                            indPtr2, CApiDType.Int32,
                            indices2, data2, CApiDType.Float32,
                            nIndPtr, nElem, numCol, predictType, 0, numIteration, parameter, ref outLen, outResult);
        }

        /// <summary>
        /// make prediction for an new data set
        ///        Note:  should pre-allocate memory for out_result,
        ///               for noraml and raw score: its length is equal to num_class* num_data
        ///               for leaf index, its length is equal to num_class* num_data * num_iteration
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="colPtr">pointer to col headers</param>
        /// <param name="colPtrType">type of col_ptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64</param>
        /// <param name="indices">findex</param>
        /// <param name="data">fvalue</param>
        /// <param name="dataType">type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64</param>
        /// <param name="nColPtr">number of cols in the matrix + 1</param>
        /// <param name="nElem">number of nonzero elements in the matrix</param>
        /// <param name="numRow"> number of rows</param>
        /// <param name="predictType">predict type</param>
        /// <param name="startIteration">Start index of the iteration to predict</param>
        /// <param name="numIteration">number of iteration for prediction, &lt;= 0 means no limit</param>
        /// <param name="parameter">Other parameters for the parameters, e.g. early stopping for prediction.</param>
        /// <param name="outLen">len of output result</param>
        /// <param name="outResult">used to set a pointer to array, should allocate memory before call this function</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterPredictForCSC", CallingConvention = CallingConvention.StdCall)]
        private static extern unsafe int BoosterPredictForCsc(
            IntPtr handle,
            int *colPtr,
            CApiDType colPtrType,
            int *indices,
            float *data,
            CApiDType dataType,
            long nColPtr,
            long nElem,
            long numRow,
            CApiPredictType predictType,
            int startIteration,
            int numIteration,
            [MarshalAs(UnmanagedType.LPStr)] string parameter,
            ref long outLen,
            double* outResult);

        public static unsafe int BoosterPredictForCsc(
            IntPtr handle,
            int[] colPtr,
            int[] indices,
            float[] data,
            long nColPtr,
            long nElem,
            long numRow,
            CApiPredictType predictType,
            int numIteration,
            [MarshalAs(UnmanagedType.LPStr)] string parameter,
            ref long outLen,
            double* outResult)
        {
            fixed (int* colPtr2 = colPtr, indices2 = indices)
            fixed (float* data2 = data)
                return BoosterPredictForCsc(handle,
                        colPtr2, CApiDType.Int32,
                        indices2, data2, CApiDType.Float32,
                        nColPtr, nElem, numRow, predictType, 0, numIteration, parameter, ref outLen, outResult);
        }

        /// <summary>
        ///  make prediction for an new data set
        ///       Note:  should pre-allocate memory for out_result,
        ///             for noraml and raw score: its length is equal to num_class* num_data
        ///             for leaf index, its length is equal to num_class* num_data * num_iteration
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="data">pointer to the data space</param>
        /// <param name="dataType">type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64</param>
        /// <param name="nRow">number of rows</param>
        /// <param name="nCol">number columns</param>
        /// <param name="isRowMajor">1 for row major, 0 for column major</param>
        /// <param name="predictType">predict_type</param>
        /// <param name="startIteration">Start index of the iteration to predict
        /// <param name="numIteration">number of iteration for prediction, &lt;= 0 means no limit</param>
        /// <param name="parameter">Other parameters for the parameters, e.g. early stopping for prediction.</param>
        /// <param name="outLen"> len of output result</param>
        /// <param name="outResult">used to set a pointer to array, should allocate memory before call this function</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterPredictForMat", CallingConvention = CallingConvention.StdCall)]
        private static extern unsafe int BoosterPredictForMat(
            IntPtr handle,
            float *data,
            CApiDType dataType,
            int nRow,
            int nCol,
            int isRowMajor,
            CApiPredictType predictType,
            int startIteration,
            int numIteration,
            [MarshalAs(UnmanagedType.LPStr)] string parameter,
            ref long outLen,
            double* outResult);

        [DllImport(DllName, EntryPoint = "LGBM_BoosterPredictForMats", CallingConvention = CallingConvention.StdCall)]
        private static extern unsafe int BoosterPredictForMats(
            IntPtr handle,
            float** data,
            CApiDType dataType,
            int nRow,
            int nCol,
            CApiPredictType predictType,
            int startIteration,
            int numIteration,
            [MarshalAs(UnmanagedType.LPStr)] string parameter,
            ref long outLen,
            double* outResult);

        public static unsafe int BoosterPredictForMat(
            IntPtr handle,
            float[] data,
            int nRow,
            int nCol,
            bool isRowMajor,
            CApiPredictType predictType,
            int numIteration,
            [MarshalAs(UnmanagedType.LPStr)] string parameter,
            ref long outLen,
            double* outResult)
        {
            fixed (float *dataPtr = data)
                return BoosterPredictForMat(
                    handle,
                    dataPtr, CApiDType.Float32,
                    nRow, nCol,
                    (isRowMajor ? 1 : 0),
                    predictType, 0, numIteration, parameter, ref outLen, outResult);
        }

        public static unsafe int BoosterPredictForMats(
            IntPtr handle,
            float[][] data,
            int nCol,
            CApiPredictType predictType,
            int numIteration,
            [MarshalAs(UnmanagedType.LPStr)] string parameter,
            long outLen,
            double* outResult)
        {
            var gcHandles = new List<GCHandle>(data.Length);
            try
            {
                float*[] dataPtrs = new float*[data.Length];
                for (int i = 0; i < data.Length; i++)
                {
                    var hdl = GCHandle.Alloc(data[i], GCHandleType.Pinned);
                    gcHandles.Add(hdl);
                    dataPtrs[i] = (float*)hdl.AddrOfPinnedObject().ToPointer();
                };

                fixed (float** dataPtr = dataPtrs)
                    return BoosterPredictForMats(
                        handle,
                        dataPtr, CApiDType.Float32,
                        data.Length, nCol,
                        predictType, 0, numIteration, parameter, ref outLen,
                        outResult);
            }
            finally
            {
                foreach (var hdl in gcHandles)
                {
                    if (hdl.IsAllocated)
                        hdl.Free();
                };
            }
        }

        /// <summary>
        /// save model into file
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="startIteration">start iteration</param>
        /// <param name="numIteration">num_iteration, &lt;= 0 means save all</param>
        /// <param name="featureImportanceType">Type of feature importance, can be ``C_API_FEATURE_IMPORTANCE_SPLIT`` or ``C_API_FEATURE_IMPORTANCE_GAIN``
        /// <param name="fileName">file name</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterSaveModel", CallingConvention = CallingConvention.StdCall)]
        public static extern unsafe int BoosterSaveModel(
            IntPtr handle,
            int startIteration,
            int numIteration,
            CApiFeatureImportanceType featureImportanceType,
            [MarshalAs(UnmanagedType.LPStr)] string fileName);

        /// <summary>
        /// save model to string
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="numIteration">&lt;= 0 means save all</param>
        /// <param name="featureImportanceType">Type of feature importance, can be ``C_API_FEATURE_IMPORTANCE_SPLIT`` or ``C_API_FEATURE_IMPORTANCE_GAIN``
        /// <param name="bufferLen">buffer length, if buffer_len &lt; out_len, re-allocate buffer</param>
        /// <param name="outLen">actual output length</param>
        /// <param name="outStr">string of model, need to pre-allocate memory before call this</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterSaveModelToString", CallingConvention = CallingConvention.StdCall)]
        public static extern unsafe int BoosterSaveModelToString(IntPtr handle,
            int startIteration,
            int numIteration,
            CApiFeatureImportanceType featureImportanceType,
            long bufferLen,
            ref long outLen,
            byte* outStr);//remember a .Net char is unicode...

        /// <summary>
        /// dump model to json
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="startIteration">&lt;= 0 means save all</param>
        /// <param name="numIteration">&lt;= 0 means save all</param>
        /// <param name="featureImportanceType">Type of feature importance, can be ``C_API_FEATURE_IMPORTANCE_SPLIT`` or ``C_API_FEATURE_IMPORTANCE_GAIN``
        /// <param name="bufferLen">buffer length, if buffer_len &lt; out_len, re-allocate buffer</param>
        /// <param name="outLen">actual output length</param>
        /// <param name="outStr">json format string of model, need to pre-allocate memory before call this</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterDumpModel", CallingConvention = CallingConvention.StdCall)]
        public static extern unsafe int BoosterDumpModel(
            IntPtr handle,
            int startIteration,
            int numIteration,
            CApiFeatureImportanceType featureImportanceType,
            long bufferLen,
            ref long outLen,
            byte* outStr);

        /// <summary>
        /// Get leaf value
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="treeIdx">index of tree</param>
        /// <param name="leafIdx">index of leaf</param>
        /// <param name="outVal">out result</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterGetLeafValue", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterGetLeafValue(
            IntPtr handle,
            int treeIdx,
            int leafIdx,
            ref double outVal);

        /// <summary>
        /// Set leaf value
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="treeIdx">index of tree</param>
        /// <param name="leafIdx">index of leaf</param>
        /// <param name="val">leaf value</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterSetLeafValue", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterSetLeafValue(
            IntPtr handle,
            int treeIdx,
            int leafIdx,
            double val);

        /// <summary>
        /// get model feature importance
        /// </summary>
        /// <param name="handle">handle</param>
        /// <param name="numIteration">lte; 0 means use all</param>
        /// <param name="importanceType">``C_API_FEATURE_IMPORTANCE_SPLIT`` for split, ``C_API_FEATURE_IMPORTANCE_GAIN`` for gain</param>
        /// <param name="outResults">output value array</param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_BoosterFeatureImportance", CallingConvention = CallingConvention.StdCall)]
        public static extern unsafe int BoosterFeatureImportance(
            IntPtr handle,
            int numIteration,
            int importanceType,
            double* outResults);

        /// <summary>
        /// Initilize the network
        /// </summary>
        /// <param name="machines">represent the nodes, format: ip1:port1,ip2:port2</param>
        /// <param name="localListenPort"></param>
        /// <param name="listenTimeOut"></param>
        /// <param name="numMachines"></param>
        /// <returns>0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_NetworkInit", CallingConvention = CallingConvention.StdCall)]
        public static extern unsafe int NetworkInit(
            [MarshalAs(UnmanagedType.LPStr)] string machines,
            int localListenPort,
            int listenTimeOut,
            int numMachines);

        /// <summary>
        /// Finalize the network
        /// </summary>
        /// <returns>return 0 when succeed, -1 when failure happens</returns>
        [DllImport(DllName, EntryPoint = "LGBM_NetworkFree", CallingConvention = CallingConvention.StdCall)]
        public static extern int NetworkFree();

        [DllImport(DllName, EntryPoint = "LGBM_NetworkInitWithFunctions", CallingConvention = CallingConvention.StdCall)]
        public static extern int NetworkInitWithFunctions(int numMachines, int rank, ReduceScatterFunction reduceScatterFuncPtr, AllGatherFunction allgatherFuncPtr);

        #endregion
    }
}
