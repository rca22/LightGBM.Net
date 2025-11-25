using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using LightGBMNet.Tree;

namespace LightGBMNet.Train
{
    /// <summary>
    /// Wrapper of Dataset object of LightGBM.
    /// </summary>
    public sealed class Dataset : IDisposable
    {
        private IntPtr _handle;
        private int _lastPushedRowID;
        internal IntPtr Handle => _handle;

        public CommonParameters CommonParameters { get; }
        public DatasetParameters DatasetParameters { get; }

        private static string ParamsToString(CommonParameters cp, DatasetParameters dp)
        {
            var dict = new Dictionary<string, string>();
            if (cp != null) cp.AddParameters(dict);
            if (dp != null) dp.AddParameters(dict);
            return ParamsHelper.JoinParameters(dict);
        }

        private Dataset(IntPtr h, CommonParameters cp, DatasetParameters dp)
        {
            _handle = h;
            CommonParameters = cp;
            DatasetParameters = dp;
        }

        public unsafe Dataset(double[][] sampleValuePerColumn,
            int[][] sampleIndicesPerColumn,
            int numCol,
            int[] sampleNonZeroCntPerColumn,
            int numSampleRow,
            int numTotalRow,
            CommonParameters cp,
            DatasetParameters dp,
            float[] labels = null,
            float[] weights = null,
            int[] groups = null)
        {
            CommonParameters = cp;
            DatasetParameters = dp;
            var pmString = ParamsToString(cp, dp);

            _handle = IntPtr.Zero;

            // Use GCHandle to pin the memory, avoid the memory relocation.
            GCHandle[] gcValues = new GCHandle[numCol];
            GCHandle[] gcIndices = new GCHandle[numCol];
            try
            {
                double*[] ptrArrayValues = new double*[numCol];
                int*[] ptrArrayIndices = new int*[numCol];
                for (int i = 0; i < numCol; i++)
                {
                    gcValues[i] = GCHandle.Alloc(sampleValuePerColumn[i], GCHandleType.Pinned);
                    ptrArrayValues[i] = (double*)gcValues[i].AddrOfPinnedObject().ToPointer();
                    gcIndices[i] = GCHandle.Alloc(sampleIndicesPerColumn[i], GCHandleType.Pinned);
                    ptrArrayIndices[i] = (int*)gcIndices[i].AddrOfPinnedObject().ToPointer();
                };
                fixed (double** ptrValues = ptrArrayValues)
                fixed (int** ptrIndices = ptrArrayIndices)
                fixed (int* ptrSampleNonZeroCntPerColumn = sampleNonZeroCntPerColumn)
                {
                    PInvokeException.Check(PInvoke.DatasetCreateFromSampledColumn(
                        (IntPtr)ptrValues, (IntPtr)ptrIndices, numCol, ptrSampleNonZeroCntPerColumn, numSampleRow, numTotalRow, numTotalRow,
                        pmString, ref _handle),nameof(PInvoke.DatasetCreateFromSampledColumn));
                }
            }
            finally
            {
                for (int i = 0; i < numCol; i++)
                {
                    if (gcValues[i].IsAllocated)
                        gcValues[i].Free();
                    if (gcIndices[i].IsAllocated)
                        gcIndices[i].Free();
                };
            }
            if(labels != null)
                SetLabels(labels);
            if (weights != null)
                SetWeights(weights);
            if (groups != null)
                SetGroups(groups);

            if (NumFeatures != numCol)
                throw new Exception("Expected GetNumCols to be equal to numCol");

            if (NumRows != numTotalRow)
                throw new Exception("Expected GetNumRows to be equal to numTotalRow");
        }

        public unsafe Dataset(float[][] data,
            int numCol,
            CommonParameters cp,
            DatasetParameters dp,
            float[] labels = null,
            float[] weights = null,
            int[] groups = null,
            Dataset reference = null)
        {
            CommonParameters = cp;
            DatasetParameters = dp;
            var pmString = ParamsToString(cp, dp);

            _handle = IntPtr.Zero;

            var gcHandles = new List<GCHandle>(data.Length);
            try
            {
                float*[] dataPtrs = new float*[data.Length];
                int[] nRows = new int[data.Length];
                int[] isRowMajor = new int[data.Length];
                for (int i = 0; i < data.Length; i++)
                {
                    var hdl = GCHandle.Alloc(data[i], GCHandleType.Pinned);
                    gcHandles.Add(hdl);
                    dataPtrs[i] = (float*)hdl.AddrOfPinnedObject().ToPointer();
                    nRows[i] = 1;
                    isRowMajor[i] = 1;
                }
                fixed (float** dataPtr = dataPtrs)
                fixed (int* nRowsPtr = nRows)
                fixed (int* isRowMajorPtr = isRowMajor)
                {
                    PInvokeException.Check(PInvoke.DatasetCreateFromMats(
                        data.Length,
                        dataPtr,
                        nRowsPtr,
                        numCol,
                        isRowMajorPtr,
                        pmString,
                        reference?._handle ?? IntPtr.Zero,
                        ref _handle
                        ), nameof(PInvoke.DatasetCreateFromMats));
                }
            }
            finally
            {
                foreach (var hdl in gcHandles)
                {
                    if (hdl.IsAllocated)
                        hdl.Free();
                };
            }
            if (labels != null)
                SetLabels(labels);
            if (weights != null)
                SetWeights(weights);
            if (groups != null)
                SetGroups(groups);

            if (NumFeatures != numCol)
                throw new Exception("Expected GetNumCols to be equal to numCol");

            if (NumRows != data.Length)
                throw new Exception("Expected GetNumRows to be equal to numTotalRow");
        }

        public unsafe Dataset(SparseMatrix data,
            int numCol,
            CommonParameters cp,
            DatasetParameters dp,
            float[] labels = null,
            float[] weights = null,
            int[] groups = null,
            Dataset reference = null)
        {
            CommonParameters = cp;
            DatasetParameters = dp;
            var pmString = ParamsToString(cp, dp);

            _handle = IntPtr.Zero;

            fixed (float* dataPtr = data.Data)
            fixed (int* indPtr = data.RowExtents, indices = data.ColumnIndices)
            {
                PInvokeException.Check(PInvoke.DatasetCreateFromCsr(
                    indPtr,
                    indices,
                    dataPtr,
                    data.RowExtents.Length,
                    data.Data.Length,
                    numCol,
                    pmString,
                    reference?._handle ?? IntPtr.Zero,
                    ref _handle
                    ), nameof(PInvoke.DatasetCreateFromCsr));
            }

            if (labels != null)
                SetLabels(labels);
            if (weights != null)
                SetWeights(weights);
            if (groups != null)
                SetGroups(groups);

            if (NumFeatures != numCol)
                throw new Exception("Expected GetNumCols to be equal to numCol");

            if (NumRows != data.RowCount)
                throw new Exception("Expected GetNumRows to be equal to numTotalRow");
        }

        // Load a dataset from file, adding additional parameters and using the optional reference dataset to align bins
        public Dataset(string fileName, CommonParameters cp, DatasetParameters dp, Dataset reference = null)
        {
            Check.NonNull(fileName,nameof(fileName));
            if (!System.IO.File.Exists(fileName))
                throw new ArgumentException(string.Format("File {0} does not exist", fileName));
            if (!fileName.EndsWith(".bin"))
                throw new ArgumentException(string.Format("File {0} is not a .bin file", fileName));

            CommonParameters = cp;
            DatasetParameters = dp;
            var pmString = ParamsToString(cp, dp);

            IntPtr refHandle = (reference == null ? IntPtr.Zero : reference.Handle);

            PInvokeException.Check(PInvoke.DatasetCreateFromFile(fileName.Substring(0,fileName.Length-4), pmString, refHandle, ref _handle),
                                   nameof(PInvoke.DatasetCreateFromFile));
        }

        public void SaveBinary(string fileName)
        {
            Check.NonNull(fileName,nameof(fileName));
            if (!fileName.EndsWith(".bin"))
                throw new ArgumentException(string.Format("File {0} is not a .bin file", fileName));

            PInvokeException.Check(PInvoke.DatasetSaveBinary(_handle, fileName),
                                   nameof(PInvoke.DatasetSaveBinary));
        }

        public void PushRows(float[] data, int numRow, int numCol, int startRowIdx)
        {
            if (startRowIdx != _lastPushedRowID)
                throw new ArgumentException("Expected startRowIdx = _lastPushedRowID", "startRowIdx");
            if (numCol != NumFeatures)
                throw new ArgumentException("Expected numCol = GetNumCols()", "numCol");
            if (numRow <= 0)
                throw new ArgumentException("Expected numRow > 0", "numRow");
            if (startRowIdx > NumRows - numRow)
                throw new ArgumentException("Expected startRowIdx > GetNumRows() - numRow", "numRow");

            PInvokeException.Check(PInvoke.DatasetPushRows(_handle, data, numRow, numCol, startRowIdx),
                                   nameof(PInvoke.DatasetPushRows));
            _lastPushedRowID = startRowIdx + numRow;
        }

        public void PushRows(int[] indPtr, int[] indices, float[] data, int nIndptr,
            long numElem, int numCol, int startRowIdx)
        {
            if (startRowIdx != _lastPushedRowID)
                throw new ArgumentException("Expected startRowIdx = _lastPushedRowID", "startRowIdx");
            if (startRowIdx >= NumRows)
                throw new ArgumentException("Expected startRowIdx >= GetNumRows()", "startRowIdx");
            if (numCol != NumFeatures)
                throw new ArgumentException("Expected numCol = GetNumCols()", "numCol");

            PInvokeException.Check(PInvoke.DatasetPushRowsByCsr(_handle, indPtr, indices, data, nIndptr, numElem, numCol, startRowIdx),
                                   nameof(PInvoke.DatasetPushRowsByCsr));
            _lastPushedRowID = startRowIdx + nIndptr - 1;
        }

        public int NumRows
        {
            get
            {
                int res = 0;
                PInvokeException.Check(PInvoke.DatasetGetNumData(_handle, ref res),
                                       nameof(PInvoke.DatasetGetNumData));
                return res;
            }
        }

        public int NumFeatures
        {
            get
            {
                int res = 0;
                PInvokeException.Check(PInvoke.DatasetGetNumFeature(_handle, ref res),
                                       nameof(PInvoke.DatasetGetNumFeature));
                return res;
            }
        }

        public unsafe void SetLabels(float[] labels)
        {
            if (labels == null)
                throw new ArgumentNullException("labels");

            if (labels.Length != NumRows)
                throw new ArgumentException("Expected labels to have a length equal to GetNumRows()", "labels");

            fixed (float* ptr = labels)
                PInvokeException.Check(PInvoke.DatasetSetField(_handle, "label", (IntPtr)ptr, labels.Length,
                    PInvoke.CApiDType.Float32), nameof(PInvoke.DatasetSetField));
        }

        // note that weights can be null
        public unsafe void SetWeights(float[] weights)
        {
            if (weights != null)
            {
                if (weights.Length != NumRows)
                    throw new ArgumentException("Expected weights to have a length equal to GetNumRows()", "weights");

                fixed (float* ptr = weights)
                    PInvokeException.Check(PInvoke.DatasetSetField(_handle, "weight", (IntPtr)ptr, weights.Length,
                        PInvoke.CApiDType.Float32), nameof(PInvoke.DatasetSetField));
            }
            else
            {
                PInvokeException.Check(PInvoke.DatasetSetField(_handle, "weight", (IntPtr)null, 0,
                    PInvoke.CApiDType.Float32), nameof(PInvoke.DatasetSetField));
            }
        }

        public unsafe void SetGroups(int[] groups)
        {
            if (groups != null)
            {
                fixed (int* ptr = groups)
                    PInvokeException.Check(PInvoke.DatasetSetField(_handle, "group", (IntPtr)ptr, groups.Length,
                        PInvoke.CApiDType.Int32), nameof(PInvoke.DatasetSetField));
            }
            else
            {
                PInvokeException.Check(PInvoke.DatasetSetField(_handle, "group", (IntPtr)null, 0,
                    PInvoke.CApiDType.Int32), nameof(PInvoke.DatasetSetField));
            }
        }

        // Not used now. Can use for the continued train.
        public unsafe void SetInitScore(double[] initScores)
        {
            if (initScores == null)
                throw new ArgumentNullException("initScores");

            if (initScores.Length % NumRows != 0)
                throw new ArgumentException("Expected initScores to have a length a multiple of GetNumRows()", "initScores");

            fixed (double* ptr = initScores)
                PInvokeException.Check(PInvoke.DatasetSetField(_handle, "init_score", (IntPtr)ptr, initScores.Length,
                    PInvoke.CApiDType.Float64), nameof(PInvoke.DatasetSetField));

        }




/*
        public static int DatasetCreateFromCsr(
            int[] indPtr,
            int[] indices,
            float[] data,
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
*/
       
/*
        public static int DatasetCreateFromCsc(
            int[] colPtr,
            int[] indices,
            float[] data,
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
*/

        /// <summary>
        /// Create from single matrix
        /// </summary>
        //Dataset(float[,] data,bool isRowMajor, Parameters pms = null, Dataset reference = null)
        //{
        //    var pmStr = (pms != null) ? pms.ToString() : "";
        //    var r = (reference != null) ? reference.Handle : IntPtr.Zero;
        //    var rows = data.GetLength(0);
        //    var cols = data.GetLength(1);

        //    PInvokeException.Check(PInvoke.DatasetCreateFromMat(data, rows, cols, isRowMajoe, pmStr, r, ref _handle),
        //                           nameof(PInvoke.DatasetCreateFromMat));
        //}
/*
        public static int DatasetCreateFromMat(
            float[] data,
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
*/
/*
        public static int DatasetCreateFromMats(
            float[][] data,
            int[] nRow,
            int nCol,
            bool isRowMajor,
            string parameters,
            IntPtr reference,
            ref IntPtr ret)
        {
            return DatasetCreateFromMats(
                data.Length,
                data, CApiDType.Float32,
                nRow, nCol,
                (isRowMajor ? 1 : 0),
                parameters, reference, ref ret);
        }
*/

        public unsafe Dataset GetSubset(int[] usedRowIndices, CommonParameters cp = null, DatasetParameters dp = null)
        {
            if (cp == null) cp = CommonParameters;
            if (dp == null) dp = DatasetParameters;
            var pmString = ParamsToString(cp, dp);
            IntPtr p = IntPtr.Zero;
            fixed (int* usedRowIndices2 = usedRowIndices)
                PInvokeException.Check(PInvoke.DatasetGetSubset(_handle, usedRowIndices2, usedRowIndices.Length, pmString, ref p),
                                   nameof(PInvoke.DatasetGetSubset));
            return new Dataset(p, cp, dp);
        }

        private static readonly int MAX_FEATURE_NAME_LENGTH = 100;

        public void SetFeatureNames(string[] featureNames)
        {
            if (featureNames.Length != NumFeatures)
                throw new ArgumentException("Array length inconsistent with number of features (columns", "featureNames");

            for (int i = 0; i < featureNames.Length; ++i)
                if (featureNames[i].Length > MAX_FEATURE_NAME_LENGTH)
                    throw new ArgumentException("Feature name too long", "featureNames");

            var ptrs = new IntPtr[featureNames.Length];
            for (int i = 0; i < ptrs.Length; ++i) ptrs[i] = IntPtr.Zero;
            try
            {
                for (int i = 0; i < ptrs.Length; ++i)
                    ptrs[i] = Marshal.StringToCoTaskMemAnsi(featureNames[i]);
                PInvokeException.Check(PInvoke.DatasetSetFeatureNames(_handle, ptrs, ptrs.Length),
                                       nameof(PInvoke.DatasetSetFeatureNames));
            }
            finally
            {
                for(int i = 0; i < ptrs.Length; ++i)
                    if (ptrs[i] != IntPtr.Zero) Marshal.FreeCoTaskMem(ptrs[i]);
            }
        }

        public string[] FeatureNames
        {
            get
            {
                var numFeatureNames = this.NumFeatures;
                var ptrs = new IntPtr[numFeatureNames];
                for (int i = 0; i < ptrs.Length; ++i) ptrs[i] = IntPtr.Zero;
                var rslts = new string[numFeatureNames];
                try
                {
                    for (int i = 0; i < ptrs.Length; ++i)
                        ptrs[i] = Marshal.AllocCoTaskMem(2 * MAX_FEATURE_NAME_LENGTH + 1);
                    int retFeatureNames = 0;
                    long out_buffer_len = 0;
                    PInvokeException.Check(PInvoke.DatasetGetFeatureNames(_handle, numFeatureNames, ref retFeatureNames, MAX_FEATURE_NAME_LENGTH, ref out_buffer_len, ptrs),
                                           nameof(PInvoke.DatasetGetFeatureNames));
                    if (retFeatureNames != numFeatureNames)
                        throw new Exception("Unexpected number of feature names returned");
                    for (int i = 0; i < ptrs.Length; ++i)
                        rslts[i] = Marshal.PtrToStringAnsi(ptrs[i]);
                }
                finally
                {
                    for (int i = 0; i < ptrs.Length; ++i)
                        Marshal.FreeCoTaskMem(ptrs[i]);
                }
                return rslts;
            }
        }

        private float[] GetFloatField(string fieldName)
        {
            int outLen = 0;
            var typ = PInvoke.CApiDType.Float32;
            var ptr = IntPtr.Zero;
            PInvokeException.Check(PInvoke.DatasetGetField(_handle, fieldName, ref outLen, ref ptr, ref typ),
                                    nameof(PInvoke.DatasetGetField));
            if (typ != PInvoke.CApiDType.Float32)
                throw new Exception(string.Format("Field {0} is of type {1}", fieldName, typ));
            if (ptr == IntPtr.Zero) return null;
            var rslts = new float[outLen];
            Marshal.Copy(ptr, rslts, 0, outLen);
            return rslts;
        }
        public float[] GetLabels()
        {
            return GetFloatField("label");
        }

        public float[] GetWeights()
        {
            return GetFloatField("weight");
        }

        private double[] GetDoubleField(string fieldName)
        {
            int outLen = 0;
            var typ = PInvoke.CApiDType.Float32;
            var ptr = IntPtr.Zero;
            PInvokeException.Check(PInvoke.DatasetGetField(_handle, fieldName, ref outLen, ref ptr, ref typ),
                                    nameof(PInvoke.DatasetGetField));
            if (typ != PInvoke.CApiDType.Float64)
                throw new Exception(string.Format("Field {0} is of type {1}", fieldName, typ));
            if (ptr == IntPtr.Zero) return null;
            var rslts = new double[outLen];
            Marshal.Copy(ptr, rslts, 0, outLen);
            return rslts;
        }

        public double[] GetInitScore()
        {
            return GetDoubleField("init_score");
        }

        private int[] GetInt32Field(string fieldName)
        {
            int outLen = 0;
            var typ = PInvoke.CApiDType.Float32;
            var ptr = IntPtr.Zero;
            PInvokeException.Check(PInvoke.DatasetGetField(_handle, fieldName, ref outLen, ref ptr, ref typ),
                                    nameof(PInvoke.DatasetGetField));
            if (typ != PInvoke.CApiDType.Int32)
                throw new Exception(string.Format("Field {0} is of type {1}", fieldName, typ));
            if (ptr == IntPtr.Zero) return null;
            var rslts = new int[outLen];
            Marshal.Copy(ptr, rslts, 0, outLen);
            return rslts;
        }

        public int[] GetGroups()
        {
            return GetInt32Field("group");
        }

        #region IDisposable

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
                PInvokeException.Check(PInvoke.DatasetFree(_handle),
                                       nameof(PInvoke.DatasetFree));
            _handle = IntPtr.Zero;
        }
        
        #endregion
    }
}
