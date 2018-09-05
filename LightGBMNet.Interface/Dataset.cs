using System;
using System.Runtime.InteropServices;

namespace LightGBMNet.Interface
{
    /// <summary>
    /// Wrapper of Dataset object of LightGBM.
    /// </summary>
    public sealed class Dataset : IDisposable
    {
        private IntPtr _handle;
        private int _lastPushedRowID;
        public IntPtr Handle => _handle;

        private Dataset(IntPtr h)
        {
            _handle = h;
        }

        public unsafe Dataset(double[][] sampleValuePerColumn,
            int[][] sampleIndicesPerColumn,
            int numCol,
            int[] sampleNonZeroCntPerColumn,
            int numSampleRow,
            int numTotalRow,
            string param, float[] labels, float[] weights = null, int[] groups = null)
        {
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
                {
                    PInvokeException.Check(PInvoke.DatasetCreateFromSampledColumn(
                        (IntPtr)ptrValues, (IntPtr)ptrIndices, numCol, sampleNonZeroCntPerColumn, numSampleRow, numTotalRow,
                        param, ref _handle),nameof(PInvoke.DatasetCreateFromSampledColumn));
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
            SetLabel(labels);
            SetWeights(weights);
            SetGroup(groups);

            if (GetNumCols() != numCol)
                throw new Exception("Expected GetNumCols to be equal to numCol");

            if (GetNumRows() != numTotalRow)
                throw new Exception("Expected GetNumRows to be equal to numTotalRow");
        }

        public Dataset(Dataset reference, int numTotalRow, float[] labels, float[] weights = null, int[] groups = null)
        {
            IntPtr refHandle = (reference == null ? IntPtr.Zero : reference.Handle);

            PInvokeException.Check(PInvoke.DatasetCreateByReference(refHandle, numTotalRow, ref _handle),
                                   nameof(PInvoke.DatasetCreateByReference));

            SetLabel(labels);
            SetWeights(weights);
            SetGroup(groups);
        }

        // Load a dataset from file, adding additional parameters and using the optional reference dataset to align bins
        public Dataset(string fileName, string parameters, Dataset reference)
        {
            IntPtr refHandle = (reference == null ? IntPtr.Zero : reference.Handle);

            PInvokeException.Check(PInvoke.DatasetCreateFromFile(fileName, parameters, refHandle, ref _handle),
                                   nameof(PInvoke.DatasetCreateFromFile));
        }

        public void PushRows(float[] data, int numRow, int numCol, int startRowIdx)
        {
            if (startRowIdx != _lastPushedRowID)
                throw new ArgumentException("Expected startRowIdx = _lastPushedRowID", "startRowIdx");
            if (numCol != GetNumCols())
                throw new ArgumentException("Expected numCol = GetNumCols()", "numCol");
            if (numRow <= 0)
                throw new ArgumentException("Expected numRow > 0", "numRow");
            if (startRowIdx > GetNumRows() - numRow)
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
            if (startRowIdx >= GetNumRows())
                throw new ArgumentException("Expected startRowIdx >= GetNumRows()", "startRowIdx");
            if (numCol != GetNumCols())
                throw new ArgumentException("Expected numCol = GetNumCols()", "numCol");

            PInvokeException.Check(PInvoke.DatasetPushRowsByCsr(_handle, indPtr, indices, data, nIndptr, numElem, numCol, startRowIdx),
                                   nameof(PInvoke.DatasetPushRowsByCsr));
            _lastPushedRowID = startRowIdx + nIndptr - 1;
        }

        public int GetNumRows()
        {
            int res = 0;
            PInvokeException.Check(PInvoke.DatasetGetNumData(_handle, ref res),
                                   nameof(PInvoke.DatasetGetNumData));
            return res;
        }

        public int GetNumCols()
        {
            int res = 0;
            PInvokeException.Check(PInvoke.DatasetGetNumFeature(_handle, ref res),
                                   nameof(PInvoke.DatasetGetNumFeature));
            return res;
        }

        public unsafe void SetLabel(float[] labels)
        {
            if (labels == null)
                throw new System.ArgumentNullException("labels");

            if (labels.Length != GetNumRows())
                throw new System.ArgumentException("Expected labels to have a length equal to GetNumRows()", "labels");

            fixed (float* ptr = labels)
                PInvokeException.Check(PInvoke.DatasetSetField(_handle, "label", (IntPtr)ptr, labels.Length,
                    PInvoke.CApiDType.Float32), nameof(PInvoke.DatasetSetField));
        }

        public unsafe void SetWeights(float[] weights)
        {
            if (weights == null)
                throw new System.ArgumentNullException("weights");

            if(weights.Length != GetNumRows())
                throw new System.ArgumentException("Expected weights to have a length equal to GetNumRows()", "weights");

            // Skip SetWeights if all weights are same.
            bool allSame = true;
            for (int i = 1; i < weights.Length; ++i)
            {
                if (weights[i] != weights[0])
                {
                    allSame = false;
                    break;
                }
            }
            if (!allSame)
            {
                fixed (float* ptr = weights)
                    PInvokeException.Check(PInvoke.DatasetSetField(_handle, "weight", (IntPtr)ptr, weights.Length,
                        PInvoke.CApiDType.Float32), nameof(PInvoke.DatasetSetField));
            }

        }

        public unsafe void SetGroup(int[] groups)
        {
            if (groups == null)
                throw new System.ArgumentNullException("groups");

            fixed (int* ptr = groups)
                PInvokeException.Check(PInvoke.DatasetSetField(_handle, "group", (IntPtr)ptr, groups.Length,
                    PInvoke.CApiDType.Int32), nameof(PInvoke.DatasetSetField));

        }

        // Not used now. Can use for the continued train.
        public unsafe void SetInitScore(double[] initScores)
        {
            if (initScores == null)
                throw new System.ArgumentNullException("initScores");

            if (initScores.Length % GetNumRows() != 0)
                throw new System.ArgumentException("Expected initScores to have a length a multiple of GetNumRows()", "initScores");

            fixed (double* ptr = initScores)
                PInvokeException.Check(PInvoke.DatasetSetField(_handle, "init_score", (IntPtr)ptr, initScores.Length,
                    PInvoke.CApiDType.Float64), nameof(PInvoke.DatasetSetField));

        }

        public void SaveBinary(string fileName)
        {
            if (fileName == null)
                throw new System.ArgumentNullException("fileName");

            PInvokeException.Check(PInvoke.DatasetSaveBinary(_handle, fileName),
                                   nameof(PInvoke.DatasetSaveBinary));
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

        public Dataset GetSubset(int[] usedRowIndices, int numUsedRowIndices, string pms)
        {
            IntPtr p = IntPtr.Zero;
            PInvokeException.Check(PInvoke.DatasetGetSubset(_handle, usedRowIndices, numUsedRowIndices, pms, ref p),
                                   nameof(PInvoke.DatasetGetSubset));
            return new Dataset(p);
        }

        private static int MAX_FEATURE_NAME_LENGTH = 100;

        public void SetFeatureNames(string[] featureNames)
        {
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

        public string[] GetFeatureNames()
        {
            var numFeatureNames = this.GetNumCols();
            var ptrs = new IntPtr[numFeatureNames];
            for (int i = 0; i < ptrs.Length; ++i) ptrs[i] = IntPtr.Zero;
            var rslts = new string[numFeatureNames];
            try
            {
                for (int i = 0; i < ptrs.Length; ++i)
                    ptrs[i] = Marshal.AllocCoTaskMem(2 * MAX_FEATURE_NAME_LENGTH + 1);
                int retFeatureNames = 0;
                PInvokeException.Check(PInvoke.DatasetGetFeatureNames(_handle, ptrs, ref retFeatureNames),
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

        public float[] GetFloatField(string fieldName)
        {
            int outLen = 0;
            var typ = PInvoke.CApiDType.Float32;
            var ptr = IntPtr.Zero;
            PInvokeException.Check(PInvoke.DatasetGetField(_handle, fieldName, ref outLen, ref ptr, ref typ),
                                    nameof(PInvoke.DatasetGetField));
            if (typ != PInvoke.CApiDType.Float32)
                throw new Exception(String.Format("Field {0} is of type {1}", fieldName, typ));
            var rslts = new float[outLen];
            Marshal.Copy(ptr, rslts, 0, outLen);
            return rslts;
        }

        public double[] GetDoubleField(string fieldName)
        {
            int outLen = 0;
            var typ = PInvoke.CApiDType.Float32;
            var ptr = IntPtr.Zero;
            PInvokeException.Check(PInvoke.DatasetGetField(_handle, fieldName, ref outLen, ref ptr, ref typ),
                                    nameof(PInvoke.DatasetGetField));
            if (typ != PInvoke.CApiDType.Float64)
                throw new Exception(String.Format("Field {0} is of type {1}", fieldName, typ));
            var rslts = new double[outLen];
            Marshal.Copy(ptr, rslts, 0, outLen);
            return rslts;
        }

        public int[] GetInt32Field(string fieldName)
        {
            int outLen = 0;
            var typ = PInvoke.CApiDType.Float32;
            var ptr = IntPtr.Zero;
            PInvokeException.Check(PInvoke.DatasetGetField(_handle, fieldName, ref outLen, ref ptr, ref typ),
                                    nameof(PInvoke.DatasetGetField));
            if (typ != PInvoke.CApiDType.Int32)
                throw new Exception(String.Format("Field {0} is of type {1}", fieldName, typ));
            var rslts = new int[outLen];
            Marshal.Copy(ptr, rslts, 0, outLen);
            return rslts;
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
