﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace LightGBMNet.Interface
{
    /// <summary>
    /// Wrapper of Booster object of LightGBM.
    /// </summary>
    public sealed class Booster : IDisposable
    {
        public enum PredictType : int
        {
            /// <summary>
            /// normal prediction, with transform(if needed)
            /// </summary>
            Normal = 0,

            /// <summary>
            /// Raw score
            /// </summary>
            RawScore = 1,

            /// <summary>
            /// Leaf index
            /// </summary>
            LeafIndex = 2,
            /// <summary>
            /// Contribution
            /// </summary>
            Contrib = 3
        }

        public enum ImportanceType : int
        {
            Split = 0,
            Gain  = 1
        }

        private readonly bool _hasValid;
        private readonly bool _hasMetric;

        internal IntPtr Handle { get; private set; }
        public int BestIteration { get; set; }

        private Booster(IntPtr h, int bestIteration)
        {
            BestIteration = bestIteration;
            Handle = h;

            int numEval = this.EvalCounts;
            // At most one metric in ML.NET: to do remove this.
            if (numEval > 1)
                throw new Exception("Expected at most one metric");
            else if (numEval == 1)
                _hasMetric = true;
        }

        public Booster(Parameters parameters, Dataset trainset, Dataset validset = null)
        {
            var param = parameters.ToString();
            var handle = IntPtr.Zero;
            PInvokeException.Check(PInvoke.BoosterCreate(trainset.Handle, param, ref handle),nameof(PInvoke.BoosterCreate));
            Handle = handle;
            if (validset != null)
            {
                PInvokeException.Check(PInvoke.BoosterAddValidData(handle, validset.Handle),nameof(PInvoke.BoosterAddValidData));
                _hasValid = true;
            }
            BestIteration = -1;

            int numEval = this.EvalCounts;
            // At most one metric in ML.NET: to do remove this.
            if (numEval > 1)
                throw new Exception("Expected at most one metric");
            else if (numEval == 1)
                _hasMetric = true;
        }
        
        // Load a booster from a model file.
        public static Booster FromFile(string fileName)
        {
            Check.NonNull(fileName, nameof(fileName));
            var handle = IntPtr.Zero;
            var numIteration = 0;
            PInvokeException.Check(PInvoke.BoosterCreateFromModelfile(fileName, ref numIteration, ref handle),
                                   nameof(PInvoke.BoosterCreateFromModelfile));
            return new Booster(handle, numIteration);
        }

        // Load a booster from a string. Note that I can't use a ctr as would have the same signature as above.
        public static Booster FromString(string model)
        {
            Check.NonNull(model, nameof(model));
            var handle = IntPtr.Zero;
            var numIteration = 0;
            PInvokeException.Check(PInvoke.BoosterLoadModelFromString(model, ref numIteration, ref handle),
                                   nameof(PInvoke.BoosterLoadModelFromString));
            return new Booster(handle, numIteration);
        }

        public void ResetTraingData(Dataset trainset)
        {
            Check.NonNull(trainset, nameof(trainset));
            PInvokeException.Check(PInvoke.BoosterResetTrainingData(Handle, trainset.Handle), 
                                   nameof(PInvoke.BoosterResetTrainingData));
        }

        public void ResetParameter(Parameters pms)
        {
            var param = pms.ToString();
            PInvokeException.Check(PInvoke.BoosterResetParameter(Handle, param),
                                   nameof(PInvoke.BoosterResetParameter));
        }

        public bool Update()
        {
            int isFinished = 0;
            PInvokeException.Check(PInvoke.BoosterUpdateOneIter(Handle, ref isFinished),
                                   nameof(PInvoke.BoosterUpdateOneIter));
            return isFinished == 1;
        }

        public bool UpdateCustom(float[] grad, float[] hess)
        {
            int isFinished = 0;
            PInvokeException.Check(PInvoke.BoosterUpdateOneIterCustom(Handle, grad, hess, ref isFinished),
                                   nameof(PInvoke.BoosterUpdateOneIterCustom));
            return isFinished == 1;
        }

        public void RollbackOneIter()
        {
            PInvokeException.Check(PInvoke.BoosterRollbackOneIter(Handle),
                                   nameof(PInvoke.BoosterRollbackOneIter));
        }

        public double EvalTrain()
        {
            return Eval(0);
        }

        public double EvalValid()
        {
            if (_hasValid)
                return Eval(1);
            else
                return double.NaN;
        }

        private unsafe double Eval(int dataIdx)
        {
            if (!_hasMetric)
                return double.NaN;
            int outLen = 0;
            double[] res = new double[1];
            fixed (double* ptr = res)
                PInvokeException.Check(PInvoke.BoosterGetEval(Handle, dataIdx, ref outLen, ptr),
                                       nameof(PInvoke.BoosterGetEval));
            return res[0];
        }

        public int EvalCounts
        {
            get
            {
                int numEval = 0;
                PInvokeException.Check(PInvoke.BoosterGetEvalCounts(Handle, ref numEval), nameof(PInvoke.BoosterGetEvalCounts));
                return numEval;
            }
        }

        //Gets the names of the metrics.
        public MetricType[] EvalNames
        {
            get
            {
                int numEval = EvalCounts;
                var ptrs = new IntPtr[numEval];
                for (int i = 0; i < ptrs.Length; ++i) ptrs[i] = IntPtr.Zero;
                var rslts = new MetricType[numEval];
                try
                {
                    for (int i = 0; i < ptrs.Length; ++i)
                        ptrs[i] = Marshal.AllocCoTaskMem(sizeof(char) * PInvoke.MAX_PREALLOCATED_STRING_LENGTH);
                    var retNumEval = 0;
                    PInvokeException.Check(PInvoke.BoosterGetEvalNames(Handle, ref retNumEval, ptrs), nameof(PInvoke.BoosterGetEvalNames));
                    if (numEval != retNumEval)
                        throw new Exception("Unexpected number of names returned");
                    for (int i = 0; i < ptrs.Length; ++i)
                        rslts[i] = EnumHelper.ParseMetric(Marshal.PtrToStringAnsi(ptrs[i]));
                }
                finally
                {
                    for (int i = 0; i < ptrs.Length; ++i)
                        Marshal.FreeCoTaskMem(ptrs[i]);
                }
                return rslts;
            }
        }

        public void SaveModel(int startIteration, int numIteration, string fileName)
        {
            Check.NonNull(fileName, nameof(fileName));

            if (startIteration < 0)
                throw new ArgumentOutOfRangeException(nameof(startIteration));
            if (numIteration < 0)
                throw new ArgumentOutOfRangeException(nameof(numIteration));

            PInvokeException.Check(PInvoke.BoosterSaveModel(Handle, startIteration, numIteration, fileName),
                                   nameof(PInvoke.BoosterSaveModel));
        }

        private unsafe string GetModelString()
        {
            long bufLen = 2L << 15;
            byte[] buffer = new byte[bufLen];
            long size = 0;
            fixed (byte* ptr = buffer)
                PInvokeException.Check(PInvoke.BoosterSaveModelToString(Handle, -1, BestIteration, bufLen, ref size, ptr),
                                       nameof(PInvoke.BoosterSaveModelToString));
            // If buffer size is not enough, reallocate buffer and get again.
            if (size > bufLen)
            {
                bufLen = size;
                buffer = new byte[bufLen];
                fixed (byte* ptr = buffer)
                    PInvokeException.Check(PInvoke.BoosterSaveModelToString(Handle, -1, BestIteration, bufLen, ref size, ptr),
                                           nameof(PInvoke.BoosterSaveModelToString));
            }
            byte[] content = new byte[size];
            Array.Copy(buffer, content, size);
            fixed (byte* ptr = content)
                return Marshal.PtrToStringAnsi((IntPtr)ptr);
        }

        private unsafe string GetModelJSON(int startIteration, int numIteration)
        {
            long bufLen = 2L << 15;
            byte[] buffer = new byte[bufLen];
            long size = 0L;
            fixed (byte* ptr = buffer)
                PInvokeException.Check(PInvoke.BoosterDumpModel(Handle, startIteration, numIteration, bufLen, ref size, ptr),
                                       nameof(PInvoke.BoosterDumpModel));
            // If buffer size is not enough, reallocate buffer and get again.
            if (size > bufLen)
            {
                bufLen = size;
                buffer = new byte[bufLen];
                fixed (byte* ptr = buffer)
                    PInvokeException.Check(PInvoke.BoosterDumpModel(Handle, startIteration, numIteration, bufLen, ref size, ptr),
                                           nameof(PInvoke.BoosterDumpModel));
            }
            byte[] content = new byte[size];
            Array.Copy(buffer, content, size);
            fixed (byte* ptr = content)
                return Marshal.PtrToStringAnsi((IntPtr)ptr);
        }

        private static double[] Str2DoubleArray(string str, char delimiter)
        {
            return str.Split(delimiter).Select(
                x => { double t; double.TryParse(x, out t); return t; }
            ).ToArray();

        }

        private static int[] Str2IntArray(string str, char delimiter)
        {
            return str.Split(delimiter).Select(int.Parse).ToArray();
        }

        private static UInt32[] Str2UIntArray(string str, char delimiter)
        {
            return str.Split(delimiter).Select(UInt32.Parse).ToArray();
        }

        private static bool GetIsDefaultLeft(UInt32 decisionType)
        {
            // The second bit.
            return (decisionType & 2) > 0;
        }

        private static bool GetIsCategoricalSplit(UInt32 decisionType)
        {
            // The first bit.
            return (decisionType & 1) > 0;
        }

        private static bool GetHasMissing(UInt32 decisionType)
        {
            // The 3rd and 4th bits.
            return ((decisionType >> 2) & 3) > 0;
        }

        private static double[] GetDefaultValue(double[] threshold, UInt32[] decisionType)
        {
            double[] ret = new double[threshold.Length];
            for (int i = 0; i < threshold.Length; ++i)
            {
                if (GetHasMissing(decisionType[i]) && !GetIsCategoricalSplit(decisionType[i]))
                {
                    if (GetIsDefaultLeft(decisionType[i]))
                        ret[i] = threshold[i];
                    else
                        ret[i] = threshold[i] + 1;
                }
            }
            return ret;
        }

        private static bool FindInBitset(UInt32[] bits, int start, int end, int pos)
        {
            int i1 = pos / 32;
            if (start + i1 >= end)
                return false;
            int i2 = pos % 32;
            return ((bits[start + i1] >> i2) & 1) > 0;
        }

        private static int[] GetCatThresholds(UInt32[] catThreshold, int lowerBound, int upperBound)
        {
            List<int> cats = new List<int>();
            for (int j = lowerBound; j < upperBound; ++j)
            {
                // 32 bits.
                for (int k = 0; k < 32; ++k)
                {
                    int cat = (j - lowerBound) * 32 + k;
                    if (FindInBitset(catThreshold, lowerBound, upperBound, cat) && cat > 0)
                        cats.Add(cat);
                }
            }
            return cats.ToArray();
        }

        public double GetLeafValue(int treeIdx, int leafIdx)
        {
            double val = 0.0;
            PInvokeException.Check(PInvoke.BoosterGetLeafValue(Handle, treeIdx, leafIdx, ref val),
                                   nameof(PInvoke.BoosterGetLeafValue));
            return val;
        }

        public void SetLeafValue(int treeIdx, int leafIdx, double val)
        {
            PInvokeException.Check(PInvoke.BoosterSetLeafValue(Handle, treeIdx, leafIdx, val),
                                   nameof(PInvoke.BoosterSetLeafValue));
        }

        public int NumClasses
        {
            get
            {
                int cnt = 0;
                PInvokeException.Check(PInvoke.BoosterGetNumClasses(Handle, ref cnt),
                                       nameof(PInvoke.BoosterGetNumClasses));
                return cnt;
            }
        }

        public int CurrentIteration
        {
            get
            {
                int cnt = 0;
                PInvokeException.Check(PInvoke.BoosterGetCurrentIteration(Handle, ref cnt),
                                       nameof(PInvoke.BoosterGetCurrentIteration));
                return cnt;
            }
        }

        /// <summary>
        /// Get number of tree per iteration
        /// </summary>
        public int NumModelPerIteration
        {
            get
            {
                int cnt = 0;
                PInvokeException.Check(PInvoke.BoosterNumModelPerIteration(Handle, ref cnt),
                                       nameof(PInvoke.BoosterNumModelPerIteration));
                return cnt;
            }
        }

        /// <summary>
        /// The number of weak sub-models
        /// </summary>
        public int NumberOfTotalModel
        {
            get
            {
                int cnt = 0;
                PInvokeException.Check(PInvoke.BoosterNumberOfTotalModel(Handle, ref cnt),
                                       nameof(PInvoke.BoosterNumberOfTotalModel));
                return cnt;
            }
        }

        public void ResetTrainingData(Dataset train)
        {
            Check.NonNull(train,nameof(train));
            PInvokeException.Check(PInvoke.BoosterResetTrainingData(Handle, train.Handle),
                                   nameof(PInvoke.BoosterResetTrainingData));
        }

        public void MergeWith(Booster other)
        {
            Check.NonNull(other, nameof(other));
            PInvokeException.Check(PInvoke.BoosterMerge(Handle, other.Handle),
                                   nameof(PInvoke.BoosterMerge));
        }

        public int NumFeatures
        {
            get
            {
                int cnt = 0;
                PInvokeException.Check(PInvoke.BoosterGetNumFeature(Handle, ref cnt),
                                       nameof(PInvoke.BoosterGetNumFeature));
                return cnt;
            }
        }

        // synonym for NumFeatures.
        public int NumCols
        {
            get { return this.NumFeatures; }
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
                        ptrs[i] = Marshal.AllocCoTaskMem(sizeof(char) * PInvoke.MAX_PREALLOCATED_STRING_LENGTH);
                    int retFeatureNames = 0;
                    PInvokeException.Check(PInvoke.BoosterGetFeatureNames(Handle, ref retFeatureNames, ptrs),
                                           nameof(PInvoke.BoosterGetFeatureNames));
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

        // Get the importance of a feature.
        public unsafe double[] GetFeatureImportance(int numIteration, ImportanceType importanceType)
        {
            // Get the number of features
            int cnt = this.NumFeatures;

            double[] res = new double[cnt];
            fixed (double* ptr = res)
                PInvokeException.Check(PInvoke.BoosterFeatureImportance(Handle, numIteration, (int)importanceType, ptr),
                                       nameof(PInvoke.BoosterFeatureImportance));
            return res;
        }

        public void ShuffleModels()
        {
            PInvokeException.Check(PInvoke.BoosterShuffleModels(Handle), nameof(PInvoke.BoosterShuffleModels));
        }

        //To Do: store this matrix efficiently.
        public void Refit(int[,] leafPreds)
        {
            Check.NonNull(leafPreds, nameof(leafPreds));
            var len1 = leafPreds.GetLength(0);//nrow
            var len2 = leafPreds.GetLength(1);//ncol
            var data = new int[len1 * len2];
            for (int i = 0; i< len1; ++i)
                for(int j = 0; j < len2; ++j)
                {
                    data[i * len2 + j] = leafPreds[i, j];
                }
            PInvokeException.Check(PInvoke.BoosterRefit(Handle,data,len1,len2), nameof(PInvoke.BoosterRefit));
        }

        public long GetNumPredict(int dataIdx)
        {
            long outLen = 0;
            PInvokeException.Check(PInvoke.BoosterGetNumPredict(Handle, dataIdx, ref outLen),
                                    nameof(PInvoke.BoosterGetNumPredict));
            return outLen;
        }

        public unsafe double[] GetPredict(int dataIdx)
        {
            var numPredict = GetNumPredict(dataIdx);
            long outLen = 0;
            double[] res = new double[outLen];
            fixed (double* ptr = res)
            {
                PInvokeException.Check(PInvoke.BoosterGetPredict(Handle, dataIdx, ref outLen, ptr),
                                      nameof(PInvoke.BoosterGetPredict));
            }
            var copy = new double[outLen];
            Array.Copy(res, copy, outLen);
            return copy;
        }

        // Calculate the number of predictions for a dataset with a given number of rows and iterations.
        public long CalcNumPredict(int numRow, PredictType predType, int numIteration)
        {
            long outLen = 0L;
            PInvokeException.Check(PInvoke.BoosterCalcNumPredict(Handle, numRow, (int)predType, numIteration, ref outLen),
                                  nameof(PInvoke.BoosterCalcNumPredict));
            return outLen;
        }



        /*
        public FastTree.Internal.Ensemble GetModel(int[] categoricalFeatureBoudaries)
        {
            FastTree.Internal.Ensemble res = new FastTree.Internal.Ensemble();
            string modelString = GetModelString();
            string[] lines = modelString.Split('\n');
            int i = 0;
            for (; i < lines.Length;)
            {
                if (lines[i].StartsWith("Tree="))
                {
                    Dictionary<string, string> kvPairs = new Dictionary<string, string>();
                    ++i;
                    while (!lines[i].StartsWith("Tree=") && lines[i].Trim().Length != 0)
                    {
                        string[] kv = lines[i].Split('=');
                        Contracts.Check(kv.Length == 2);
                        kvPairs[kv[0].Trim()] = kv[1].Trim();
                        ++i;
                    }
                    int numLeaves = int.Parse(kvPairs["num_leaves"]);
                    int numCat = int.Parse(kvPairs["num_cat"]);
                    if (numLeaves > 1)
                    {
                        var leftChild = Str2IntArray(kvPairs["left_child"], ' ');
                        var rightChild = Str2IntArray(kvPairs["right_child"], ' ');
                        var splitFeature = Str2IntArray(kvPairs["split_feature"], ' ');
                        var threshold = Str2DoubleArray(kvPairs["threshold"], ' ');
                        var splitGain = Str2DoubleArray(kvPairs["split_gain"], ' ');
                        var leafOutput = Str2DoubleArray(kvPairs["leaf_value"], ' ');
                        var decisionType = Str2UIntArray(kvPairs["decision_type"], ' ');
                        var defaultValue = GetDefalutValue(threshold, decisionType);
                        var categoricalSplitFeatures = new int[numLeaves - 1][];
                        var categoricalSplit = new bool[numLeaves - 1];
                        if (categoricalFeatureBoudaries != null)
                        {
                            // Add offsets to split features.
                            for (int node = 0; node < numLeaves - 1; ++node)
                                splitFeature[node] = categoricalFeatureBoudaries[splitFeature[node]];
                        }

                        if (numCat > 0)
                        {
                            var catBoundaries = Str2IntArray(kvPairs["cat_boundaries"], ' ');
                            var catThreshold = Str2UIntArray(kvPairs["cat_threshold"], ' ');
                            for (int node = 0; node < numLeaves - 1; ++node)
                            {
                                if (GetIsCategoricalSplit(decisionType[node]))
                                {
                                    int catIdx = (int)threshold[node];
                                    var cats = GetCatThresholds(catThreshold, catBoundaries[catIdx], catBoundaries[catIdx + 1]);
                                    categoricalSplitFeatures[node] = new int[cats.Length];
                                    // Convert Cat thresholds to feature indices.
                                    for (int j = 0; j < cats.Length; ++j)
                                        categoricalSplitFeatures[node][j] = splitFeature[node] + cats[j] - 1;

                                    splitFeature[node] = -1;
                                    categoricalSplit[node] = true;
                                    // Swap left and right child.
                                    int t = leftChild[node];
                                    leftChild[node] = rightChild[node];
                                    rightChild[node] = t;
                                }
                                else
                                {
                                    categoricalSplit[node] = false;
                                }
                            }
                        }
                        RegressionTree tree = RegressionTree.Create(numLeaves, splitFeature, splitGain,
                            threshold.Select(x => (float)(x)).ToArray(), defaultValue.Select(x => (float)(x)).ToArray(), leftChild, rightChild, leafOutput,
                            categoricalSplitFeatures, categoricalSplit);
                        res.AddTree(tree);
                    }
                    else
                    {
                        RegressionTree tree = new RegressionTree(2);
                        var leafOutput = Str2DoubleArray(kvPairs["leaf_value"], ' ');
                        if (leafOutput[0] != 0)
                        {
                            // Convert Constant tree to Two-leaf tree, avoid being filter by TLC.
                            var categoricalSplitFeatures = new int[1][];
                            var categoricalSplit = new bool[1];
                            tree = RegressionTree.Create(2, new int[] { 0 }, new double[] { 0 },
                                new float[] { 0 }, new float[] { 0 }, new int[] { -1 }, new int[] { -2 }, new double[] { leafOutput[0], leafOutput[0] },
                                categoricalSplitFeatures, categoricalSplit);
                        }
                        res.AddTree(tree);
                    }
                }
                else
                    ++i;
            }
            return res;
        }
        */
        #region IDisposable
        public void Dispose()
        {
            if (Handle != IntPtr.Zero)
                PInvokeException.Check(PInvoke.BoosterFree(Handle), nameof(PInvoke.BoosterFree));
            Handle = IntPtr.Zero;
        }
        #endregion
    }
}
