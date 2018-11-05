using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace LightGBMNet.Train
{
    /// <summary>
    /// Wrapper of Booster object of LightGBM.
    /// </summary>
    public sealed class Booster : IDisposable
    {
        private static ParamsHelper<CommonParameters>    _helperCommon    = new ParamsHelper<CommonParameters>();
        private static ParamsHelper<DatasetParameters>   _helperDataset   = new ParamsHelper<DatasetParameters>();
        private static ParamsHelper<MetricParameters>    _helperMetric    = new ParamsHelper<MetricParameters>();
        private static ParamsHelper<ObjectiveParameters> _helperObjective = new ParamsHelper<ObjectiveParameters>();
        private static ParamsHelper<LearningParameters>  _helperLearning  = new ParamsHelper<LearningParameters>();

        //private static float secondBiggestFloat  = NextDown(Single.MaxValue);
        //private static float secondSmallestFloat = NextDown(Single.MinValue);

        //private static unsafe float NextDown(float x)
        //{
        //    uint temp = *(uint*)&x;
        //    temp--;
        //    return *(float*)&temp;
        //}
        //
        //private static unsafe float NextUp(float x)
        //{
        //    uint temp = *(uint*)&x;
        //    temp++;
        //    return *(float*)&temp;
        //}

        private static unsafe double NextDown(double x)
        {
            ulong temp = *(ulong*)&x;
            temp--;
            return *(double*)&temp;
        }

        private static unsafe double NextUp(double x)
        {
            ulong temp = *(ulong*)&x;
            temp++;
            return *(double*)&temp;
        }

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
                throw new Exception($"Expected at most one metric, got {numEval}");
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

        public unsafe string GetModelString()
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
            return str.Split(delimiter).Select(double.Parse).ToArray();

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
            var ret = new double[threshold.Length];
            for (int i = 0; i < threshold.Length; ++i)
            {
                if (GetHasMissing(decisionType[i]) && !GetIsCategoricalSplit(decisionType[i])) // NOTE: categorical always take RHS branch for missing
                {
                    // need to be careful here to generate a value that is genuinely LEQ for left branch, and GT for right branch!
                    var t = threshold[i];
                    if (GetIsDefaultLeft(decisionType[i]))
                        ret[i] = t; //  (t == 0.0f) ? -1.0f : ((t > 0) ? NextDown(t) : NextUp(t)); 
                    else
                        ret[i] = (t == 0.0f) ? +1.0f : ((t > 0) ? NextUp(t) : NextDown(t));  // TODO: INFINITY!!!
                }
            }
            return ret;
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

        public unsafe double [] PredictForMat(PredictType predictType, float [] data)
        {
            long outLen = NumClasses; // TODO
            double[] outResult = new double[outLen];
            fixed (double* ptr = outResult)
                PInvokeException.Check(PInvoke.BoosterPredictForMat( Handle
                                                                   , data
                                                                   , /*nRow*/1
                                                                   , /*nCol*/data.Length
                                                                   , /*isRowMajor*/true
                                                                   , (PInvoke.CApiPredictType)predictType
                                                                   , /*numIteration*/BestIteration
                                                                   , ""
                                                                   , ref outLen
                                                                   , ptr
                                                                   ), nameof(PInvoke.BoosterPredictForMat));
            return outResult;
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
            long outLen = GetNumPredict(dataIdx);
            double[] res = new double[outLen];
            fixed (double* ptr = res)
            {
                PInvokeException.Check(PInvoke.BoosterGetPredict(Handle, dataIdx, ref outLen, ptr),
                                    nameof(PInvoke.BoosterGetPredict));
            }
            Debug.Assert(outLen == res.Length);
            return res;
        }

        // Calculate the number of predictions for a dataset with a given number of rows and iterations.
        public long CalcNumPredict(int numRow, PredictType predType, int numIteration)
        {
            long outLen = 0L;
            PInvokeException.Check(PInvoke.BoosterCalcNumPredict(Handle, numRow, (PInvoke.CApiPredictType)predType, numIteration, ref outLen),
                                  nameof(PInvoke.BoosterCalcNumPredict));
            return outLen;
        }


        public (Tree.Ensemble, Parameters) GetModel() // int[] categoricalFeatureBoudaries)
        {
            Tree.Ensemble res = new Tree.Ensemble();
            string modelString = GetModelString();
            string[] lines = modelString.Split('\n');
            var prms = new Dictionary<string, string>();
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
                        if (kv.Length != 2) throw new FormatException();
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
                        var thresholdDbl = Str2DoubleArray(kvPairs["threshold"], ' ');
                        var splitGain = Str2DoubleArray(kvPairs["split_gain"], ' ');
                        var leafOutput = Str2DoubleArray(kvPairs["leaf_value"], ' ');
                        var decisionType = Str2UIntArray(kvPairs["decision_type"], ' ');

                        //var thresholdSgl = new float[thresholdDbl.Length];
                        //for (var j = 0; j < thresholdDbl.Length; j++)
                        //{
                        //    // See 'AvoidInf' in lightgbm source
                        //    var t = thresholdDbl[j];
                        //    if (t == 1e300)
                        //        thresholdSgl[j] = secondBiggestFloat;
                        //    else if (t == -1e300)
                        //        thresholdSgl[j] = secondSmallestFloat;
                        //    else
                        //        thresholdSgl[j] = (float)t;
                        //}

                        for (var j = 0; j < thresholdDbl.Length; j++)
                        {
                            // See 'AvoidInf' in lightgbm source
                            var t = thresholdDbl[j];
                            if (t == 1e300)
                                thresholdDbl[j] = double.PositiveInfinity;
                            else if (t == -1e300)
                                thresholdDbl[j] = double.NegativeInfinity;
                        }

                        //var defaultValue = GetDefaultValue(thresholdSgl, decisionType);
                        var defaultValue = GetDefaultValue(thresholdDbl, decisionType);

                        var categoricalSplit = new bool[numLeaves - 1];
                        var catBoundaries = Array.Empty<int>();
                        var catThresholds = Array.Empty<uint>();
                        if (numCat > 0)
                        {
                            catBoundaries = Str2IntArray(kvPairs["cat_boundaries"], ' ');
                            catThresholds = Str2UIntArray(kvPairs["cat_threshold"], ' ');
                            for (int node = 0; node < numLeaves - 1; ++node)
                            {
                                categoricalSplit[node] = GetIsCategoricalSplit(decisionType[node]);
                            }
                        }

                        var tree = Tree.RegressionTree.Create(
                                        numLeaves,
                                        splitFeature,
                                        splitGain,
                                        thresholdDbl, //thresholdSgl,
                                        defaultValue,
                                        leftChild,
                                        rightChild,
                                        leafOutput,
                                        catBoundaries,
                                        catThresholds,
                                        categoricalSplit);
                        res.AddTree(tree);
                    }
                    else
                    {
                        //var tree = new Tree.RegressionTree(2);
                        var leafOutput = Str2DoubleArray(kvPairs["leaf_value"], ' ');
                        //if (leafOutput[0] != 0) // still need to add tree, e.g., for multiclass
                        {
                            var tree = Tree.RegressionTree.Create(
                                        2,
                                        new int[] { 0 },
                                        new double[] { 0 },
                                        new double[] { 0 },
                                        new double[] { 0 },
                                        new int[] { -1 },
                                        new int[] { -2 },
                                        new double[] { leafOutput[0], leafOutput[0] },
                                        new int[] { },
                                        new uint[] { },
                                        new bool[] { false });
                            res.AddTree(tree);
                        }
                    }
                }
                else
                {
                    // [objective: binary]
                    if (lines[i].StartsWith("["))
                    {
                        var bits = lines[i].Split(new char[] { '[', ']', ' ', ':' }, StringSplitOptions.RemoveEmptyEntries);
                        if (bits.Length == 2)   // ignores, e.g. [data: ]
                            prms.Add(bits[0], bits[1]);
                    }
                    ++i;
                }
            }

            // extract parameters
            var p = new Parameters {
                Common = _helperCommon.FromParameters(prms),
                Dataset = _helperDataset.FromParameters(prms),
                Metric = _helperMetric.FromParameters(prms),
                Objective = _helperObjective.FromParameters(prms),
                Learning = _helperLearning.FromParameters(prms)
                };
            
            return (res, p);
        }
 
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
