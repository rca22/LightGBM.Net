using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace LightGBMNet.Interface
{
    /// <summary>
    /// Wrapper of Booster object of LightGBM.
    /// </summary>
    internal sealed class Booster : IDisposable
    {
        private readonly bool _hasValid;
        private readonly bool _hasMetric;

        public IntPtr Handle { get; private set; }
        public int BestIteration { get; set; }

        public Booster(Dictionary<string, string> parameters, Dataset trainset, Dataset validset = null)
        {
            var param = ParamsHelper.JoinParameters(parameters);
            var handle = IntPtr.Zero;
            PInvokeException.Check(PInvoke.BoosterCreate(trainset.Handle, param, ref handle),nameof(PInvoke.BoosterCreate));
            Handle = handle;
            if (validset != null)
            {
                PInvokeException.Check(PInvoke.BoosterAddValidData(Handle, validset.Handle),nameof(PInvoke.BoosterAddValidData));
                _hasValid = true;
            }

            int numEval = 0;
            BestIteration = -1;
            PInvokeException.Check(PInvoke.BoosterGetEvalCounts(Handle, ref numEval),nameof(PInvoke.BoosterGetEvalCounts));
            // At most one metric in ML.NET.
            if (numEval > 1)
                throw new Exception("Expected at most one metric");
            else if (numEval == 1)
                _hasMetric = true;
        }

        public bool Update()
        {
            int isFinished = 0;
            PInvokeException.Check(PInvoke.BoosterUpdateOneIter(Handle, ref isFinished),
                                   nameof(PInvoke.BoosterUpdateOneIter));
            return isFinished == 1;
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

        private unsafe string GetModelString()
        {
            int bufLen = 2 << 15;
            byte[] buffer = new byte[bufLen];
            int size = 0;
            fixed (byte* ptr = buffer)
                PInvokeException.Check(PInvoke.BoosterSaveModelToString(Handle, BestIteration, bufLen, ref size, ptr),
                                       nameof(PInvoke.BoosterSaveModelToString));
            // If buffer size is not enough, reallocate buffer and get again.
            if (size > bufLen)
            {
                bufLen = size;
                buffer = new byte[bufLen];
                fixed (byte* ptr = buffer)
                    PInvokeException.Check(PInvoke.BoosterSaveModelToString(Handle, BestIteration, bufLen, ref size, ptr),
                                           nameof(PInvoke.BoosterSaveModelToString));
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

        private static double[] GetDefalutValue(double[] threshold, UInt32[] decisionType)
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

        public string[] FeatureNames
        {
            get
            {
                int cnt = this.NumFeatures;
                int cnt2 = 0;

                IntPtr[] res = new IntPtr[cnt];
                PInvokeException.Check(PInvoke.BoosterGetFeatureNames(Handle, ref cnt2, ref res),
                                       nameof(PInvoke.BoosterGetFeatureNames));
                if (cnt != cnt2)
                    throw new Exception("Expected Feature Names to have consistent size with number of features");
                string[] ret = new string[cnt2];
                for (int i = 0; i < cnt; ++i)
                {
                    ret[i] = Marshal.PtrToStringAnsi(res[i]);
                }
                return ret;
            }
        }
        // Get the importance of a feature.
        public unsafe double[] GetFeatureImportance(int numIteration, PInvoke.CApiImportanceType importanceType)
        {
            // Get the number of features
            int cnt = this.NumFeatures;

            double[] res = new double[cnt];
            fixed (double* ptr = res)
                PInvokeException.Check(PInvoke.BoosterFeatureImportance(Handle, numIteration, importanceType, ptr),
                                       nameof(PInvoke.BoosterFeatureImportance));
            return res;
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
