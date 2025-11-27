// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Diagnostics;

namespace LightGBMNet.Tree
{
    public class Ensemble
    {
        private readonly List<RegressionTree> _trees;

        public IEnumerable<RegressionTree> Trees => _trees;

        public int NumTrees => _trees.Count;

        private int _maxThreads = 1;
        public int MaxThreads
        {
            get => _maxThreads;
            set {
                    if (value <= 0) throw new ArgumentException(nameof(MaxThreads), "Must be positive");
                    _maxThreads = value;
                }
        }

        // cached calculation of groups of ensemble indices calculated by each thread
        private int _groupMaxThreads = 1;
        private int _groupNumTrees = 0;
        private (int, int) [] _groups = null;
        private double[] _groupVals = null;

        private void SetGroups(int numTrees)
        {
            // _groups calculation already cached?
            if (_groupMaxThreads == MaxThreads && _groupNumTrees == numTrees)
                return;

            if (_maxThreads == 1 || numTrees == 0)
            {
                _groups = null;
                _groupVals = null;
            }
            else
            {
                var numGroups = Math.Min(numTrees, MaxThreads);
                var per = Math.DivRem(numTrees, numGroups, out var stub);
                _groups = new (int, int)[numGroups];
                _groupVals = new double[numGroups];
                var idx0 = 0;
                for (var i=0; i < _groups.Length; i++)
                {
                    var size = (i < stub) ? per + 1 : per;
                    var idx1 = idx0 + size;
                    _groups[i] = (idx0, idx1);
                    idx0 = idx1;
                }
                Debug.Assert(idx0 == numTrees);
            }
            _groupMaxThreads = MaxThreads;
            _groupNumTrees = numTrees;
        }

        private static ParamsHelper<CommonParameters> _helperCommon = new ParamsHelper<CommonParameters>();
        private static ParamsHelper<DatasetParameters> _helperDataset = new ParamsHelper<DatasetParameters>();
        private static ParamsHelper<ObjectiveParameters> _helperObjective = new ParamsHelper<ObjectiveParameters>();
        private static ParamsHelper<LearningParameters> _helperLearning = new ParamsHelper<LearningParameters>();

        public Ensemble()
        {
            _trees = new List<RegressionTree>();
        }

        public Ensemble(BinaryReader reader, bool legacyVersion)
        {
            int numTrees = reader.ReadInt32();
            if(!(numTrees >= 0)) throw new FormatException();
            _trees = new List<RegressionTree>(numTrees);
            for (int t = 0; t < numTrees; ++t)
                AddTree(RegressionTree.Load(reader, legacyVersion));
            MaxThreads = reader.ReadInt32();
        }

        public void Save(BinaryWriter writer)
        {
            writer.Write(NumTrees);
            foreach (RegressionTree tree in Trees)
                tree.Save(writer);
            writer.Write(MaxThreads);
        }

        public void AddTree(RegressionTree tree) => _trees.Add(tree);
        public void AddTreeAt(RegressionTree tree, int index) => _trees.Insert(index, tree);
        public void RemoveTree(int index) => _trees.RemoveAt(index);
        public void RemoveAfter(int index) => _trees.RemoveRange(index, NumTrees - index);
        public RegressionTree GetTreeAt(int index) => _trees[index];

        public double GetOutput(ref VBuffer<float> feat, int startIteration, int numIterations)
        {
            int numTrees = (numIterations == -1) ? NumTrees - startIteration : numIterations;
            SetGroups(numTrees);

            double result = 0.0;

            if (_maxThreads > 1 && numTrees > 0)
            {
                var featcopy = feat;
                Parallel.For(0, _groups.Length, i =>
                {
                    double output = 0.0;
                    (int lo, int hi) = _groups[i];
                    for (int h = lo; h < hi; h++)
                        output += _trees[startIteration+h].GetOutput(ref featcopy);
                    _groupVals[i] = output;
                }
                );
                result = _groupVals.Sum();
            }
            else
            {
                for (int h = 0; h < numTrees; h++)
                    result += _trees[startIteration+h].GetOutput(ref feat);
            }

            return result;
        }

        private static double[] Str2DoubleArray(string str, char[] delimiters)
        {
            return str.Split(delimiters, StringSplitOptions.RemoveEmptyEntries)
                      .Select(s => double.TryParse(s.Replace("inf", "Infinity"), System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out double rslt) ? rslt :
                                    (s.Contains("nan") ? double.NaN : throw new Exception($"Cannot parse as double: {s}")))
                      .ToArray();
        }

        private static int[] Str2IntArray(string str, char[] delimiters)
        {
            return str.Split(delimiters, StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
        }

        private static uint[] Str2UIntArray(string str, char[] delimiters)
        {
            return str.Split(delimiters, StringSplitOptions.RemoveEmptyEntries).Select(uint.Parse).ToArray();
        }

        private static bool GetIsDefaultLeft(uint decisionType)
        {
            // The second bit.
            return (decisionType & 2) > 0;
        }

        private static bool GetIsCategoricalSplit(uint decisionType)
        {
            // The first bit.
            return (decisionType & 1) > 0;
        }

        private static bool GetHasMissing(uint decisionType)
        {
            // The 3rd and 4th bits.
            return ((decisionType >> 2) & 3) > 0;
        }

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

        private static double[] GetDefaultValue(double[] threshold, uint[] decisionType)
        {
            var ret = new double[threshold.Length];
            for (int i = 0; i < threshold.Length; ++i)
            {
                if (GetHasMissing(decisionType[i]) && !GetIsCategoricalSplit(decisionType[i])) // NOTE: categorical always take RHS branch for missing
                {
                    // need to be careful here to generate a value that is genuinely LEQ for left branch, and GT for right branch!
                    var t = threshold[i];
                    if (GetIsDefaultLeft(decisionType[i]))
                        ret[i] = t;
                    else
                        ret[i] = (t == 0.0f) ? +1.0f : ((t > 0) ? NextUp(t) : NextDown(t));  // TODO: INFINITY!!!
                }
            }
            return ret;
        }

        public static (Ensemble, Parameters, int) GetModelFromString(string modelString)
        {
            Ensemble res = new Ensemble();
            // Note that this doesn't allow for files being checked out in Windows with autocrlf
            string[] lines = modelString.Split('\n');
            var prms = new Dictionary<string, string>();
            var delimiters = new char[] { ' ' };
            int i = 0;
            int max_feature_idx = -1;
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
                    var leftChild = Str2IntArray(kvPairs["left_child"], delimiters);
                    var rightChild = Str2IntArray(kvPairs["right_child"], delimiters);
                    var splitFeature = Str2IntArray(kvPairs["split_feature"], delimiters);
                    var threshold = Str2DoubleArray(kvPairs["threshold"], delimiters);
                    var splitGain = Str2DoubleArray(kvPairs["split_gain"], delimiters);
                    var leafOutput = Str2DoubleArray(kvPairs["leaf_value"], delimiters);
                    var decisionType = Str2UIntArray(kvPairs["decision_type"], delimiters);

                    for (var j = 0; j < threshold.Length; j++)
                    {
                        // See 'AvoidInf' in lightgbm source
                        var t = threshold[j];
                        if (t == 1e300)
                            threshold[j] = double.PositiveInfinity;
                        else if (t == -1e300)
                            threshold[j] = double.NegativeInfinity;
                    }

                    var defaultValue = GetDefaultValue(threshold, decisionType);

                    var categoricalSplit = new bool[numLeaves - 1];
                    var catBoundaries = Array.Empty<int>();
                    var catThresholds = Array.Empty<uint>();
                    if (numCat > 0)
                    {
                        catBoundaries = Str2IntArray(kvPairs["cat_boundaries"], delimiters);
                        catThresholds = Str2UIntArray(kvPairs["cat_threshold"], delimiters);
                        for (int node = 0; node < numLeaves - 1; ++node)
                        {
                            categoricalSplit[node] = GetIsCategoricalSplit(decisionType[node]);
                        }
                    }

                    double[] leafConst = null;
                    int[][] leafFeaturesUnpacked = null;
                    double[][] leafCoeffUnpacked = null;

                    var isLinear = Int32.Parse(kvPairs["is_linear"]) > 0;
                    if (isLinear)
                    {
                        leafConst = Str2DoubleArray(kvPairs["leaf_const"], delimiters);
                        var numFeatures = Str2IntArray(kvPairs["num_features"], delimiters);
                        var leafFeatures = Str2IntArray(kvPairs["leaf_features"], delimiters);
                        var leafCoeff = Str2DoubleArray(kvPairs["leaf_coeff"], delimiters);

                        leafFeaturesUnpacked = new int[numFeatures.Length][];
                        leafCoeffUnpacked = new double[numFeatures.Length][];
                        var idx = 0;
                        for (var j = 0; j < numFeatures.Length; j++)
                        {
                            var len = numFeatures[j];
                            leafFeaturesUnpacked[j] = new int[len];
                            leafCoeffUnpacked[j] = new double[len];
                            for (var k = 0; k < len; k++)
                            {
                                leafFeaturesUnpacked[j][k] = leafFeatures[idx];
                                leafCoeffUnpacked[j][k] = leafCoeff[idx];
                                idx++;
                            }
                        }
                        if (idx != leafFeatures.Length)
                            throw new Exception("Failed to parse leaf features");
                    }

                    var tree = RegressionTree.Create(
                                    numLeaves,
                                    splitFeature,
                                    splitGain,
                                    threshold,
                                    defaultValue,
                                    leftChild,
                                    rightChild,
                                    leafOutput,
                                    catBoundaries,
                                    catThresholds,
                                    categoricalSplit,
                                    isLinear,
                                    leafConst,
                                    leafFeaturesUnpacked,
                                    leafCoeffUnpacked);
                    res.AddTree(tree);
                }
                else
                {
                    // [objective: binary]
                    if (lines[i].StartsWith("["))
                    {
                        var bits = lines[i].Split(new char[] { '[', ']', ' ', ':'}, StringSplitOptions.RemoveEmptyEntries);
                        if (bits.Length == 2)   // ignores, e.g. [data: ]
                            prms.Add(bits[0], bits[1]);
                    }
                    else if (lines[i].StartsWith("max_feature_idx="))
                    {
                        var bits = lines[i].Split(new char[] { '=' }, StringSplitOptions.RemoveEmptyEntries);
                        if (bits.Length == 2)   // ignores, e.g. [data: ]
                            max_feature_idx = Int32.Parse(bits[1]);
                        else
                            throw new Exception($"Failed to parse {lines[i]}");
                    }
                    ++i;
                }
            }

            if (max_feature_idx < 0)
                throw new Exception("Failed to detect max_feature_idx");
            int maxNumFeatures = max_feature_idx + 1;

            // extract parameters
            var p = new Parameters
            {
                Common = _helperCommon.FromParameters(prms),
                Dataset = _helperDataset.FromParameters(prms),
                Objective = _helperObjective.FromParameters(prms),
                Learning = _helperLearning.FromParameters(prms)
            };

            // irrelevant parameter for managed trees which always use NaN for missing value
            prms.Remove("zero_as_missing");
            prms.Remove("saved_feature_importance_type");
            if (prms.Count > 0)
            {
                Console.WriteLine($"WARNING: Unknown new parameters {String.Join(",", prms.Keys)}");
            }

            return (res, p, maxNumFeatures);
        }

    }

    public class FeatureToGainMap : Dictionary<int, double>
    {
        public FeatureToGainMap() { }
        // Override default Dictionary to return 0.0 for non-existing keys
        public new double this[int key] {
            get {
                TryGetValue(key, out double retval);
                return retval;
            }
            set {
                base[key] = value;
            }
        }

        public FeatureToGainMap(IList<RegressionTree> trees, bool normalize = false, bool splits = false)
        {
            if (trees.Count == 0)
                return;

            IList<int> combinedKeys = null;
            for (int iteration = 0; iteration < trees.Count; iteration++)
            {
                FeatureToGainMap currentGains = splits ? trees[iteration].SplitMap : trees[iteration].GainMap;
                combinedKeys = Keys.Union(currentGains.Keys).Distinct().ToList();
                foreach (var k in combinedKeys)
                    this[k] += currentGains[k];
            }
            if (normalize)
            {
                foreach (var k in combinedKeys)
                    this[k] = this[k] / trees.Count;
            }
        }
    }

}
