// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace LightGBMNet.Tree
{
    public class RegressionTree
    {
        [DebuggerDisplay("SplitFeature: {SplitFeature} Threshold: {RawThreshold} LteChild: {LteChild} GtChild: {GtChild} DefaultValue: {DefaultValueForMissing} Categorical: {CategoricalSplit}")]
        public class Node
        {
            public int LteChild { get; set; }
            public int GtChild { get; set; }
            public int SplitFeature { get; set; }
            public double SplitGain { get; set; }
            public bool CategoricalSplit { get; set; }
            public double RawThreshold { get; set; }
            public double DefaultValueForMissing { get; set; }

            public Node() { }

            public void WriteBinary(BinaryWriter writer)
            {
                writer.Write(LteChild);
                writer.Write(GtChild);
                writer.Write(SplitFeature);
                writer.Write(SplitGain);
                writer.Write(CategoricalSplit);
                writer.Write(RawThreshold);
                writer.Write(DefaultValueForMissing);
            }

            public static Node ReadBinary(BinaryReader reader)
            {
                var node = new Node()
                {
                    LteChild = reader.ReadInt32(),
                    GtChild = reader.ReadInt32(),
                    SplitFeature = reader.ReadInt32(),
                    SplitGain = reader.ReadDouble(),
                    CategoricalSplit = reader.ReadBoolean(),
                    RawThreshold = reader.ReadDouble(),
                    DefaultValueForMissing = reader.ReadDouble()
                };
                return node;
            }
        }

        public Node [] Nodes{ get; }

        /// <summary>
        /// Array of categorical values for the categorical feature that might be chosen as 
        /// a split feature for a node.
        /// </summary>
        public uint[] CategoricalThresholds { get; }
        public int[] CategoricalBoundaries { get; }

        public double[] LeafValues { get; }

        // Linear trees only
        public bool IsLinear { get;  }
        public double[] LeafConst { get; }
        public int[][] LeafFeatures { get; }
        public double[][] LeafCoeff { get; }

        public int NumLeaves => LeafValues.Length;

        /// <summary>
        /// constructs a regression tree with an upper bound on depth
        /// </summary>
        public RegressionTree(int numLeaves)
        {
            Nodes = new Node[numLeaves - 1];
            for (int i = 0; i < Nodes.Length; i++)
                Nodes[i] = new Node();
            LeafValues = new double[numLeaves];
        }

        /// <summary>
        /// Create a Regression Tree object from raw tree contents.
        /// </summary>
        public static RegressionTree Create(int numLeaves, int[] splitFeatures, double[] splitGain,
            double[] rawThresholds, double[] defaultValueForMissing, int[] lteChild, int[] gtChild, double[] leafValues,
            int[] categoricalBoundaries, uint[] categoricalThresholds, bool[] categoricalSplit, bool isLinear, double [] leafConst, int[][] leafFeatures, double [][] leafCoeff)
        {
            var numCat = categoricalSplit.Where(x => x).Count();
            CheckParam(numLeaves - 1 == Size(splitFeatures), nameof(splitFeatures), "Size error, should equal to numLeaves - 1.");
            CheckParam(numLeaves - 1 == Size(splitGain), nameof(splitGain), "Size error, should equal to numLeaves - 1.");
            CheckParam(numLeaves - 1 == Size(rawThresholds), nameof(rawThresholds), "Size error, should equal to numLeaves - 1.");
            CheckParam(numLeaves - 1 == Size(lteChild), nameof(lteChild), "Size error, should equal to numLeaves - 1.");
            CheckParam(numLeaves - 1 == Size(gtChild), nameof(gtChild), "Size error, should equal to numLeaves - 1.");
            CheckParam(numLeaves - 1 == Size(defaultValueForMissing), nameof(defaultValueForMissing), "Size error, should equal to numLeaves - 1.");
            CheckParam(numLeaves == Size(leafValues), nameof(leafValues), "Size error, should equal to numLeaves.");
            if (isLinear)
            {
                CheckParam(null != leafConst, nameof(leafConst), "Should not be null if isLinear=true.");
                CheckParam(null != leafFeatures, nameof(leafFeatures), "Should not be null if isLinear=true.");
                CheckParam(null != leafCoeff, nameof(leafCoeff), "Should not be null if isLinear=true.");
                CheckParam(numLeaves == Size(leafConst), nameof(leafConst), "Size error, should equal to numLeaves.");
                CheckParam(numLeaves == Size(leafFeatures), nameof(leafFeatures), "Size error, should equal to numLeaves.");
                CheckParam(numLeaves == Size(leafCoeff), nameof(leafCoeff), "Size error, should equal to numLeaves.");
                for (var i = 0; i < numLeaves; i++)
                {
                    CheckParam(null != leafFeatures[i], nameof(leafFeatures), $"Should not contain null array at index {i} if isLinear=true.");
                    CheckParam(null != leafCoeff[i], nameof(leafCoeff), $"Should not contain null array at index {i} if isLinear=true.");
                    CheckParam(leafFeatures[i].Length == leafCoeff[i].Length, nameof(leafCoeff), $"Arrays leafFeatures and leafCoeff should have same length at index {i} if isLinear=true.");
                }
            }
            else
            {
                CheckParam(null == leafConst, nameof(leafConst), "Should be null if isLinear=false.");
                CheckParam(null == leafFeatures, nameof(leafFeatures), "Should be null if isLinear=false.");
                CheckParam(null == leafCoeff, nameof(leafCoeff), "Should be null if isLinear=false.");
            }
            // TODO: size depends on range of categorical variables, below only true for small range
            //CheckParam((numCat > 0 ? numCat + 1 : 0) == Size(categoricalBoundaries), nameof(categoricalBoundaries), "Size error, should equal to numCat + 1.");
            //CheckParam(numCat == Size(categoricalThresholds), nameof(categoricalThresholds), "Size error, should equal to numCat.");
            CheckParam(numLeaves - 1 == Size(categoricalSplit), nameof(categoricalSplit), "Size error, should equal to numLeaves - 1.");
            return new RegressionTree(splitFeatures, splitGain, /*null,*/ rawThresholds, defaultValueForMissing, lteChild, gtChild, leafValues, categoricalBoundaries, categoricalThresholds, categoricalSplit, isLinear, leafConst, leafFeatures, leafCoeff);
        }

        internal RegressionTree(int[] splitFeatures, double[] splitGains,
            double[] rawThresholds, double[] defaultValueForMissing, int[] lteChild, int[] gtChild, double[] leafValues,
            int[] categoricalBoundaries, uint[] categoricalThresholds, bool[] categoricalSplit, bool isLinear, double[] leafConst, int[][] leafFeatures, double[][] leafCoeff)
        {
            Nodes = new Node[Size(splitFeatures)];
            for (int i = 0; i < Nodes.Length; i++)
            {
                var node = new Node()
                {
                    SplitFeature = splitFeatures[i],
                    SplitGain = splitGains[i],
                    RawThreshold = rawThresholds[i],
                    DefaultValueForMissing = (defaultValueForMissing != null) ? defaultValueForMissing[i] : Double.NaN,
                    LteChild = lteChild[i],
                    GtChild = gtChild[i],
                    CategoricalSplit = categoricalSplit[i]
                };
                Nodes[i] = node;
            }
            LeafValues = leafValues;
            CategoricalBoundaries = categoricalBoundaries;
            CategoricalThresholds = categoricalThresholds;
            
            IsLinear = isLinear;
            LeafConst = leafConst;
            LeafFeatures = leafFeatures;
            LeafCoeff = leafCoeff;

            CheckValid(Check);
        }

        #region Read/write arrays to/from binary stream
        public static void WriteIntArray(BinaryWriter writer, int[] values)
        {
            if (values == null)
            {
                writer.Write(-1);
                return;
            }

            writer.Write(values.Length);
            foreach (int val in values)
                writer.Write(val);
        }

        public static void WriteIntArrayArray(BinaryWriter writer, int[][] values)
        {
            if (values == null)
            {
                writer.Write(-1);
                return;
            }

            writer.Write(values.Length);
            foreach (int[] val in values)
                WriteIntArray(writer, val);
        }

        private static void WriteUIntArray(BinaryWriter writer, uint[] values)
        {
            if (values == null)
            {
                writer.Write(-1);
                return;
            }

            writer.Write(values.Length);
            foreach (uint val in values)
                writer.Write(val);
        }

        public static void WriteDoubleArray(BinaryWriter writer, double[] values)
        {
            if (values == null)
            {
                writer.Write(-1);
                return;
            }

            writer.Write(values.Length);
            foreach (double val in values)
                writer.Write(val);
        }

        public static void WriteDoubleArrayArray(BinaryWriter writer, double[][] values)
        {
            if (values == null)
            {
                writer.Write(-1);
                return;
            }

            writer.Write(values.Length);
            foreach (double[] val in values)
                WriteDoubleArray(writer, val);
        }

        private static int[] ReadIntArray(BinaryReader reader, int size)
        {
            if (size < -1) throw new FormatException();
            if (size == -1) return null;
            var values = new int[size];
            for (int i = 0; i < size; i++)
                values[i] = reader.ReadInt32();
            return values;
        }

        private static int[] ReadIntArray(BinaryReader reader)
        {
            int size = reader.ReadInt32();
            return ReadIntArray(reader, size);
        }

        private static int[][] ReadIntArrayArray(BinaryReader reader)
        {
            int size = reader.ReadInt32();
            if (size < -1) throw new FormatException();
            if (size == -1) return null;
            var values = new int[size][];
            for (int i = 0; i < size; i++)
                values[i] = ReadIntArray(reader);
            return values;
        }

        private static uint[] ReadUIntArray(BinaryReader reader)
        {
            int size = reader.ReadInt32();
            if (size < -1) throw new FormatException();
            if (size == -1) return null;
            var values = new uint[size];
            for (int i = 0; i < size; i++)
                values[i] = reader.ReadUInt32();
            return values;
        }

        private static double[] ReadDoubleArray(BinaryReader reader)
        {
            int size = reader.ReadInt32();
            if (size < -1) throw new FormatException();
            if (size == -1) return null;
            var values = new double[size];
            for (int i = 0; i < size; i++)
                values[i] = reader.ReadDouble();
            return values;
        }

        private static double[][] ReadDoubleArrayArray(BinaryReader reader)
        {
            int size = reader.ReadInt32();
            if (size < -1) throw new FormatException();
            if (size == -1) return null;
            var values = new double[size][];
            for (int i = 0; i < size; i++)
                values[i] = ReadDoubleArray(reader);
            return values;
        }
        #endregion

        #region Read/write RegressionTree to binary stream
        internal RegressionTree(BinaryReader reader, bool legacyVersion)
        {
            Nodes = new Node[reader.ReadInt32()];
            for (int i = 0; i < Nodes.Length; i++)
                Nodes[i] = Node.ReadBinary(reader);

            CategoricalBoundaries = ReadIntArray(reader);
            CategoricalThresholds = ReadUIntArray(reader);

            LeafValues = ReadDoubleArray(reader);

            IsLinear = legacyVersion ? false : reader.ReadBoolean();
            if (IsLinear)
            {
                LeafConst = ReadDoubleArray(reader);
                LeafFeatures = ReadIntArrayArray(reader);
                LeafCoeff = ReadDoubleArrayArray(reader);
            }
            else
            {
                LeafConst = null;
                LeafFeatures = null;
                LeafCoeff = null;
            }

            CheckValid(CheckDecode);
        }

        public void Save(BinaryWriter writer)
        {
            writer.Write(Nodes.Length);
            foreach (var node in Nodes)
                node.WriteBinary(writer);

            WriteIntArray(writer, CategoricalBoundaries);
            WriteUIntArray(writer, CategoricalThresholds);

            WriteDoubleArray(writer, LeafValues);

            writer.Write(IsLinear);
            if (IsLinear)
            {
                WriteDoubleArray(writer, LeafConst);
                WriteIntArrayArray(writer, LeafFeatures);
                WriteDoubleArrayArray(writer, LeafCoeff);
            }
        }

        public static RegressionTree Load(BinaryReader reader, bool legacyVersion)
        {
            return new RegressionTree(reader, legacyVersion);
        }
        #endregion


        private void CheckValid(Action<bool, string> checker)
        {
            int numMaxNodes = Size(Nodes);
            int numMaxLeaves = numMaxNodes + 1;
            checker(NumLeaves >= 1, "non-positive number of leaves");
            checker(numMaxLeaves >= NumLeaves, "inconsistent number of leaves with maximum leaf capacity");
            checker(LeafValues != null && LeafValues.Length == numMaxLeaves, "bad leaf value length");
        }

        public virtual double GetOutput(ref VBuffer<float> feat)
        {
            int leaf = GetLeaf(ref feat);
            if (IsLinear)
            {
                double output = LeafConst[leaf];
                bool nan_found = false;
                var leafFeatures = LeafFeatures[leaf];
                var leafCoeff = LeafCoeff[leaf];
                for (var i = 0; i < leafFeatures.Length; ++i)
                {
                    int feat_raw = leafFeatures[i];
                    double feat_val = 0.0;
                    if (feat.IsDense)
                    {
                        feat_val = feat.Values[feat_raw];
                    }
                    else
                    {
                        int ii = VBuffer<int>.FindIndexSorted(feat.Indices, 0, feat.Count, feat_raw);
                        if (ii < feat.Count && feat.Indices[ii] == feat_raw)
                            feat_val = feat.Values[ii];
                    }
                    if (Double.IsNaN(feat_val))
                    {
                        nan_found = true;
                        break;
                    }
                    else
                    {
                        output += leafCoeff[i] * feat_val;
                    }
                }
                if (nan_found)
                    return GetOutput(leaf);
                else
                    return output;
            }
            else
            {
                return GetOutput(leaf);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetOutput(int leaf)
        {
            return LeafValues[leaf];
        }

        public void SetOutput(int leaf, double value)
        {
            LeafValues[leaf] = value;
        }
        
        // Returns index to a leaf an instance/document belongs to.
        // Input are the raw feature values in dense format.
        // For empty tree returns 0.
        public int GetLeaf(ref VBuffer<float> feat)
        {
            if (feat.IsDense)
                return GetLeafCore(feat.Values);
            return GetLeafCore(feat.Count, feat.Indices, feat.Values);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double GetFeatureValue(double x, Node node)
        {
            if (double.IsNaN(x))
                return node.DefaultValueForMissing;
            else
                return x;
        }

        private static bool FindInBitset(uint[] bits, int start, int end, int pos)
        {
            int i1 = pos / 32;
            if (i1 >= end)
                return false;
            int i2 = pos % 32;
            return ((bits[start + i1] >> i2) & 1) > 0;
        }

        private int GetLeafCore(float[] nonBinnedInstance, int root = 0)
        {
            Debug.Assert(nonBinnedInstance != null);
            Debug.Assert(root >= 0);

            // Check for an empty tree.
            if (NumLeaves == 1)
                return 0;

            int node = root;
            while (node >= 0)
            {
                Node nodeObj = Nodes[node];
                double fv = nonBinnedInstance[nodeObj.SplitFeature];
                if (nodeObj.CategoricalSplit)
                {
                    Debug.Assert(CategoricalThresholds != null);
                    Debug.Assert(CategoricalBoundaries != null);

                    int int_fval = (int)fv;
                    int cat_idx = (int)nodeObj.RawThreshold;

                    if (int_fval >= 0 && !double.IsNaN(fv) && FindInBitset(CategoricalThresholds, CategoricalBoundaries[cat_idx],
                         CategoricalBoundaries[cat_idx + 1] - CategoricalBoundaries[cat_idx], int_fval))
                        node = nodeObj.LteChild;
                    else
                        node = nodeObj.GtChild;
                }
                else
                {
                    fv = GetFeatureValue(fv, nodeObj);
                    if (fv <= nodeObj.RawThreshold)
                        node = nodeObj.LteChild;
                    else
                        node = nodeObj.GtChild;
                }
            }

            return ~node;
        }

        private int GetLeafCore(int count, int[] featIndices, float[] featValues, int root = 0)
        {
            Debug.Assert(count >= 0);
            Debug.Assert(Size(featIndices) >= count);
            Debug.Assert(Size(featValues) >= count);
            Debug.Assert(root >= 0);

            // check for an empty tree
            if (NumLeaves == 1)
                return 0;

            int node = root;

            while (node >= 0)
            {
                Node nodeObj = Nodes[node];

                double val = 0;
                int ifeat = nodeObj.SplitFeature;
                int ii = VBuffer<int>.FindIndexSorted(featIndices, 0, count, ifeat);
                if (ii < count && featIndices[ii] == ifeat)
                    val = featValues[ii];

                if (nodeObj.CategoricalSplit)
                {
                    Debug.Assert(CategoricalThresholds != null);
                    Debug.Assert(CategoricalBoundaries != null);
                    
                    int int_fval = (int)val;
                    int cat_idx = (int)nodeObj.RawThreshold;

                    if (int_fval >= 0 && !double.IsNaN(val) && FindInBitset(CategoricalThresholds, CategoricalBoundaries[cat_idx],
                         CategoricalBoundaries[cat_idx + 1] - CategoricalBoundaries[cat_idx], int_fval))
                        node = nodeObj.LteChild;
                    else
                        node = nodeObj.GtChild;
                }
                else
                {
                    val = GetFeatureValue(val, nodeObj);
                    if (val <= nodeObj.RawThreshold)
                        node = nodeObj.LteChild;
                    else
                        node = nodeObj.GtChild;
                }
            }
            return ~node;
        }

        public FeatureToGainMap GainMap
        {
            get
            {
                var result = new FeatureToGainMap();
                foreach(var node in Nodes)
                {
                    if (node.SplitGain != 0)
                        result[node.SplitFeature] += node.SplitGain;
                }
                return result;
            }
        }

        public FeatureToGainMap SplitMap
        {
            get
            {
                var result = new FeatureToGainMap();
                foreach (var node in Nodes)
                {
                    if (node.SplitGain > 0)
                        result[node.SplitFeature] += 1;
                }
                return result;
            }
        }

        public IEnumerable<double> FeatureGains(int feature)
        {
            foreach (var node in Nodes)
                if (feature == node.SplitFeature)
                    yield return node.SplitGain;
        }

        #region check helpers
        private static void Check(bool f, string msg)
        {
            if (!f)
                throw new InvalidOperationException(msg);
        }

        private static void CheckDecode(bool f, string msg)
        {
            if (!f)
                throw new FormatException(msg);
        }

        private static void CheckParam(bool f, string paramName, string msg)
        {
            if (!f)
                throw new ArgumentOutOfRangeException(paramName, msg);
        }

        private static int Size<T>(T[] x)
        {
            return x == null ? 0 : x.Length;
        }
        #endregion
    }
}
