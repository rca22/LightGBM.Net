// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Diagnostics;

namespace LightGBMNet.Tree
{
    public class RegressionTree
    {
        private double[] DefaultValueForMissing { get; }

        public int[] LteChild { get; }
        public int[] GtChild { get; }
        public int[] SplitFeatures { get; }
        /// <summary>
        /// Indicates if a node's split feature was categorical.
        /// </summary>
        public bool[] CategoricalSplit { get; }
        /// <summary>
        /// Array of categorical values for the categorical feature that might be chosen as 
        /// a split feature for a node.
        /// </summary>
        public UInt32[] CategoricalThresholds { get; }
        public int[] CategoricalBoundaries { get; }
        // These are the thresholds based on the binned values of the raw features.
        //public UInt32[] Thresholds { get; }
        // These are the thresholds based on the raw feature values. Populated after training.
        public double[] RawThresholds { get; }
        public double[] SplitGains { get; }
        public double[] LeafValues { get; }

        /// <summary>
        /// constructs a regression tree with an upper bound on depth
        /// </summary>
        public RegressionTree(int maxLeaves)
        {
            SplitFeatures = new int[maxLeaves - 1];
            CategoricalSplit = new bool[maxLeaves - 1];
            SplitGains = new double[maxLeaves - 1];
            DefaultValueForMissing = null;
            LteChild = new int[maxLeaves - 1];
            GtChild = new int[maxLeaves - 1];
            LeafValues = new double[maxLeaves];
            NumLeaves = 1;
        }
        

        /// <summary>
        /// Create a Regression Tree object from raw tree contents.
        /// </summary>
        public static RegressionTree Create(int numLeaves, int[] splitFeatures, Double[] splitGain,
            double[] rawThresholds, double[] defaultValueForMissing, int[] lteChild, int[] gtChild, Double[] leafValues,
            int[] categoricalBoundaries, uint[] categoricalThresholds, bool[] categoricalSplit)
        {
            if (numLeaves <= 1)
            {
                // Create a dummy tree.
                RegressionTree tree = new RegressionTree(2);
                tree.SetOutput(0, 0.0);
                tree.SetOutput(1, 0.0);
                return tree;
            }
            else
            {
                var numCat = categoricalSplit.Where(x => x).Count();
                CheckParam(numLeaves - 1 == Size(splitFeatures), nameof(splitFeatures), "Size error, should equal to numLeaves - 1.");
                CheckParam(numLeaves - 1 == Size(splitGain), nameof(splitGain), "Size error, should equal to numLeaves - 1.");
                CheckParam(numLeaves - 1 == Size(rawThresholds), nameof(rawThresholds), "Size error, should equal to numLeaves - 1.");
                CheckParam(numLeaves - 1 == Size(lteChild), nameof(lteChild), "Size error, should equal to numLeaves - 1.");
                CheckParam(numLeaves - 1 == Size(gtChild), nameof(gtChild), "Size error, should equal to numLeaves - 1.");
                CheckParam(numLeaves - 1 == Size(defaultValueForMissing), nameof(defaultValueForMissing), "Size error, should equal to numLeaves - 1.");
                CheckParam(numLeaves == Size(leafValues), nameof(leafValues), "Size error, should equal to numLeaves.");
              // TODO: size depends on range of categorical variables, below only true for small range
              //CheckParam((numCat > 0 ? numCat + 1 : 0) == Size(categoricalBoundaries), nameof(categoricalBoundaries), "Size error, should equal to numCat + 1.");
              //CheckParam(numCat == Size(categoricalThresholds), nameof(categoricalThresholds), "Size error, should equal to numCat.");
                CheckParam(numLeaves - 1 == Size(categoricalSplit), nameof(categoricalSplit), "Size error, should equal to numLeaves - 1.");
                return new RegressionTree(splitFeatures, splitGain, /*null,*/ rawThresholds, defaultValueForMissing, lteChild, gtChild, leafValues, categoricalBoundaries, categoricalThresholds, categoricalSplit);
            }
        }

        internal RegressionTree(int[] splitFeatures, Double[] splitGains,
            double[] rawThresholds, double[] defaultValueForMissing, int[] lteChild, int[] gtChild, Double[] leafValues,
            int[] categoricalBoundaries, uint[] categoricalThresholds, bool[] categoricalSplit)
        {
            CheckParam(Size(splitFeatures) > 0, nameof(splitFeatures), "Number of split features must be positive");

            NumLeaves = Size(splitFeatures) + 1;
            SplitFeatures = splitFeatures;
            SplitGains = splitGains;
            RawThresholds = rawThresholds;
            DefaultValueForMissing = defaultValueForMissing;
            LteChild = lteChild;
            GtChild = gtChild;
            LeafValues = leafValues;
            CategoricalBoundaries = categoricalBoundaries;
            CategoricalThresholds = categoricalThresholds;
            CategoricalSplit = categoricalSplit;

            CheckValid(Check);

            if (DefaultValueForMissing != null)
            {
                bool allZero = true;
                foreach (var val in DefaultValueForMissing)
                {
                    if (val != 0.0f)
                    {
                        allZero = false;
                        break;
                    }
                }
                if (allZero)
                    DefaultValueForMissing = null;
            }
        }

        #region Read/write arrays to/from binary stream
        public static void WriteIntArray(BinaryWriter writer, int[] values)
        {
            if (values == null)
            {
                writer.Write(0);
                return;
            }

            writer.Write(values.Length);
            foreach (int val in values)
                writer.Write(val);
        }

        private static void WriteUIntArray(BinaryWriter writer, uint[] values)
        {
            if (values == null)
            {
                writer.Write(0);
                return;
            }

            writer.Write(values.Length);
            foreach (uint val in values)
                writer.Write(val);
        }

        private static void WriteBooleanArray(BinaryWriter writer, bool[] values)
        {
            if (values == null)
            {
                writer.Write(0);
                return;
            }

            writer.Write(values.Length);
            foreach (bool val in values)
                writer.Write(val);
        }

        public static void WriteDoubleArray(BinaryWriter writer, Double[] values)
        {
            if (values == null)
            {
                writer.Write(0);
                return;
            }

            writer.Write(values.Length);
            foreach (Double val in values)
                writer.Write(val);
        }

        private static int[] ReadIntArray(BinaryReader reader, int size)
        {
            if (size < 0) throw new FormatException();
            if (size == 0) return null;
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

        private static uint[] ReadUIntArray(BinaryReader reader)
        {
            int size = reader.ReadInt32();
            if (size < 0) throw new FormatException();
            if (size == 0) return null;
            var values = new uint[size];
            for (int i = 0; i < size; i++)
                values[i] = reader.ReadUInt32();
            return values;
        }

        private static bool[] ReadBooleanArray(BinaryReader reader)
        {
            int size = reader.ReadInt32();
            if (size < 0) throw new FormatException();
            if (size == 0) return null;
            var values = new bool[size];
            for (int i = 0; i < size; i++)
                values[i] = reader.ReadBoolean();
            return values;
        }

        private static Double[] ReadDoubleArray(BinaryReader reader)
        {
            int size = reader.ReadInt32();
            if (size < 0) throw new FormatException();
            if (size == 0) return null;
            var values = new Double[size];
            for (int i = 0; i < size; i++)
                values[i] = reader.ReadDouble();
            return values;
        }
        #endregion

        #region Read/write RegressionTree to binary stream
        internal RegressionTree(BinaryReader reader)
        {

            NumLeaves = reader.ReadInt32();
            LteChild = ReadIntArray(reader);
            GtChild = ReadIntArray(reader);
            SplitFeatures = ReadIntArray(reader);

            CategoricalBoundaries = ReadIntArray(reader);
            CategoricalThresholds = ReadUIntArray(reader);
            CategoricalSplit = ReadBooleanArray(reader);

            RawThresholds = ReadDoubleArray(reader);

            DefaultValueForMissing = ReadDoubleArray(reader);

            LeafValues = ReadDoubleArray(reader);
            SplitGains = ReadDoubleArray(reader);

            CheckValid(CheckDecode);

            // Check the need of DefaultValueForMissing
            if (DefaultValueForMissing != null)
            {
                bool allZero = true;
                foreach (var val in DefaultValueForMissing)
                {
                    if (val != 0.0f)
                    {
                        allZero = false;
                        break;
                    }
                }
                if (allZero)
                    DefaultValueForMissing = null;
            }
        }

        public void Save(BinaryWriter writer)
        {
#if DEBUG
            // This must be compiled only in the debug case, since you can't
            // have delegates on functions with conditional attributes.
            CheckValid((t, s) => Debug.Assert(t, s));
#endif
            writer.Write(NumLeaves);

            WriteIntArray(writer, LteChild);
            WriteIntArray(writer, GtChild);
            WriteIntArray(writer, SplitFeatures);

            Debug.Assert(CategoricalSplit != null);
            Debug.Assert(CategoricalSplit.Length >= NumNodes);

            WriteIntArray(writer, CategoricalBoundaries);
            WriteUIntArray(writer, CategoricalThresholds);
            WriteBooleanArray(writer, CategoricalSplit);

            WriteDoubleArray(writer, RawThresholds);
            WriteDoubleArray(writer, DefaultValueForMissing);
            WriteDoubleArray(writer, LeafValues);

            WriteDoubleArray(writer, SplitGains);
        }

        public static RegressionTree Load(BinaryReader reader)
        {
            return new RegressionTree(reader);
        }
        #endregion


        private void CheckValid(Action<bool, string> checker)
        {
            int numMaxNodes = Size(LteChild);
            int numMaxLeaves = numMaxNodes + 1;
            checker(NumLeaves > 1, "non-positive number of leaves");
            checker(numMaxLeaves >= NumLeaves, "inconsistent number of leaves with maximum leaf capacity");
            checker(GtChild != null && GtChild.Length == numMaxNodes, "bad gtchild");
            checker(LteChild != null && LteChild.Length == numMaxNodes, "bad ltechild");
            checker(SplitFeatures != null && SplitFeatures.Length == numMaxNodes, "bad split feature length");
            checker(CategoricalSplit != null &&
                (CategoricalSplit.Length == numMaxNodes || CategoricalSplit.Length == NumNodes), "bad categorical split length");

            if (CategoricalSplit.Any(x => x))
            {
                // TODO: FIXME

            }

            checker(Size(RawThresholds) == 0 || RawThresholds.Length == NumLeaves - 1, "bad rawthreshold length");
            checker(RawThresholds != null, // || Thresholds != null,
                "at most one of raw or indexed thresholds can be null");
            checker(Size(SplitGains) == 0 || SplitGains.Length == numMaxNodes, "bad splitgains length");
            checker(LeafValues != null && LeafValues.Length == numMaxLeaves, "bad leaf value length");
        }

        /// <summary>
        /// The current number of leaves in the tree.
        /// </summary>
        public int NumLeaves { get; private set; }

        /// <summary>
        /// The current number of nodes in the tree.
        /// </summary>
        public int NumNodes => NumLeaves - 1;

        public virtual double GetOutput(ref VBuffer<float> feat)
        {
            if (LteChild[0] == 0)
                return 0;
            int leaf = GetLeaf(ref feat);
            return GetOutput(leaf);
        }

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
            // REVIEW: This really should validate feat.Length!
            if (feat.IsDense)
                return GetLeafCore(feat.Values);
            return GetLeafCore(feat.Count, feat.Indices, feat.Values);
        }

        private double GetFeatureValue(double x, int node)
        {
            // Not need to convert missing vaules.
            if (DefaultValueForMissing == null)
                return x;

            if (Double.IsNaN(x))
            {
                return DefaultValueForMissing[node];
            }
            else
            {
                return x;
            }
        }

        private static bool FindInBitset(UInt32[] bits, int start, int end, int pos)
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
                double fv = nonBinnedInstance[SplitFeatures[node]];
                if (CategoricalSplit[node])
                {
                    Debug.Assert(CategoricalThresholds != null);
                    Debug.Assert(CategoricalBoundaries != null);

                    int int_fval = (int)fv;
                    int cat_idx = (int)RawThresholds[node];

                    if (int_fval >= 0 && !Double.IsNaN(fv) && FindInBitset(CategoricalThresholds, CategoricalBoundaries[cat_idx],
                         CategoricalBoundaries[cat_idx + 1] - CategoricalBoundaries[cat_idx], int_fval))
                        node = LteChild[node];
                    else
                        node = GtChild[node];
                }
                else
                {
                    fv = GetFeatureValue(fv, node);
                    if (fv <= RawThresholds[node])
                        node = LteChild[node];
                    else
                        node = GtChild[node];
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
                double val = 0;
                int ifeat = SplitFeatures[node];
                int ii = VBuffer<int>.FindIndexSorted(featIndices, 0, count, ifeat);
                if (ii < count && featIndices[ii] == ifeat)
                    val = featValues[ii];

                if (CategoricalSplit[node])
                {
                    Debug.Assert(CategoricalThresholds != null);
                    Debug.Assert(CategoricalBoundaries != null);
                    
                    int int_fval = (int)val;
                    int cat_idx = (int)RawThresholds[node];

                    if (int_fval >= 0 && !Double.IsNaN(val) && FindInBitset(CategoricalThresholds, CategoricalBoundaries[cat_idx],
                         CategoricalBoundaries[cat_idx + 1] - CategoricalBoundaries[cat_idx], int_fval))
                        node = LteChild[node];
                    else
                        node = GtChild[node];
                }
                else
                {
                    val = GetFeatureValue(val, node);
                    if (val <= RawThresholds[node])
                        node = LteChild[node];
                    else
                        node = GtChild[node];
                }
            }
            return ~node;
        }

        public FeatureToGainMap GainMap
        {
            get
            {
                var result = new FeatureToGainMap();
                int numNonLeaves = NumLeaves - 1;
                for (int n = 0; n < numNonLeaves; ++n)
                    result[SplitFeatures[n]] += SplitGains[n];
                return result;
            }
        }

        public FeatureToGainMap SplitMap
        {
            get
            {
                var result = new FeatureToGainMap();
                int numNonLeaves = NumLeaves - 1;
                for (int n = 0; n < numNonLeaves; ++n)
                    result[SplitFeatures[n]] += 1;
                return result;
            }
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
