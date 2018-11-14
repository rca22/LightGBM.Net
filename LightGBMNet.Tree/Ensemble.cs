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

        private void SetGroups()
        {
            // _groups calculation already cached?
            if (_groupMaxThreads == MaxThreads && _groupNumTrees == NumTrees)
                return;

            if (_maxThreads == 1)
            {
                _groups = null;
                _groupVals = null;
            }
            else
            {
                var numGroups = Math.Min(NumTrees, MaxThreads);
                var per = Math.DivRem(NumTrees, numGroups, out var stub);
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
                Debug.Assert(idx0 == NumTrees);
            }
            _groupMaxThreads = MaxThreads;
            _groupNumTrees = NumTrees;
        }

        public Ensemble()
        {
            _trees = new List<RegressionTree>();
        }

        public Ensemble(BinaryReader reader)
        {
            int numTrees = reader.ReadInt32();
            if(!(numTrees >= 0)) throw new FormatException();
            _trees = new List<RegressionTree>(numTrees);
            for (int t = 0; t < numTrees; ++t)
                AddTree(RegressionTree.Load(reader));
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

        public double GetOutput(ref VBuffer<float> feat)
        {
            SetGroups();

            double result = 0.0;

            if (_maxThreads > 1)
            {
                var featcopy = feat;
                Parallel.For(0, _groups.Length, i =>
                {
                    double output = 0.0;
                    (int lo, int hi) = _groups[i];
                    for (int h = lo; h < hi; h++)
                        output += _trees[h].GetOutput(ref featcopy);
                    _groupVals[i] = output;
                }
                );
                result = _groupVals.Sum();
            }
            else
            {
                for (int h = 0; h < NumTrees; h++)
                    result += _trees[h].GetOutput(ref feat);
            }

            return result;
        }

    }

    public class FeatureToGainMap : Dictionary<int, double>
    {
        public FeatureToGainMap() { }
        // Override default Dictionary to return 0.0 for non-eisting keys
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
