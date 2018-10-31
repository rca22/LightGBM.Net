// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace LightGBMNet.FastTree
{
    public class Ensemble
    {
        private readonly List<RegressionTree> _trees;

        public IEnumerable<RegressionTree> Trees => _trees;

        public int NumTrees => _trees.Count;

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
        }

        public void Save(BinaryWriter writer)
        {
            writer.Write(NumTrees);
            foreach (RegressionTree tree in Trees)
                tree.Save(writer);
        }

        public void AddTree(RegressionTree tree) => _trees.Add(tree);
        public void AddTreeAt(RegressionTree tree, int index) => _trees.Insert(index, tree);
        public void RemoveTree(int index) => _trees.RemoveAt(index);
        public void RemoveAfter(int index) => _trees.RemoveRange(index, NumTrees - index);
        public RegressionTree GetTreeAt(int index) => _trees[index];

        public double GetOutput(ref VBuffer<float> feat)
        {
            double output = 0.0;
            for (int h = 0; h < NumTrees; h++)
                output += _trees[h].GetOutput(ref feat);
            return output;
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
