// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using LightGBMNet.Interface;
using LightGBMNet.FastTree;

namespace LightGBMNet.Training
{
    public sealed class MulticlassTrainer : TrainerBase<VBuffer<float>, OvaPredictor>
    {
      //private const int _minDataToUseSoftmax = 50000;
      //private const double _maxNumClass = 1e6;
      //private int _numClass;
      //private int _tlcNumClass;

        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        public MulticlassTrainer(Parameters args) : base(args)
        {
            if(!(args.Core.Objective == ObjectiveType.MultiClass || args.Core.Objective == ObjectiveType.MultiClassOva))
                throw new Exception("Require Objective == MultiClass or MultiClassOva");

            if (args.Objective.NumClass <= 1)
                throw new Exception("Require NumClass > 1");

            if (args.Metric.Metric == MetricType.DefaultMetric)
                args.Metric.Metric = MetricType.MultiLogLoss;     // TODO: why was this MultiError?????
        }

        private Ensemble GetBinaryEnsemble(int classID)
        {
            var numClass = Args.Objective.NumClass;
            Ensemble res = new Ensemble();
            for (int i = classID; i < TrainedEnsemble.NumTrees; i += numClass)
            {
                // Ignore dummy trees.
                if (TrainedEnsemble.GetTreeAt(i).NumLeaves > 1)
                    res.AddTree(TrainedEnsemble.GetTreeAt(i));
            }
            return res;
        }

        private BinaryPredictor CreateBinaryPredictor(int classID)
        {
            return new BinaryPredictor(GetBinaryEnsemble(classID), FeatureCount);
        }

        private protected override OvaPredictor CreatePredictor()
        {
            //Host.Check(TrainedEnsemble != null, "The predictor cannot be created before training is complete.");

            //Host.Assert(_numClass > 1, "Must know the number of classes before creating a predictor.");
            //Host.Assert(TrainedEnsemble.NumTrees % _numClass == 0, "Number of trees should be a multiple of number of classes.");

            var numClass = Args.Objective.NumClass;
            var isSoftMax = (Args.Core.Objective == ObjectiveType.MultiClass);
            IPredictorProducing<float>[] predictors = new IPredictorProducing<float>[numClass];
            for (int i = 0; i < numClass; ++i)
            {
                var pred = CreateBinaryPredictor(i);
                if (isSoftMax)
                {
                    predictors[i] = pred;
                }
                else
                {
                    var cali = new PlattCalibrator(-Args.Objective.Sigmoid, 0);
                    predictors[i] = new FeatureWeightsCalibratedPredictor(pred, cali);
                }
            }
            return OvaPredictor.Create(isSoftMax, predictors);
        }

        //protected override void ConvertNaNLabels(IChannel ch, RoleMappedData data, float[] labels)
        //{
        //    // Only initialize one time.
        //    if (_numClass < 0)
        //    {
        //        float minLabel = float.MaxValue;
        //        float maxLabel = float.MinValue;
        //        bool hasNaNLabel = false;
        //        foreach (var label in labels)
        //        {
        //            if (float.IsNaN(label))
        //                hasNaNLabel = true;
        //            else
        //            {
        //                minLabel = Math.Min(minLabel, label);
        //                maxLabel = Math.Max(maxLabel, label);
        //            }
        //        }
        //        ch.CheckParam(minLabel >= 0, nameof(data), "min label cannot be negative");
        //        if (maxLabel >= _maxNumClass)
        //            throw ch.ExceptParam(nameof(data), $"max label cannot exceed {_maxNumClass}");
        //
        //        if (data.Schema.Label.Type.IsKey)
        //        {
        //            ch.Check(data.Schema.Label.Type.AsKey.Contiguous, "label value should be contiguous");
        //            if (hasNaNLabel)
        //                _numClass = data.Schema.Label.Type.AsKey.Count + 1;
        //            else
        //                _numClass = data.Schema.Label.Type.AsKey.Count;
        //            _tlcNumClass = data.Schema.Label.Type.AsKey.Count;
        //        }
        //        else
        //        {
        //            if (hasNaNLabel)
        //                _numClass = (int)maxLabel + 2;
        //            else
        //                _numClass = (int)maxLabel + 1;
        //            _tlcNumClass = (int)maxLabel + 1;
        //        }
        //    }
        //    float defaultLabel = _numClass - 1;
        //    for (int i = 0; i < labels.Length; ++i)
        //        if (float.IsNaN(labels[i]))
        //            labels[i] = defaultLabel;
        //}

        //protected override void GetDefaultParameters(int numRow, bool hasCategorical, int totalCats, bool hiddenMsg=false)
        //{
        //    base.GetDefaultParameters(ch, numRow, hasCategorical, totalCats, true);
        //    int numLeaves = (int)Options["num_leaves"];
        //    int minDataPerLeaf = Args.MinDataPerLeaf ?? DefaultMinDataPerLeaf(numRow, numLeaves, _numClass);
        //    Options["min_data_per_leaf"] = minDataPerLeaf;
        //    if (!hiddenMsg)
        //    {
        //        if (!Args.LearningRate.HasValue)
        //            ch.Info("Auto-tuning parameters: " + nameof(Args.LearningRate) + " = " + Options["learning_rate"]);
        //        if (!Args.NumLeaves.HasValue)
        //            ch.Info("Auto-tuning parameters: " + nameof(Args.NumLeaves) + " = " + numLeaves);
        //        if (!Args.MinDataPerLeaf.HasValue)
        //            ch.Info("Auto-tuning parameters: " + nameof(Args.MinDataPerLeaf) + " = " + minDataPerLeaf);
        //    }
        //}

    }

}
