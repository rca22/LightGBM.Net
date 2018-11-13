// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using LightGBMNet.Train;
using LightGBMNet.Tree;

namespace LightGBMNet.Train
{
    public sealed class MulticlassTrainer : TrainerBase<double []>
    {
        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        public MulticlassTrainer(LearningParameters lp, ObjectiveParameters op) : base(lp, op)
        {
            if(!(op.Objective == ObjectiveType.MultiClass || op.Objective == ObjectiveType.MultiClassOva))
                throw new Exception("Require Objective == MultiClass or MultiClassOva");

            if (op.NumClass <= 1)
                throw new Exception("Require NumClass > 1");

            if (op.Metric == MetricType.DefaultMetric)
                op.Metric = MetricType.MultiLogLoss;     // TODO: why was this MultiError?????
        }

        private Ensemble GetBinaryEnsemble(int classID)
        {
            var numClass = Objective.NumClass;
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
            return new BinaryPredictor(GetBinaryEnsemble(classID), FeatureCount, AverageOutput);
        }

        private protected override IPredictorWithFeatureWeights<double []> CreatePredictor()
        {
            var numClass = Objective.NumClass;
            if (TrainedEnsemble.NumTrees % numClass != 0)
                throw new Exception("Number of trees should be a multiple of number of classes.");

            var isSoftMax = (Objective.Objective == ObjectiveType.MultiClass);
            IPredictorWithFeatureWeights<double>[] predictors = new IPredictorWithFeatureWeights<double>[numClass];
            var cali = isSoftMax ? null : new PlattCalibrator(-Objective.Sigmoid);
            for (int i = 0; i < numClass; ++i)
            {
                var pred = CreateBinaryPredictor(i) as IPredictorWithFeatureWeights<double>;
                predictors[i] = isSoftMax ? pred : new CalibratedPredictor(pred, cali);
            }
            return OvaPredictor.Create(isSoftMax, predictors);
        }

    }

}
