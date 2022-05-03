// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using LightGBMNet.Tree;

namespace LightGBMNet.Train
{
    public class BinaryNativePredictor : NativePredictorBase<double>
    {
        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        public BinaryNativePredictor(Booster booster) : base(booster)
        {
        }

        private protected override double ConvertOutput(double [] output)
        {
            return output[0];
        }

        public override double[] GetOutputs(float[][] rows, int startIteration, int numIterations)
        {
            return Booster.PredictForMats(Booster.PredictType.Normal, rows, startIteration, (numIterations == -1) ? MaxNumTrees : numIterations, MaxThreads);
        }
    }

    public sealed class BinaryTrainer : TrainerBase<double>
    {
        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        public BinaryTrainer(LearningParameters lp, ObjectiveParameters op) : base(lp, op)
        {
            if (op.Objective != ObjectiveType.Binary)
                throw new Exception("Require Objective == ObjectiveType.Binary");
            if (op.Metric == MetricType.DefaultMetric)
                op.Metric = MetricType.BinaryLogLoss;
        }

        /// <summary>
        /// </summary>
        /// <param name="lp"></param>
        /// <param name="op"></param>
        /// <param name="nativePredictor">Note: underlying booster is cloned</param>
        /// <param name="datasets"></param>
        public BinaryTrainer( Parameters parameters
                            , IVectorisedPredictorWithFeatureWeights<double> nativePredictor
                            , Datasets datasets
                            ) : base(parameters.Learning, parameters.Objective)
        {
            var op = parameters.Objective;
            if (op.Objective != ObjectiveType.Binary)
                throw new Exception("Require Objective == ObjectiveType.Binary");
            if (op.Metric == MetricType.DefaultMetric)
                op.Metric = MetricType.BinaryLogLoss;

            if (nativePredictor == null)
                throw new Exception("nativePredictor is null");
            if (datasets == null)
                throw new Exception("datasets is null");

            Datasets = datasets;
            if (nativePredictor is BinaryNativePredictor b)
            {
                Booster = new Booster(parameters, datasets.Training, datasets.Validation);
                Booster.MergeWith(b.Booster);
            }
            else 
                throw new Exception("nativePredictor is not a binary predictor");
        }

        private protected override IPredictorWithFeatureWeights<double> CreateManagedPredictor()
        {
            var pred = new BinaryPredictor(TrainedEnsemble, FeatureCount, AverageOutput);
            var cali = new PlattCalibrator(-Objective.Sigmoid);
            return new CalibratedPredictor(pred, cali);
        }

        private protected override IVectorisedPredictorWithFeatureWeights<double> CreateNativePredictor()
        {            
            return new BinaryNativePredictor(Booster.Clone());
        }
    }

}
