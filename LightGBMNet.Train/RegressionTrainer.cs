// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using LightGBMNet.Tree;

namespace LightGBMNet.Train
{
    public class RegressionNativePredictor : NativePredictorBase<double>
    {
        public override PredictionKind PredictionKind => PredictionKind.Regression;

        public RegressionNativePredictor(Booster booster) : base(booster)
        {
        }

        private protected override double ConvertOutput(double[] output)
        {
            return output[0];
        }

        public override double[] GetOutputs(float[][] rows, int startIteration, int numIterations)
        {
            return Booster.PredictForMats(Booster.PredictType.Normal, rows, startIteration, (numIterations == -1) ? MaxNumTrees : numIterations, MaxThreads);
        }
    }

    public sealed class RegressionTrainer : TrainerBase<double>
    {
        public override PredictionKind PredictionKind => PredictionKind.Regression;

        public RegressionTrainer(LearningParameters lp, ObjectiveParameters op) : base(lp, op)
        {
            if (!(op.Objective == ObjectiveType.Regression ||
                  op.Objective == ObjectiveType.RegressionL1 ||
                  op.Objective == ObjectiveType.Huber ||
                  op.Objective == ObjectiveType.Fair ||
                  op.Objective == ObjectiveType.Poisson ||
                  op.Objective == ObjectiveType.Quantile ||
                  op.Objective == ObjectiveType.Mape ||
                  op.Objective == ObjectiveType.Gamma ||
                  op.Objective == ObjectiveType.Tweedie
                  ))
                throw new Exception("Require regression ObjectiveType");

            if (op.Metric == MetricType.DefaultMetric)
                op.Metric = MetricType.Mse;
        }

        public RegressionTrainer( LearningParameters lp
                                , ObjectiveParameters op
                                , IVectorisedPredictorWithFeatureWeights<double> nativePredictor
                                , Datasets datasets
                                ) : base(lp, op)
        {
            if (!(op.Objective == ObjectiveType.Regression ||
                  op.Objective == ObjectiveType.RegressionL1 ||
                  op.Objective == ObjectiveType.Huber ||
                  op.Objective == ObjectiveType.Fair ||
                  op.Objective == ObjectiveType.Poisson ||
                  op.Objective == ObjectiveType.Quantile ||
                  op.Objective == ObjectiveType.Mape ||
                  op.Objective == ObjectiveType.Gamma ||
                  op.Objective == ObjectiveType.Tweedie
                  ))
                throw new Exception("Require regression ObjectiveType");

            if (op.Metric == MetricType.DefaultMetric)
                op.Metric = MetricType.Mse;

            if (nativePredictor == null)
                throw new Exception("nativePredictor is null");
            if (datasets == null)
                throw new Exception("datasets is null");
            // this is because there is no equivalent of Booster.ResetTrainingData for validation data
            if (datasets.Validation != null)
                throw new Exception("Not supported: new validation dataset for existing booster. Please set dataset.Validation to null.");
            Datasets = datasets;
            if (nativePredictor is RegressionNativePredictor b)
            {
                Booster = b.Booster.Clone();
                Booster.ResetTrainingData(datasets.Training);
            }
            else
                throw new Exception("nativePredictor is not a regression predictor");
        }
        private bool PositiveOutput()
        {
            return (Objective.Objective == ObjectiveType.Poisson ||
                    Objective.Objective == ObjectiveType.Gamma ||
                    Objective.Objective == ObjectiveType.Tweedie);
        }

        private bool SqrtOutput()
        {
            return Objective.RegSqrt &&
                   Objective.Objective != ObjectiveType.Huber &&
                   !PositiveOutput();
        }

        private protected override IPredictorWithFeatureWeights<double> CreateManagedPredictor()
        {
            var pred = new RegressionPredictor(TrainedEnsemble, FeatureCount, AverageOutput);
            if (PositiveOutput())
                return new CalibratedPredictor(pred, ExponentialCalibrator.Instance);
            else if (SqrtOutput())
                return new CalibratedPredictor(pred, SqrtCalibrator.Instance);
            else
                return pred;
        }

        private protected override IVectorisedPredictorWithFeatureWeights<double> CreateNativePredictor()
        {
            return new RegressionNativePredictor(Booster.Clone());
        }
    }
}
