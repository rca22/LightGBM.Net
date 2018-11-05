// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using LightGBMNet.Train;
using LightGBMNet.Tree;

namespace LightGBMNet.Train
{
    public sealed class RegressionTrainer : TrainerBase<double>
    {
        public override PredictionKind PredictionKind => PredictionKind.Regression;

        public RegressionTrainer(LearningParameters lp, ObjectiveParameters op, MetricParameters mp) : base(lp, op, mp)
        {
            if (!(lp.Objective == ObjectiveType.Regression ||
                  lp.Objective == ObjectiveType.RegressionL1 ||
                  lp.Objective == ObjectiveType.Huber ||
                  lp.Objective == ObjectiveType.Fair ||
                  lp.Objective == ObjectiveType.Poisson ||
                  lp.Objective == ObjectiveType.Quantile ||
                  lp.Objective == ObjectiveType.Mape ||
                  lp.Objective == ObjectiveType.Gamma ||
                  lp.Objective == ObjectiveType.Tweedie
                  ))
                throw new Exception("Require regression ObjectiveType");

            if (mp.Metric == MetricType.DefaultMetric)
                mp.Metric = MetricType.Mse;
        }

        private bool PositiveOutput()
        {
            return Learning.Objective == ObjectiveType.Poisson ||
                   Learning.Objective == ObjectiveType.Gamma ||
                   Learning.Objective == ObjectiveType.Tweedie;
        }

        private bool SqrtOutput()
        {
            return Objective.RegSqrt &&
                   Learning.Objective != ObjectiveType.Huber &&
                   !PositiveOutput();
        }

        private protected override IPredictorWithFeatureWeights<double> CreatePredictor()
        {
            var pred = new RegressionPredictor(TrainedEnsemble, FeatureCount, AverageOutput);
            if (PositiveOutput())
                return new CalibratedPredictor(pred, ExponentialCalibrator.Instance);
            else if (SqrtOutput())
                return new CalibratedPredictor(pred, SqrtCalibrator.Instance);
            else
                return pred;
        }
    }

}
