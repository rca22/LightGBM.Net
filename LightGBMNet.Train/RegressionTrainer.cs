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

        public RegressionTrainer(Parameters parameters
                                , IVectorisedPredictorWithFeatureWeights<double> nativePredictor
                                , Datasets datasets
                                ) : base(parameters.Learning, parameters.Objective)
        {
            var op = parameters.Objective;
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

            Datasets = datasets;
            if (nativePredictor is RegressionNativePredictor b)
            {
                Booster = new Booster(parameters, datasets.Training, datasets.Validation);
                Booster.MergeWith(b.Booster);
            }
            else
                throw new Exception("nativePredictor is not a regression predictor");
        }
        private static bool PositiveOutput(ObjectiveType objectiveType)
        {
            return (objectiveType == ObjectiveType.Poisson ||
                    objectiveType == ObjectiveType.Gamma ||
                    objectiveType == ObjectiveType.Tweedie);
        }

        private static bool SqrtOutput(ObjectiveParameters objective)
        {
            return objective.RegSqrt &&
                   objective.Objective != ObjectiveType.Huber &&
                   !RegressionTrainer.PositiveOutput(objective.Objective);
        }

        /// <summary>
        /// Load an externally trained model from a string
        /// </summary>
        /// <param name="modelString">Externally trained model string</param>
        public static Predictors<double> PredictorsFromString(string modelString)
        {
            var Booster = LightGBMNet.Train.Booster.FromString(modelString);
            IVectorisedPredictorWithFeatureWeights<double> native = new RegressionNativePredictor(Booster);

            (var model, var args) = Booster.GetModel();

            var averageOutput = (args.Learning.Boosting == BoostingType.RandomForest);
            IPredictorWithFeatureWeights<double> managed = CreateManagedPredictor(model, Booster.NumFeatures, averageOutput, args.Objective);

            return new Predictors<double>(managed, native);
        }
        public static Predictors<double> PredictorsFromFile(string fileName)
        {
            if (!System.IO.File.Exists(fileName))
                throw new Exception($"File does not exist: {fileName}");
            return PredictorsFromString(System.IO.File.ReadAllText(fileName));
        }

        private static IPredictorWithFeatureWeights<double> CreateManagedPredictor(Ensemble trainedEnsemble, int featureCount, bool averageOutput, ObjectiveParameters objective)
        {
            var pred = new RegressionPredictor(trainedEnsemble, featureCount, averageOutput);
            if (PositiveOutput(objective.Objective))
                return new CalibratedPredictor(pred, ExponentialCalibrator.Instance);
            else if (SqrtOutput(objective))
                return new CalibratedPredictor(pred, SqrtCalibrator.Instance);
            else
                return pred;
        }

        private protected override IPredictorWithFeatureWeights<double> CreateManagedPredictor()
        {
            return CreateManagedPredictor(TrainedEnsemble, FeatureCount, AverageOutput, Objective);
        }

        private protected override IVectorisedPredictorWithFeatureWeights<double> CreateNativePredictor()
        {
            return new RegressionNativePredictor(Booster.Clone());
        }
    }
}
