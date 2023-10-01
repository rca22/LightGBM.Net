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

        /// <summary>
        /// Load an externally trained model from a string
        /// </summary>
        /// <param name="modelString">Externally trained model string</param>
        public static Predictors<double> PredictorsFromString(string modelString)
        {
            var Booster = LightGBMNet.Train.Booster.FromString(modelString);
            IVectorisedPredictorWithFeatureWeights<double> native = new BinaryNativePredictor(Booster);
            
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
            var pred = new BinaryPredictor(trainedEnsemble, featureCount, averageOutput);
            var cali = new PlattCalibrator(-objective.Sigmoid);
            return new CalibratedPredictor(pred, cali);
        }

        private protected override IPredictorWithFeatureWeights<double> CreateManagedPredictor()
        {
            return CreateManagedPredictor(TrainedEnsemble, FeatureCount, AverageOutput, Objective);
        }

        private protected override IVectorisedPredictorWithFeatureWeights<double> CreateNativePredictor()
        {            
            return new BinaryNativePredictor(Booster.Clone());
        }
    }

}
