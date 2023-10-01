// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using LightGBMNet.Tree;

namespace LightGBMNet.Train
{
    public class RankingNativePredictor : NativePredictorBase<double>
    {
        public override PredictionKind PredictionKind => PredictionKind.Ranking;

        public RankingNativePredictor(Booster booster) : base(booster)
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

    public sealed class RankingTrainer : TrainerBase<double>
    {
        public override PredictionKind PredictionKind => PredictionKind.Ranking;

        public RankingTrainer(LearningParameters lp, ObjectiveParameters op) : base(lp, op)
        {
            if (op.Objective != ObjectiveType.LambdaRank)
                throw new Exception("Require Objective == ObjectiveType.LambdaRank");
            if (op.Metric == MetricType.DefaultMetric)
                op.Metric = MetricType.Ndcg;
        }
        public RankingTrainer( Parameters parameters
                             , IVectorisedPredictorWithFeatureWeights<double> nativePredictor
                             , Datasets datasets
                             ) : base(parameters.Learning, parameters.Objective)
        {
            var op = parameters.Objective;
            if (op.Objective != ObjectiveType.LambdaRank)
                throw new Exception("Require Objective == ObjectiveType.LambdaRank");
            if (op.Metric == MetricType.DefaultMetric)
                op.Metric = MetricType.Ndcg;
            
            if (nativePredictor == null)
                throw new Exception("nativePredictor is null");
            if (datasets == null)
                throw new Exception("datasets is null");
            // this is because there is no equivalent of Booster.ResetTrainingData for validation data
            if (datasets.Validation != null)
                throw new Exception("Not supported: new validation dataset for existing booster. Please set dataset.Validation to null.");
            
            Datasets = datasets;
            if (nativePredictor is RankingNativePredictor b)
            {
                Booster = new Booster(parameters, datasets.Training, datasets.Validation);
                Booster.MergeWith(b.Booster);
            }
            else
                throw new Exception("nativePredictor is not a ranking predictor");
        }

        /// <summary>
        /// Load an externally trained model from a string
        /// </summary>
        /// <param name="modelString">Externally trained model string</param>
        public static Predictors<double> PredictorsFromString(string modelString)
        {
            var Booster = LightGBMNet.Train.Booster.FromString(modelString);
            IVectorisedPredictorWithFeatureWeights<double> native = new RankingNativePredictor(Booster);

            (var model, var args) = Booster.GetModel();
            var averageOutput = (args.Learning.Boosting == BoostingType.RandomForest);
            var managed = new RankingPredictor(model, Booster.NumFeatures, averageOutput);

            return new Predictors<double>(managed, native);
        }
        public static Predictors<double> PredictorsFromFile(string fileName)
        {
            if (!System.IO.File.Exists(fileName))
                throw new Exception($"File does not exist: {fileName}");
            return PredictorsFromString(System.IO.File.ReadAllText(fileName));
        }

        private protected override IPredictorWithFeatureWeights<double> CreateManagedPredictor()
        {
            return new RankingPredictor(TrainedEnsemble, FeatureCount, AverageOutput);
        }

        private protected override IVectorisedPredictorWithFeatureWeights<double> CreateNativePredictor()
        {
            return new RankingNativePredictor(Booster.Clone());
        }
    }

}
