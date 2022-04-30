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
        public RankingTrainer( LearningParameters lp
                             , ObjectiveParameters op
                             , IVectorisedPredictorWithFeatureWeights<double> nativePredictor
                             , Datasets datasets
                             ) : base(lp, op)
        {
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
                Booster = b.Booster.Clone();
                Booster.ResetTrainingData(datasets.Training);
            }
            else
                throw new Exception("nativePredictor is not a ranking predictor");
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
