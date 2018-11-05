// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using LightGBMNet.Interface;
using LightGBMNet.FastTree;

namespace LightGBMNet.Training
{

    public sealed class RankingPredictor : FastTreePredictionWrapper
    {
        public override PredictionKind PredictionKind => PredictionKind.Ranking;

        internal RankingPredictor(Ensemble trainedEnsemble, int featureCount, bool avgOutput)
            : base(trainedEnsemble, featureCount, avgOutput)
        {
        }

        private RankingPredictor(BinaryReader reader) : base(reader)
        {
        }

        public void Save(BinaryWriter writer)
        {
            base.SaveCore(writer);
        }

        public static RankingPredictor Create(BinaryReader reader)
        {
            return new RankingPredictor(reader);
        }
    }

    public sealed class RankingTrainer : TrainerBase<double>
    {
        public override PredictionKind PredictionKind => PredictionKind.Ranking;

        public RankingTrainer(LearningParameters lp, ObjectiveParameters op, MetricParameters mp) : base(lp, op, mp)
        {
            if (lp.Objective != ObjectiveType.LambdaRank)
                throw new Exception("Require Objective == ObjectiveType.LambdaRank");
            if (mp.Metric == MetricType.DefaultMetric)
                mp.Metric = MetricType.Ndcg;
        }

        private protected override IPredictorWithFeatureWeights<double> CreatePredictor()
        {
            return new RankingPredictor(TrainedEnsemble, FeatureCount, AverageOutput);
        }

    }

}
