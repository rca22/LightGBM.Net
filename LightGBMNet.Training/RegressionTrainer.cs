// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using LightGBMNet.Interface;
using LightGBMNet.FastTree;

namespace LightGBMNet.Training
{
    public sealed class RegressionPredictor : FastTreePredictionWrapper
    {
        public override PredictionKind PredictionKind => PredictionKind.Regression;

        internal RegressionPredictor(Ensemble trainedEnsemble, int featureCount)
            : base(trainedEnsemble, featureCount)
        {
        }

        private RegressionPredictor(BinaryReader reader) : base(reader)
        {
        }

        protected override void SaveCore(BinaryWriter writer)
        {
            base.SaveCore(writer);
        }

        //public static RegressionPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        //{
        //    Contracts.CheckValue(env, nameof(env));
        //    env.CheckValue(ctx, nameof(ctx));
        //    ctx.CheckAtModel(GetVersionInfo());
        //    return new RegressionPredictor(env, ctx);
        //}
    }

    public sealed class RegressionTrainer : TrainerBase<float, RegressionPredictor>
    {
        public override PredictionKind PredictionKind => PredictionKind.Regression;

        public RegressionTrainer(Parameters args) : base(args)
        {
            if (!(args.Core.Objective == ObjectiveType.Regression ||
                  args.Core.Objective == ObjectiveType.RegressionL1 ||
                  args.Core.Objective == ObjectiveType.Huber ||
                  args.Core.Objective == ObjectiveType.Fair ||
                  args.Core.Objective == ObjectiveType.Poisson ||
                  args.Core.Objective == ObjectiveType.Quantile ||
                  args.Core.Objective == ObjectiveType.Mape ||
                  args.Core.Objective == ObjectiveType.Gamma ||
                  args.Core.Objective == ObjectiveType.Tweedie
                  ))
                throw new Exception("Require regression ObjectiveType");

            if (args.Metric.Metric == MetricType.DefaultMetric)
                args.Metric.Metric = MetricType.Mse;
        }

        private protected override RegressionPredictor CreatePredictor()
        {
            //Host.Check(TrainedEnsemble != null,
            //    "The predictor cannot be created before training is complete");
            return new RegressionPredictor(TrainedEnsemble, FeatureCount);
        }

    }

}
