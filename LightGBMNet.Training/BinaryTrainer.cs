// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using LightGBMNet.Interface;
using LightGBMNet.FastTree;

namespace LightGBMNet.Training
{
    public sealed class BinaryPredictor : FastTreePredictionWrapper
    {
        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        internal BinaryPredictor(Ensemble trainedEnsemble, int featureCount)
            : base(trainedEnsemble, featureCount)
        {
        }

        private BinaryPredictor(BinaryReader reader) : base(reader)
        {
        }

        protected override void SaveCore(BinaryWriter writer)
        {
            base.SaveCore(writer);
        }

        //public static IPredictorProducing<float> Create(BinaryReader reader)
        //{
        //    var predictor = new BinaryPredictor(reader);
        //    var calibrator = PlattCalibrator.Create(reader);
        //    return new FeatureWeightsCalibratedPredictor(predictor, calibrator);
        //}
    }

    public sealed class BinaryTrainer : TrainerBase<float, FeatureWeightsCalibratedPredictor>
    {
        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        public BinaryTrainer(Parameters args) : base(args)
        {
            if (args.Core.Objective != ObjectiveType.Binary)
                throw new Exception("Require Objective == ObjectiveType.Binary");
            if (args.Metric.Metric == MetricType.DefaultMetric)
                args.Metric.Metric = MetricType.BinaryLogLoss;
        }

        private protected override FeatureWeightsCalibratedPredictor CreatePredictor()
        {
            //Host.Check(TrainedEnsemble != null, "The predictor cannot be created before training is complete");
            //var innerArgs = LightGbmInterfaceUtils.JoinParameters(Options);
            var pred = new BinaryPredictor(TrainedEnsemble, FeatureCount);
            var cali = new PlattCalibrator(-Args.Objective.Sigmoid, 0);
            return new FeatureWeightsCalibratedPredictor(pred, cali);
        }

        //protected override void CheckDataValid(DataDense data)
        //{
        //    var labelType = data.Schema.Label.Type;
        //    if (!(labelType.IsBool || labelType.IsKey || labelType == NumberType.R4))
        //    {
        //        throw ch.ExceptParam(nameof(data),
        //            $"Label column '{data.Schema.Label.Name}' is of type '{labelType}', but must be key, boolean or R4.");
        //    }
        //}

        //protected override void CheckAndUpdateParametersBeforeTraining(IChannel ch, RoleMappedData data, float[] labels, int[] groups)
        //{
        //    Options["objective"] = "binary";
        //    // Add default metric.
        //    if (!Options.ContainsKey("metric"))
        //        Options["metric"] = "binary_logloss";
        //}
    }

}
