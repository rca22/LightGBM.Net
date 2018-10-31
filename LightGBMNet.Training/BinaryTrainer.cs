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

        internal BinaryPredictor(Ensemble trainedEnsemble, int featureCount, bool avgOutput)
            : base(trainedEnsemble, featureCount, avgOutput)
        {
        }

        private BinaryPredictor(BinaryReader reader) : base(reader)
        {
        }

        public void Save(BinaryWriter writer)
        {
            base.SaveCore(writer);
        }

        public static BinaryPredictor Create(BinaryReader reader)
        {
            return new BinaryPredictor(reader);
        }
    }

    public sealed class BinaryTrainer : TrainerBase<float>
    {
        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        public BinaryTrainer(LearningParameters lp, ObjectiveParameters op, MetricParameters mp) : base(lp, op, mp)
        {
            if (lp.Objective != ObjectiveType.Binary)
                throw new Exception("Require Objective == ObjectiveType.Binary");
            if (mp.Metric == MetricType.DefaultMetric)
                mp.Metric = MetricType.BinaryLogLoss;
        }

        private protected override IPredictorWithFeatureWeights<float> CreatePredictor()
        {
            var pred = new BinaryPredictor(TrainedEnsemble, FeatureCount, AverageOutput);
            var cali = new PlattCalibrator(-Objective.Sigmoid);
            return new CalibratedPredictor(pred, cali);
        }

        public static CalibratedPredictor Create(BinaryReader reader)
        {
            var pred = BinaryPredictor.Create(reader);
            var cali = PlattCalibrator.Create(reader);
            return new CalibratedPredictor(pred, cali);
        }

        public static void Save(IPredictorProducing<float> p, BinaryWriter writer)
        {
            var pred = p as CalibratedPredictor;
            if (pred == null) throw new Exception("Unexpected predictor type");
            (pred.SubPredictor as BinaryPredictor).Save(writer);
            (pred.Calibrator as PlattCalibrator).Save(writer);
        }

    }

}
