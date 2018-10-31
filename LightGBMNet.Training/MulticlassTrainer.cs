// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using LightGBMNet.Interface;
using LightGBMNet.FastTree;

namespace LightGBMNet.Training
{
    public sealed class MulticlassTrainer : TrainerBase<VBuffer<float>>
    {
        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        public MulticlassTrainer(LearningParameters lp, ObjectiveParameters op, MetricParameters mp) : base(lp, op, mp)
        {
            if(!(lp.Objective == ObjectiveType.MultiClass || lp.Objective == ObjectiveType.MultiClassOva))
                throw new Exception("Require Objective == MultiClass or MultiClassOva");

            if (op.NumClass <= 1)
                throw new Exception("Require NumClass > 1");

            if (mp.Metric == MetricType.DefaultMetric)
                mp.Metric = MetricType.MultiLogLoss;     // TODO: why was this MultiError?????
        }

        private Ensemble GetBinaryEnsemble(int classID)
        {
            var numClass = Objective.NumClass;
            Ensemble res = new Ensemble();
            for (int i = classID; i < TrainedEnsemble.NumTrees; i += numClass)
            {
                // Ignore dummy trees.
                if (TrainedEnsemble.GetTreeAt(i).NumLeaves > 1)
                    res.AddTree(TrainedEnsemble.GetTreeAt(i));
            }
            return res;
        }

        private BinaryPredictor CreateBinaryPredictor(int classID)
        {
            return new BinaryPredictor(GetBinaryEnsemble(classID), FeatureCount, AverageOutput);
        }

        private protected override IPredictorWithFeatureWeights<VBuffer<float>> CreatePredictor()
        {
            var numClass = Objective.NumClass;
            if (TrainedEnsemble.NumTrees % numClass != 0)
                throw new Exception("Number of trees should be a multiple of number of classes.");

            var isSoftMax = (Learning.Objective == ObjectiveType.MultiClass);
            IPredictorWithFeatureWeights<float>[] predictors = new IPredictorWithFeatureWeights<float>[numClass];
            var cali = isSoftMax ? null : new PlattCalibrator(-Objective.Sigmoid);
            for (int i = 0; i < numClass; ++i)
            {
                var pred = CreateBinaryPredictor(i) as IPredictorWithFeatureWeights<float>;
                predictors[i] = isSoftMax ? pred : new CalibratedPredictor(pred, cali);
            }
            return OvaPredictor.Create(isSoftMax, predictors);
        }

        public static void Save(IPredictorWithFeatureWeights<VBuffer<float>> input, BinaryWriter writer)
        {
            var pred = input as OvaPredictor;
            if (pred == null) throw new Exception("Unexpected predictor type");
            writer.Write(pred.IsSoftMax);
            writer.Write(pred.Predictors.Length);
            foreach(var p in pred.Predictors) {
                if (pred.IsSoftMax)
                    (p as BinaryPredictor).Save(writer);
                else
                {
                    var q = p as CalibratedPredictor;
                    (q.SubPredictor as BinaryPredictor).Save(writer);
                    (q.Calibrator as PlattCalibrator).Save(writer);
                }
            }
        }

        public static OvaPredictor Create(BinaryReader reader)
        {
            var isSoftMax = reader.ReadBoolean();
            var len = reader.ReadInt32();
            var predictors = new IPredictorWithFeatureWeights<float>[len];
            for (var i = 0; i < len; i++)
            {
                var pred = BinaryPredictor.Create(reader) as IPredictorWithFeatureWeights<float>;
                predictors[i] = isSoftMax ? pred : new CalibratedPredictor(pred, PlattCalibrator.Create(reader));
            }
            return OvaPredictor.Create(isSoftMax, predictors);
        }

    }

}
