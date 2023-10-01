// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using LightGBMNet.Tree;

namespace LightGBMNet.Train
{
    public class MulticlassNativePredictor : NativePredictorBase<double []>
    {
        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        public MulticlassNativePredictor(Booster booster) : base(booster)
        {
        }

        private protected override double [] ConvertOutput(double[] output)
        {
            return output;
        }

        public override double[][] GetOutputs(float[][] rows, int startIteration, int numIterations)
        {
            var output = Booster.PredictForMatsMulti(Booster.PredictType.Normal, rows, startIteration, (numIterations == -1) ? MaxNumTrees : numIterations);

            // convert from 2D array to array of rows
            var numRows = output.GetLength(0);
            var numCols = output.GetLength(1);
            var rslt = new double[numRows][];
            for (int i=0; i<rslt.Length; i++)
            {
                var row = new double[numCols];
                for (int j = 0; j < row.Length; j++)
                    row[j] = output[i, j];
                rslt[i] = row;
            }
            return rslt;
        }

    }

    public sealed class MulticlassTrainer : TrainerBase<double []>
    {
        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        public MulticlassTrainer(LearningParameters lp, ObjectiveParameters op) : base(lp, op)
        {
            if(!(op.Objective == ObjectiveType.MultiClass || op.Objective == ObjectiveType.MultiClassOva))
                throw new Exception("Require Objective == MultiClass or MultiClassOva");

            if (op.NumClass <= 1)
                throw new Exception("Require NumClass > 1");

            if (op.Metric == MetricType.DefaultMetric)
                op.Metric = MetricType.MultiLogLoss;     // TODO: why was this MultiError?????
        }

        public MulticlassTrainer( Parameters parameters
                                , IVectorisedPredictorWithFeatureWeights<double []> nativePredictor
                                , Datasets datasets
                                ) : base(parameters.Learning, parameters.Objective)
        {
            var op = parameters.Objective;
            if (!(op.Objective == ObjectiveType.MultiClass || op.Objective == ObjectiveType.MultiClassOva))
                throw new Exception("Require Objective == MultiClass or MultiClassOva");

            if (op.NumClass <= 1)
                throw new Exception("Require NumClass > 1");

            if (op.Metric == MetricType.DefaultMetric)
                op.Metric = MetricType.MultiLogLoss;     // TODO: why was this MultiError?????

            if (nativePredictor == null)
                throw new Exception("nativePredictor is null");
            if (datasets == null)
                throw new Exception("datasets is null");

            Datasets = datasets;
            if (nativePredictor is MulticlassNativePredictor b)
            {
                Booster = new Booster(parameters, datasets.Training, datasets.Validation);
                Booster.MergeWith(b.Booster);
            }
            else
                throw new Exception("nativePredictor is not a multiclass predictor");
        }

        private static Ensemble GetBinaryEnsemble(Ensemble trainedEnsemble, int numClass, int classID)
        {
          //var numClass = Objective.NumClass;
            Ensemble res = new Ensemble();
            for (int i = classID; i < trainedEnsemble.NumTrees; i += numClass)
            {
                res.AddTree(trainedEnsemble.GetTreeAt(i));
            }
            return res;
        }

        /// <summary>
        /// Load an externally trained model from a string
        /// </summary>
        /// <param name="modelString">Externally trained model string</param>
        public static Predictors<double[]> PredictorsFromString(string modelString)
        {
            var Booster = LightGBMNet.Train.Booster.FromString(modelString);
            IVectorisedPredictorWithFeatureWeights<double[]> native = new MulticlassNativePredictor(Booster);

            (var model, var args) = Booster.GetModel();
            var averageOutput = (args.Learning.Boosting == BoostingType.RandomForest);
            var managed = CreateManagedPredictor(model, Booster.NumFeatures, averageOutput, args.Objective);

            return new Predictors<double[]>(managed, native);
        }
        public static Predictors<double []> PredictorsFromFile(string fileName)
        {
            if (!System.IO.File.Exists(fileName))
                throw new Exception($"File does not exist: {fileName}");
            return PredictorsFromString(System.IO.File.ReadAllText(fileName));
        }

        private static BinaryPredictor CreateBinaryPredictor(Ensemble trainedEnsemble, int numClass, int featureCount, bool averageOutput, int classID)
        {
            return new BinaryPredictor(GetBinaryEnsemble(trainedEnsemble, numClass, classID), featureCount, averageOutput);
        }

        private static IPredictorWithFeatureWeights<double[]> CreateManagedPredictor(Ensemble trainedEnsemble, int featureCount, bool averageOutput, ObjectiveParameters objective)
        {
            var numClass = objective.NumClass;
            if (trainedEnsemble.NumTrees % numClass != 0)
                throw new Exception("Number of trees should be a multiple of number of classes.");

            var isSoftMax = (objective.Objective == ObjectiveType.MultiClass);
            IPredictorWithFeatureWeights<double>[] predictors = new IPredictorWithFeatureWeights<double>[numClass];
            var cali = isSoftMax ? null : new PlattCalibrator(-objective.Sigmoid);
            for (int i = 0; i < numClass; ++i)
            {
                var pred = CreateBinaryPredictor(trainedEnsemble, numClass, featureCount, averageOutput, i) as IPredictorWithFeatureWeights<double>;
                predictors[i] = isSoftMax ? pred : new CalibratedPredictor(pred, cali);
            }
            return OvaPredictor.Create(isSoftMax, predictors);
        }

        private protected override IPredictorWithFeatureWeights<double []> CreateManagedPredictor()
        {
            return CreateManagedPredictor(TrainedEnsemble, FeatureCount, AverageOutput, Objective);
        }

        private protected override IVectorisedPredictorWithFeatureWeights<double []> CreateNativePredictor()
        {
            return new MulticlassNativePredictor(Booster.Clone());
        }
    }

}
