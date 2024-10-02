// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;

namespace LightGBMNet.Tree
{
    using TScalarPredictor = IPredictorWithFeatureWeights<double>;

    public class OvaPredictor : IPredictorWithFeatureWeights<double []>
    {

        public TScalarPredictor[] Predictors { get; }
        public bool IsSoftMax { get; }
        public int NumInputs => Predictors[0].NumInputs;

        public PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        public static OvaPredictor Create(bool isSoftMax, TScalarPredictor[] predictors)
        {
            return new OvaPredictor(isSoftMax, predictors);
        }

        protected OvaPredictor(bool isSoftMax, TScalarPredictor[] predictors)
        {
            if ((predictors?.Length ?? 0) == 0)
                throw new ArgumentException("Predictors must be non-empty");
            IsSoftMax = isSoftMax;
            Predictors = predictors;
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

        private static BinaryPredictor CreateBinaryPredictor(Ensemble trainedEnsemble, int numClass, int featureCount, bool averageOutput, int classID)
        {
            return new BinaryPredictor(GetBinaryEnsemble(trainedEnsemble, numClass, classID), featureCount, averageOutput);
        }

        public static IPredictorWithFeatureWeights<double[]> CreateManagedPredictor(Ensemble trainedEnsemble, int featureCount, bool averageOutput, ObjectiveParameters objective)
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

        public static IPredictorWithFeatureWeights<double[]> FromString(string modelString)
        {
            (var model, var args, int numFeatures) = Ensemble.GetModelFromString(modelString);
            var averageOutput = (args.Learning.Boosting == BoostingType.RandomForest);
            var managed = OvaPredictor.CreateManagedPredictor(model, numFeatures, averageOutput, args.Objective);
            return managed;
        }
        public static IPredictorWithFeatureWeights<double[]> FromFile(string fileName)
        {
            if (!System.IO.File.Exists(fileName))
                throw new Exception($"File does not exist: {fileName}");
            return FromString(System.IO.File.ReadAllText(fileName));
        }

        public int MaxNumTrees
        {
            get => Predictors.Select(x => x.MaxNumTrees).Max();
            set { foreach (var p in Predictors) p.MaxNumTrees = value; }
        }

        public int MaxThreads
        {
            get => Predictors[0].MaxThreads;
            set { foreach (var p in Predictors) p.MaxThreads = value; }
        }

        public FeatureToGainMap GetFeatureWeights(bool normalise = false, bool splits = false)
        {
            var gainMap = new FeatureToGainMap();
            var numTrees = 0;
            foreach (var p in Predictors)
            {
                numTrees += p.MaxNumTrees;
                foreach (var kv in p.GetFeatureWeights(false, splits))
                    gainMap[kv.Key] += kv.Value;
            }
            if (normalise)
            {
                foreach (var k in gainMap.Keys.ToList())
                    gainMap[k] = gainMap[k] / numTrees;
            }
            return gainMap;            
        }

        public IEnumerable<double> GetFeatureGains(int feature)
        {
            return Predictors.SelectMany(tree => tree.GetFeatureGains(feature));
        }

        public void GetOutput(ref VBuffer<float> src, ref double [] dst, int startIteration, int numIterations)
        {
            if (startIteration + numIterations > MaxNumTrees)
                throw new ArgumentException("startIteration + numIterations must be <= MaxNumTrees");

            if ((dst?.Length ?? 0) < Predictors.Length)
                dst = new double[Predictors.Length];

            for (var i = 0; i < Predictors.Length; i++)
                Predictors[i].GetOutput(ref src, ref dst[i], startIteration, numIterations);

            if (IsSoftMax)
                Softmax(dst, Predictors.Length);
            else
                Normalize(dst, Predictors.Length);
        }

        public static void Normalize(double[] output, int count)
        {
            // Clamp to zero and normalize.
            double sum = 0;
            for (int i = 0; i < count; i++)
            {
                var value = output[i];
                if (value >= 0)
                    sum += value;
                else
                    output[i] = 0;
            }

            if (sum > 0)
            {
                for (int i = 0; i < count; i++)
                    output[i] = (output[i] / sum);
            }
        }

        public static void Softmax(double[] output, int count)
        {
            double wmax = output[0];
            for (int i = 1; i < count; ++i)
            {
                wmax = Math.Max(output[i], wmax);
            }
            double wsum = 0.0f;
            for (int i = 0; i < count; ++i)
            {
                output[i] = Math.Exp(output[i] - wmax);
                wsum += output[i];
            }
            for (int i = 0; i < count; ++i)
            {
                output[i] = (output[i] / wsum);
            }
        }

    }
}