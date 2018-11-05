// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Threading.Tasks;

namespace LightGBMNet.Tree
{
    using TScalarPredictor = IPredictorWithFeatureWeights<double>;

    public sealed class OvaPredictor : IPredictorWithFeatureWeights<VBuffer<double>>
    {

        public TScalarPredictor[] Predictors { get; }
        public bool IsSoftMax { get; }
        public int NumInputs => Predictors[0].NumInputs;

        public PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        public static OvaPredictor Create(bool isSoftMax, TScalarPredictor[] predictors)
        {
            return new OvaPredictor(isSoftMax, predictors);
        }

        private OvaPredictor(bool isSoftMax, TScalarPredictor[] predictors)
        {
            if ((predictors?.Length ?? 0) == 0)
                throw new ArgumentException("Predictors must be non-empty");
            IsSoftMax = isSoftMax;
            Predictors = predictors;
        }
        
        public int MaxNumTrees => Predictors.Select(x => x.MaxNumTrees).Max();
        public void SetMaxNumTrees(int numTrees)
        {
            foreach (var p in Predictors)
                p.SetMaxNumTrees(numTrees);
        }

        public FeatureToGainMap GetFeatureWeights(bool normalise = false, bool splits = false)
        {
            if (normalise)
                throw new ArgumentException("Cannot normalise across multiple ensembles");

            var gainMap = new FeatureToGainMap();
            foreach (var p in Predictors)
            {
                foreach (var kv in p.GetFeatureWeights(normalise, splits))
                    gainMap[kv.Key] += kv.Value;
            }
            return gainMap;            
        }

        public void GetOutput(ref VBuffer<float> src, ref VBuffer<double> dst)
        {
            var values = dst.Values;
            if ((values?.Length ?? 0) < Predictors.Length)
                values = new double[Predictors.Length];

#if DEBUG
            for(var i=0; i < Predictors.Length; i++)
                Predictors[i].GetOutput(ref src, ref values[i]);
#else
            var tmp = src;
            Parallel.For(0, Predictors.Length, i => Predictors[i].GetOutput(ref tmp, ref values[i]));
#endif

            if (IsSoftMax)
                Softmax(values, Predictors.Length);
            else
                Normalize(values, Predictors.Length);

            dst = new VBuffer<double>(Predictors.Length, values, dst.Indices);
        }

        private static void Normalize(double[] output, int count)
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

        private static void Softmax(double[] output, int count)
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