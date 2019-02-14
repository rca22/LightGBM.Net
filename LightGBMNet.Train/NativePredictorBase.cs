using System;
using System.IO;
using System.Collections.Generic;
using LightGBMNet.Tree;

namespace LightGBMNet.Train
{
    public abstract class NativePredictorBase<TOutput> : IVectorisedPredictorWithFeatureWeights<TOutput>
    {
        public Booster Booster { get; }
        public int MaxNumTrees { get; set; }

        public abstract PredictionKind PredictionKind { get; }

        private protected abstract TOutput ConvertOutput(double [] output);

        public NativePredictorBase(Booster booster)
        {
            Booster = booster;
            MaxNumTrees = booster.BestIteration < 0 ? booster.CurrentIteration : booster.BestIteration;
        }
        public int NumInputs => Booster.NumFeatures;

        // TODO: not used
        public int MaxThreads { get; set; }

        public void GetOutput(ref VBuffer<float> features, ref TOutput prob)
        {
            var output = Booster.PredictForMat(Booster.PredictType.Normal, features.Values, MaxNumTrees);
            prob = ConvertOutput(output);
        }

        public abstract TOutput[] GetOutputs(float[][] rows);

        public FeatureToGainMap GetFeatureWeights(bool normalise = false, bool splits = false)
        {
            var rslt = new FeatureToGainMap();
            var totalTrees = MaxNumTrees * Booster.NumModelPerIteration;
            var featimps = Booster.GetFeatureImportance(MaxNumTrees, splits ? Booster.ImportanceType.Split : Booster.ImportanceType.Gain);
            for (int i = 0; i < featimps.Length; i++)
            {
                if (featimps[i] != 0)
                    rslt[i] = normalise ? featimps[i]/totalTrees : featimps[i];
            }
            return rslt;
        }

        public IEnumerable<double> GetFeatureGains(int feature)
        {
            throw new NotImplementedException();   // TODO: 
        }

        #region IDisposable
        public void Dispose()
        {
            Booster.Dispose();
        }
        #endregion
    }

    public static class NativePredictorPersist
    {
        public static void Save<T>(IPredictorWithFeatureWeights<T> pred, BinaryWriter writer)
        {
            Booster booster = null;
            int maxNumTrees = pred.MaxNumTrees;
            if (pred is BinaryNativePredictor b)
            {
                writer.Write(0);
                booster = b.Booster;
            }
            else if (pred is RegressionNativePredictor r)
            {
                writer.Write(1);
                booster = r.Booster;
            }
            else if (pred is MulticlassNativePredictor m)
            {
                writer.Write(2);
                booster = m.Booster;
            }
            else if (pred is RankingNativePredictor k)
            {
                writer.Write(3);
                booster = k.Booster;
            }
            else
                throw new Exception("Unknown IPredictorWithFeatureWeights type");
            writer.Write(booster.GetModelString());
            writer.Write(maxNumTrees);
        }

        public static IPredictorWithFeatureWeights<T> Load<T>(BinaryReader reader)
        {
            var flag = reader.ReadInt32();
            var booster = Booster.FromString(reader.ReadString());
            var maxNumTrees = reader.ReadInt32();
            if (typeof(T) == typeof(double))
            {
                if (flag == 0)
                    return new BinaryNativePredictor(booster) { MaxNumTrees = maxNumTrees } as IPredictorWithFeatureWeights<T>;
                else if (flag == 1)
                    return new RegressionNativePredictor(booster) { MaxNumTrees = maxNumTrees } as IPredictorWithFeatureWeights<T>;
                else if (flag == 3)
                    return new RankingNativePredictor(booster) { MaxNumTrees = maxNumTrees } as IPredictorWithFeatureWeights<T>;
                else
                    throw new FormatException("Invalid IPredictorWithFeatureWeights flag");
            }
            else if (typeof(T) == typeof(double[]))
            {
                if (flag == 2)
                    return new MulticlassNativePredictor(booster) { MaxNumTrees = maxNumTrees } as IPredictorWithFeatureWeights<T>;
                else
                    throw new FormatException("Invalid IPredictorWithFeatureWeights flag");
            }
            else
                throw new Exception("Unexpected prediction type");
        }
    }
}
