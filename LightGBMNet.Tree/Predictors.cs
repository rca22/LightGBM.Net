// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace LightGBMNet.Tree
{
    public enum PredictionKind
    {
        BinaryClassification = 0,
        MultiClassClassification = 1,
        Regression = 2,
        Ranking = 3,
    }

    /// <summary>
    /// Weakly typed version of IPredictor.
    /// </summary>
    public interface IPredictor
    {
        /// <summary>
        /// Return the type of prediction task.
        /// </summary>
        PredictionKind PredictionKind { get; }
    }

    /// <summary>
    /// A predictor the produces values of the indicated type.
    /// </summary>
    public interface IPredictorProducing<TResult> : IPredictor
    {
        /// <summary>
        /// Dimension of input to GetOutput method
        /// </summary>
        int NumInputs { get; }
        /// <summary>
        /// Evaluate the model on a single row of features
        /// <paramref name="input"/>
        /// <paramref name="output"/>
        /// <paramref name="startIteration"/>
        /// <paramref name="numIterations">Set to -1 to evaluate all trees</paramref>
        /// </summary>
        void GetOutput(ref VBuffer<float> input, ref TResult output, int startIteration, int numIterations);
    }

    public interface IVectorisedPredictorProducing<TResult> : IPredictorProducing<TResult>, ITreeEnsemble
    {
        /// <summary>
        /// Evaluate the model on multiple rows of features
        /// <paramref name="numIterations">Set to -1 to evaluate all trees</paramref>
        /// </summary>
        TResult[] GetOutputs(float[][] rows, int startIteration, int numIterations);
    }

    /// <summary>
    /// Interface implemented by components that can assign weights to features.
    /// </summary>
    public interface IHaveFeatureWeights
    {
        FeatureToGainMap GetFeatureWeights(bool normalise = false, bool splits = false);
        IEnumerable<double> GetFeatureGains(int feature);
    }

    public interface ITreeEnsemble
    {
        /// <summary>
        /// Get: Returns the maximum number of trees in the underlying tree ensemble
        /// Set: Reduces the number of trees in the underlying ensemble to at most numTrees trees, removing trees from the end of the ensemble.
        /// </summary>
        int MaxNumTrees { get; set; }
        /// <summary>
        /// Maximum number of threads to use when evaluating ensemble.
        /// </summary>
        int MaxThreads { get; set; }
    }

    /// <summary>
    /// Interface implemented by predictors that can score features.
    /// </summary>
    public interface IPredictorWithFeatureWeights<TResult> : IPredictorProducing<TResult>, IHaveFeatureWeights, ITreeEnsemble
    {
    }

    /// <summary>
    /// Native predictor with vectorised evaluation
    /// </summary>
    /// <typeparam name="TResult"></typeparam>
    public interface IVectorisedPredictorWithFeatureWeights<TResult> : IVectorisedPredictorProducing<TResult>, IPredictorWithFeatureWeights<TResult>, IDisposable
    {
    }

    public abstract class PredictorBase : IPredictorWithFeatureWeights<double>
    {
        public abstract PredictionKind PredictionKind { get; }

        public Ensemble TrainedEnsemble { get; }
        public int NumInputs => NumFeatures;

        public int MaxNumTrees
        {
            get => TrainedEnsemble.NumTrees;
            set
            {
                if (value <= 0)
                {
                    throw new ArgumentException(nameof(MaxNumTrees), "Must be positive");
                }
                else if (value < TrainedEnsemble.NumTrees)
                {
                    TrainedEnsemble.RemoveAfter(value);
                    System.Diagnostics.Debug.Assert(TrainedEnsemble.NumTrees == value);
                }
            }
        }

        public int MaxThreads
        {
            get => TrainedEnsemble.MaxThreads;
            set => TrainedEnsemble.MaxThreads = value;
        }

        // The total number of features used in training (takes the value of zero if the 
        // written version of the loaded model is less than VerNumFeaturesSerialized)
        protected readonly int NumFeatures;
        public readonly bool AverageOutput = false;

        // Maximum index of the split features of trainedEnsemble trees
        protected readonly int MaxSplitFeatIdx;

        protected PredictorBase(Ensemble trainedEnsemble, int numFeatures, bool avgOutput)
        {
            if (numFeatures <= 0) throw new ArgumentException(nameof(numFeatures), "must be positive");


            TrainedEnsemble = trainedEnsemble ?? throw new ArgumentNullException(nameof(trainedEnsemble));
            NumFeatures = numFeatures;
            AverageOutput = avgOutput && TrainedEnsemble.NumTrees > 0;

            MaxSplitFeatIdx = FindMaxFeatureIndex(trainedEnsemble);
            if (!(NumFeatures > MaxSplitFeatIdx)) throw new ArgumentException("Require NumFeatures > MaxSplitFeatIdx");
        }

        protected virtual void SaveCore(BinaryWriter writer)
        {
            TrainedEnsemble.Save(writer);
            writer.Write(NumFeatures);
            writer.Write(AverageOutput);
        }

        protected PredictorBase(BinaryReader reader, bool legacyVersion)
        {
            TrainedEnsemble = new Ensemble(reader, legacyVersion);
            NumFeatures = reader.ReadInt32();
            AverageOutput = reader.ReadBoolean();

            MaxSplitFeatIdx = FindMaxFeatureIndex(TrainedEnsemble);
        }

        public FeatureToGainMap GetFeatureWeights(bool normalise = false, bool splits = false)
        {
            return new FeatureToGainMap(TrainedEnsemble.Trees.ToList(), normalise, splits);
        }

        public IEnumerable<double> GetFeatureGains(int feature)
        {
            return TrainedEnsemble.Trees.SelectMany(tree => tree.FeatureGains(feature));
        }

        private static int FindMaxFeatureIndex(Ensemble ensemble)
        {
            int ifeatMax = 0;
            for (int i = 0; i < ensemble.NumTrees; i++)
            {
                var tree = ensemble.GetTreeAt(i);
                for (int n = 0; n < tree.Nodes.Length; n++)
                {
                    int ifeat = tree.Nodes[n].SplitFeature;
                    if (ifeat > ifeatMax)
                        ifeatMax = ifeat;
                }
            }

            return ifeatMax;
        }

        public virtual void GetOutput(ref VBuffer<float> src, ref double dst, int startIteration, int numIterations)
        {
            if(!(src.Length > MaxSplitFeatIdx))
                throw new ArgumentException("Feature vector too small");

            dst = TrainedEnsemble.GetOutput(ref src, startIteration, numIterations);

            if (AverageOutput)
            {
                int n = (numIterations == -1) ? (TrainedEnsemble.NumTrees - startIteration) : numIterations;
                if (n > 0)
                    dst /= n;
            }
        }
        
    }

    public sealed class RegressionPredictor : PredictorBase
    {
        public override PredictionKind PredictionKind => PredictionKind.Regression;

        public RegressionPredictor(Ensemble trainedEnsemble, int featureCount, bool avgOutput)
            : base(trainedEnsemble, featureCount, avgOutput)
        {
        }

        public RegressionPredictor(BinaryReader reader, bool legacyVersion) : base(reader, legacyVersion)
        {
        }

        public void Save(BinaryWriter writer)
        {
            base.SaveCore(writer);
        }
    }

    public sealed class BinaryPredictor : PredictorBase
    {
        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        public BinaryPredictor(Ensemble trainedEnsemble, int featureCount, bool avgOutput)
            : base(trainedEnsemble, featureCount, avgOutput)
        {
        }

        private BinaryPredictor(BinaryReader reader, bool legacyVersion) : base(reader, legacyVersion)
        {
        }

        public void Save(BinaryWriter writer)
        {
            base.SaveCore(writer);
        }

        public static BinaryPredictor Create(BinaryReader reader, bool legacyVersion)
        {
            return new BinaryPredictor(reader, legacyVersion);
        }
    }

    public sealed class RankingPredictor : PredictorBase
    {
        public override PredictionKind PredictionKind => PredictionKind.Ranking;

        public RankingPredictor(Ensemble trainedEnsemble, int featureCount, bool avgOutput)
            : base(trainedEnsemble, featureCount, avgOutput)
        {
        }

        private RankingPredictor(BinaryReader reader, bool legacyVersion) : base(reader, legacyVersion)
        {
        }

        public void Save(BinaryWriter writer)
        {
            base.SaveCore(writer);
        }

        public static RankingPredictor Create(BinaryReader reader, bool legacyVersion)
        {
            return new RankingPredictor(reader, legacyVersion);
        }
    }

    public static class PredictorPersist
    {
        public static void Save<T>(IPredictorWithFeatureWeights<T> pred, BinaryWriter writer)
        {
            if (pred is BinaryPredictor b)
            {
                writer.Write(0);
                b.Save(writer);
            }
            else if (pred is RegressionPredictor r)
            {
                writer.Write(1);
                r.Save(writer);
            }
            else if (pred is CalibratedPredictor c)
            {
                writer.Write(2);
                PredictorPersist.Save(c.SubPredictor, writer);
                CalibratorPersist.Save(c.Calibrator, writer);
            }
            else if (pred is RankingPredictor k)
            {
                writer.Write(3);
                k.Save(writer);
            }
            else if (pred is OvaPredictor o)
            {
                writer.Write(o.IsSoftMax);
                writer.Write(o.Predictors.Length);
                foreach (var p in o.Predictors)
                    PredictorPersist.Save(p, writer);
            }
            else
                throw new Exception("Unknown IPredictorWithFeatureWeights type");
        }

        /// <summary>
        /// Load model from binary stream.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="reader"></param>
        /// <param name="legacyVersion">Set to true to load files saved prior to inclusion of linear models at tree leaves</param>
        /// <returns></returns>
        public static IPredictorWithFeatureWeights<T> Load<T>(BinaryReader reader, bool legacyVersion)
        {
            if (typeof(T) == typeof(double))
            {
                var flag = reader.ReadInt32();
                if (flag == 0)
                    return BinaryPredictor.Create(reader, legacyVersion) as IPredictorWithFeatureWeights<T>;
                else if (flag == 1)
                    return new RegressionPredictor(reader, legacyVersion) as IPredictorWithFeatureWeights<T>;
                else if (flag == 2)
                {
                    var pred = PredictorPersist.Load<double>(reader, legacyVersion);
                    var calib = CalibratorPersist.Load(reader);
                    return new CalibratedPredictor(pred, calib) as IPredictorWithFeatureWeights<T>;
                }
                else if (flag == 3)
                    return RankingPredictor.Create(reader, legacyVersion) as IPredictorWithFeatureWeights<T>;
                else
                    throw new FormatException("Invalid IPredictorWithFeatureWeights flag");
            }
            else if (typeof(T) == typeof(double []))
            {
                var isSoftMax = reader.ReadBoolean();
                var len = reader.ReadInt32();
                var predictors = new IPredictorWithFeatureWeights<double>[len];
                for (var i = 0; i < len; i++)
                    predictors[i] = PredictorPersist.Load<double>(reader, legacyVersion);
                return OvaPredictor.Create(isSoftMax, predictors) as IPredictorWithFeatureWeights<T>;
            }
            else
                throw new Exception("Unexpected prediction type");
        }


    }
}
