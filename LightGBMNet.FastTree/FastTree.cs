// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;

namespace LightGBMNet.FastTree
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
        void GetOutput(ref VBuffer<float> input, ref TResult output);
    }

    /// <summary>
    /// Interface implemented by components that can assign weights to features.
    /// </summary>
    public interface IHaveFeatureWeights
    {
        FeatureToGainMap GetFeatureWeights(bool normalise = false, bool splits = false);
    }

    /// <summary>
    /// Interface implemented by predictors that can score features.
    /// </summary>
    public interface IPredictorWithFeatureWeights<TResult> : IPredictorProducing<TResult>, IHaveFeatureWeights
    {
    }

    public abstract class FastTreePredictionWrapper : IPredictorWithFeatureWeights<double>
    {
        public abstract PredictionKind PredictionKind { get; }

        public Ensemble TrainedEnsemble { get; }
        public int NumTrees => TrainedEnsemble.NumTrees;

        // The total number of features used in training (takes the value of zero if the 
        // written version of the loaded model is less than VerNumFeaturesSerialized)
        protected readonly int NumFeatures;
        protected readonly bool AverageOutput = false;

        // Maximum index of the split features of trainedEnsemble trees
        protected readonly int MaxSplitFeatIdx;

        protected FastTreePredictionWrapper(Ensemble trainedEnsemble, int numFeatures, bool avgOutput)
        {
            if (trainedEnsemble == null) throw new ArgumentNullException(nameof(trainedEnsemble));
            if (numFeatures <= 0) throw new ArgumentException(nameof(numFeatures), "must be positive");

            TrainedEnsemble = trainedEnsemble;
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

        protected FastTreePredictionWrapper(BinaryReader reader)
        {
            TrainedEnsemble = new Ensemble(reader);
            NumFeatures = reader.ReadInt32();
            AverageOutput = reader.ReadBoolean();

            MaxSplitFeatIdx = FindMaxFeatureIndex(TrainedEnsemble);
        }

        public FeatureToGainMap GetFeatureWeights(bool normalise = false, bool splits = false)
        {
            return new FeatureToGainMap(TrainedEnsemble.Trees.ToList(), normalise, splits);
        }

        private static int FindMaxFeatureIndex(Ensemble ensemble)
        {
            int ifeatMax = 0;
            for (int i = 0; i < ensemble.NumTrees; i++)
            {
                var tree = ensemble.GetTreeAt(i);
                for (int n = 0; n < tree.NumNodes; n++)
                {
                    int ifeat = tree.SplitFeatures[n];
                    if (ifeat > ifeatMax)
                        ifeatMax = ifeat;
                }
            }

            return ifeatMax;
        }

        public virtual void GetOutput(ref VBuffer<float> src, ref double dst)
        {
            if(!(src.Length > MaxSplitFeatIdx))
                throw new ArgumentException("Feature vector too small");

            dst = TrainedEnsemble.GetOutput(ref src);

            if (AverageOutput)
                dst /= TrainedEnsemble.NumTrees;
        }
        
    }
}
