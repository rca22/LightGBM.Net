// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Float = System.Single;

using System;
using System.Diagnostics;
using System.Linq;
using System.Collections.Generic;
using LightGBMNet;
using LightGBMNet.Interface;
using LightGBMNet.FastTree;

namespace LightGBMNet.Training
{


    public class DataDense
    {
        public float[][] Features;
        public float[] Labels;
        public float[] Weights;
        /// Just for lambdarank
        public int[] Groups;

        public int NumRows => Labels.Length;
        public int NumColumns => Features[0].Length;

        public DataDense() { }

        public void Validate()
        {
            if (Features == null) throw new Exception("Features is null");
            if (Labels == null) throw new Exception("Labels is null");

            var numRows = Labels.Length;
            if (Features.Length != numRows) throw new Exception("Number of Features must match number of Labels");
            if (Features.Length > 0)
            {
                var dim = NumColumns;
                foreach (var row in Features)
                {
                    if (row == null) throw new Exception("All feature vectors must be non-null.");
                    if (row.Length != dim) throw new Exception("Number of columns in all feature vectors must be identical.");
                }
            }

            if (Weights != null && Weights.Length != numRows) throw new Exception("Number of Weights must match number of Labels");
            if (Groups != null && Groups.Sum() != numRows) throw new Exception("Sum of group sizes must match number of Labels");
        }
    }

    /// <summary>
    /// Base class for all training with LightGBM.
    /// </summary>
    public abstract class TrainerBase<TOutput, TPredictor> //: TrainerBase<TPredictor>
        where TPredictor : IPredictorProducing<TOutput>
    {
        //private sealed class CategoricalMetaData
        //{
        //    public int NumCol;
        //    public int TotalCats;
        //    public int[] CategoricalBoundaries;
        //    public int[] OnehotIndices;
        //    public int[] OnehotBias;
        //    public bool[] IsCategoricalFeature;
        //}
        public abstract PredictionKind PredictionKind { get; }
        private protected abstract TPredictor CreatePredictor();

        private protected readonly Parameters Args;

        /// <summary>
        /// Stores argumments as objects to convert them to invariant string type in the end so that
        /// the code is culture agnostic. When retrieving key value from this dictionary as string
        /// please convert to string invariant by string.Format(CultureInfo.InvariantCulture, "{0}", Option[key]).
        /// </summary>
        //private protected readonly Dictionary<string, object> Options;
        //private protected readonly IParallel ParallelTraining;

        // Store _featureCount and _trainedEnsemble to construct predictor.
        private protected int FeatureCount;
        private protected LightGBMNet.FastTree.Ensemble TrainedEnsemble;

        //private static readonly TrainerInfo _info = new TrainerInfo(normalization: false, caching: false, supportValid: true);
        //public override TrainerInfo Info => _info;

        private protected TrainerBase(Parameters args)
        {
            Args = args;
            //ParallelTraining = Args.ParallelTrainer != null ? Args.ParallelTrainer.CreateComponent(env) : new SingleTrainer();
            //InitParallelTraining();
        }

        public TPredictor Train(DataDense trainData, DataDense validData)
        {
            //Host.CheckValue(context, nameof(context));

            //CategoricalMetaData catMetaData;
            Dataset dtrain = null;
            Dataset dvalid = null;
            Booster bst = null;
            TPredictor predictor = default;
            try
            {
                dtrain = LoadTrainingData(trainData);
                if (validData != null)
                    dvalid = LoadValidationData(dtrain, validData);
                bst = TrainCore(dtrain, dvalid);
                TrainedEnsemble = bst.GetModel(); //  catMetaData.CategoricalBoundaries);
                FeatureCount = dtrain.NumFeatures;
                predictor = CreatePredictor();

                // REMOVE ME
                foreach (var row in trainData.Features)
                {
                    var pred1 = bst.PredictForMat(Booster.PredictType.Normal, row);

                    var input = new VBuffer<float>(row.Length, row);
                    TOutput pred2 = default;
                    (predictor as IPredictorProducing<TOutput>).GetOutput(ref input, ref pred2);
                    //var pred2 = TrainedEnsemble.GetOutput(ref input); // raw score

                    if (typeof(TOutput) == typeof(float))
                        Console.WriteLine(String.Join(" ", pred1) + " " + pred2);
                    else
                    {
                        VBuffer<float> probs = (VBuffer<float>) (pred2 as Object);
                        Console.WriteLine(String.Join(" ", pred1) + " (" + pred1.Sum() + ") | " +
                                          String.Join(" ", probs.Values) + " (" + probs.Values.Sum() + ")");
                    }
                }

            }
            finally
            {
                dtrain?.Dispose();
                dvalid?.Dispose();
                bst?.Dispose();
                DisposeParallelTraining();
            }
            return predictor;
        }

        // TODO TODO TODO
        //private void InitParallelTraining()
        //{
        //    if (ParallelTraining.ParallelType() != "serial" && ParallelTraining.NumMachines() > 1)
        //    {
        //        Options["tree_learner"] = ParallelTraining.ParallelType();
        //        var otherParams = ParallelTraining.AdditionalParams();
        //        if (otherParams != null)
        //        {
        //            foreach (var pair in otherParams)
        //                Options[pair.Key] = pair.Value;
        //        }
        //
        //        Contracts.CheckValue(ParallelTraining.GetReduceScatterFunction(), nameof(ParallelTraining.GetReduceScatterFunction));
        //        Contracts.CheckValue(ParallelTraining.GetAllgatherFunction(), nameof(ParallelTraining.GetAllgatherFunction));
        //        LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.NetworkInitWithFunctions(
        //                ParallelTraining.NumMachines(),
        //                ParallelTraining.Rank(),
        //                ParallelTraining.GetReduceScatterFunction(),
        //                ParallelTraining.GetAllgatherFunction()
        //            ));
        //    }
        //}

        private void DisposeParallelTraining()
        {
            //if (ParallelTraining.NumMachines() > 1)
            //    LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.NetworkFree());
        }

        //protected virtual void CheckDataValid(IChannel ch, RoleMappedData data)
        //{
        //    data.CheckFeatureFloatVector();
        //    ch.CheckParam(data.Schema.Label != null, nameof(data), "Need a label column");
        //}

        //protected virtual void GetDefaultParameters(IChannel ch, int numRow, bool hasCategarical, int totalCats, bool hiddenMsg = false)
        //{
        //    double learningRate = Args.LearningRate ?? DefaultLearningRate(numRow, hasCategarical, totalCats);
        //    int numLeaves = Args.NumLeaves ?? DefaultNumLeaves(numRow, hasCategarical, totalCats);
        //    int minDataPerLeaf = Args.MinDataPerLeaf ?? DefaultMinDataPerLeaf(numRow, numLeaves, 1);
        //    Options["learning_rate"] = learningRate;
        //    Options["num_leaves"] = numLeaves;
        //    Options["min_data_per_leaf"] = minDataPerLeaf;
        //    if (!hiddenMsg)
        //    {
        //        if (!Args.LearningRate.HasValue)
        //            ch.Info("Auto-tuning parameters: " + nameof(Args.LearningRate) + " = " + learningRate);
        //        if (!Args.NumLeaves.HasValue)
        //            ch.Info("Auto-tuning parameters: " + nameof(Args.NumLeaves) + " = " + numLeaves);
        //        if (!Args.MinDataPerLeaf.HasValue)
        //            ch.Info("Auto-tuning parameters: " + nameof(Args.MinDataPerLeaf) + " = " + minDataPerLeaf);
        //    }
        //}

        //private FloatLabelCursor.Factory CreateCursorFactory(RoleMappedData data)
        //{
        //    var loadFlags = CursOpt.AllLabels | CursOpt.AllWeights | CursOpt.Features;
        //    if (PredictionKind == PredictionKind.Ranking)
        //        loadFlags |= CursOpt.Group;

        //    var factory = new FloatLabelCursor.Factory(data, loadFlags);
        //    return factory;
        //}

        //private static List<int> GetCategoricalBoundires(int[] categoricalFeatures, int rawNumCol)
        //{
        //    List<int> catBoundaries = new List<int> { 0 };
        //    int curFidx = 0;
        //    int j = 0;
        //    while (curFidx < rawNumCol)
        //    {
        //        if (j < categoricalFeatures.Length && curFidx == categoricalFeatures[j])
        //        {
        //            if (curFidx > catBoundaries[catBoundaries.Count - 1])
        //                catBoundaries.Add(curFidx);
        //            if (categoricalFeatures[j + 1] - categoricalFeatures[j] >= 0)
        //            {
        //                curFidx = categoricalFeatures[j + 1] + 1;
        //                catBoundaries.Add(curFidx);
        //            }
        //            else
        //            {
        //                for (int i = curFidx + 1; i <= categoricalFeatures[j + 1] + 1; ++i)
        //                    catBoundaries.Add(i);
        //                curFidx = categoricalFeatures[j + 1] + 1;
        //            }
        //            j += 2;
        //        }
        //        else
        //        {
        //            catBoundaries.Add(curFidx + 1);
        //            ++curFidx;
        //        }
        //    }
        //    return catBoundaries;
        //}

        //private static List<string> ConstructCategoricalFeatureMetaData(int[] categoricalFeatures, int rawNumCol, ref CategoricalMetaData catMetaData)
        //{
        //    List<int> catBoundaries = GetCategoricalBoundires(categoricalFeatures, rawNumCol);
        //    catMetaData.NumCol = catBoundaries.Count - 1;
        //    catMetaData.CategoricalBoundaries = catBoundaries.ToArray();
        //    catMetaData.IsCategoricalFeature = new bool[catMetaData.NumCol];
        //    catMetaData.OnehotIndices = new int[rawNumCol];
        //    catMetaData.OnehotBias = new int[rawNumCol];
        //    List<string> catIndices = new List<string>();
        //    int j = 0;
        //    for (int i = 0; i < catMetaData.NumCol; ++i)
        //    {
        //        var numCat = catMetaData.CategoricalBoundaries[i + 1] - catMetaData.CategoricalBoundaries[i];
        //        if (numCat > 1)
        //        {
        //            catMetaData.TotalCats += numCat;
        //            catMetaData.IsCategoricalFeature[i] = true;
        //            catIndices.Add(i.ToString());
        //            for (int k = catMetaData.CategoricalBoundaries[i]; k < catMetaData.CategoricalBoundaries[i + 1]; ++k)
        //            {
        //                catMetaData.OnehotIndices[j] = i;
        //                catMetaData.OnehotBias[j] = k - catMetaData.CategoricalBoundaries[i];
        //                ++j;
        //            }
        //        }
        //        else
        //        {
        //            catMetaData.IsCategoricalFeature[i] = false;
        //            catMetaData.OnehotIndices[j] = i;
        //            catMetaData.OnehotBias[j] = 0;
        //            ++j;
        //        }
        //    }
        //    return catIndices;
        //}

        //private CategoricalMetaData GetCategoricalMetaData(IChannel ch, RoleMappedData trainData, int numRow)
        //{
        //    CategoricalMetaData catMetaData = new CategoricalMetaData();
        //    int[] categoricalFeatures = null;
        //    const int useCatThreshold = 50000;
        //    // Disable cat when data is too small, reduce the overfitting.
        //    bool useCat = Args.UseCat ?? numRow > useCatThreshold;
        //    if (!Args.UseCat.HasValue)
        //        ch.Info("Auto-tuning parameters: " + nameof(Args.UseCat) + " = " + useCat);
        //    if (useCat)
        //    {
        //        trainData.Schema.Schema.TryGetColumnIndex(DefaultColumnNames.Features, out int featureIndex);
        //        MetadataUtils.TryGetCategoricalFeatureIndices(trainData.Schema.Schema, featureIndex, out categoricalFeatures);
        //    }
        //    var colType = trainData.Schema.Feature.Type;
        //    int rawNumCol = colType.VectorSize;
        //    FeatureCount = rawNumCol;
        //    catMetaData.TotalCats = 0;
        //    if (categoricalFeatures == null)
        //    {
        //        catMetaData.CategoricalBoundaries = null;
        //        catMetaData.NumCol = rawNumCol;
        //    }
        //    else
        //    {
        //        var catIndices = ConstructCategoricalFeatureMetaData(categoricalFeatures, rawNumCol, ref catMetaData);
        //        // Set categorical features
        //        Options["categorical_feature"] = string.Join(",", catIndices);
        //    }
        //    return catMetaData;
        //}


        private Dataset LoadTrainingData(DataDense trainData)
        {
            if (trainData == null) throw new ArgumentNullException(nameof(trainData));
            trainData.Validate();

            // TODO: not parallelised, better off to concat data and pass in as a single matrix?
            Dataset dtrain = CreateDatasetFromSamplingData2(trainData, Args.Common, Args.Dataset);

            // NOTE: does not respect BinConstructSampleCnt parameter!
            // Dataset dtrain = CreateDatasetFromSamplingData(trainData, Args.Common, Args.Dataset);
            // Push rows into dataset.
            // LoadDataset(trainData, dtrain, Args.Dataset.BatchSize);

            // Some checks.
            //CheckAndUpdateParametersBeforeTraining(ch, trainData, labels, groups);
            return dtrain;
        }

        private Dataset LoadValidationData(Dataset dtrain, DataDense validData)
        {
            if (validData == null) throw new ArgumentNullException(nameof(validData));
            validData.Validate();

            //Dataset dvalid = new Dataset(dtrain, validData.NumRows, validData.Labels, validData.Weights, validData.Groups);
            // Push rows into dataset.
            //LoadDataset(validData, dvalid, Args.Dataset.BatchSize);

            var dvalid = new Dataset( validData.Features
                                    , validData.NumColumns
                                    , Args.Common
                                    , Args.Dataset
                                    , validData.Labels
                                    , validData.Weights
                                    , validData.Groups
                                    , dtrain
                                    );
            return dvalid;
        }

        private Booster TrainCore(Dataset dtrain, Dataset dvalid = null)
        {
            if (dtrain == null) throw new ArgumentNullException(nameof(dtrain));

            // For multi class, the number of labels is required.
            if(!(PredictionKind != PredictionKind.MultiClassClassification || Args.Objective.NumClass > 1))
                throw new Exception("LightGBM requires the number of classes to be specified in the parameters.");

            // Only enable one trainer to run at one time.
            //lock (LightGbmShared.LockForMultiThreadingInside)
            return WrappedLightGbmTraining.Train(Args, dtrain, dvalid: dvalid); //, numIteration: Args.NumBoostRound,
                //verboseEval: Args.VerboseEval, earlyStoppingRound: Args.EarlyStoppingRound))
        }

        /// <summary>
        /// Calculate the density of data. Only use top 1000 rows to calculate.
        /// </summary>
        //private static double DetectDensity(FloatLabelCursor.Factory factory, int numRows = 1000)
        //{
        //    int nonZeroCount = 0;
        //    int totalCount = 0;
        //    using (var cursor = factory.Create())
        //    {
        //        while (cursor.MoveNext() && numRows > 0)
        //        {
        //            nonZeroCount += cursor.Features.Count;
        //            totalCount += cursor.Features.Length;
        //            --numRows;
        //        }
        //    }
        //    return (double)nonZeroCount / totalCount;
        //}

        /// <summary>
        /// Compute row count, list of labels, weights and group counts of the dataset.
        /// </summary>
        //private void GetMetainfo(IChannel ch, FloatLabelCursor.Factory factory,
        //    out int numRow, out float[] labels, out float[] weights, out int[] groups)
        //{
        //    ch.Check(factory.Data.Schema.Label != null, "The data should have label.");
        //    List<float> labelList = new List<float>();
        //    bool hasWeights = factory.Data.Schema.Weight != null;
        //    bool hasGroup = false;
        //    if (PredictionKind == PredictionKind.Ranking)
        //    {
        //        ch.Check(factory.Data.Schema != null, "The data for ranking task should have group field.");
        //        hasGroup = true;
        //    }
        //    List<float> weightList = hasWeights ? new List<float>() : null;
        //    List<ulong> cursorGroups = hasGroup ? new List<ulong>() : null;

        //    using (var cursor = factory.Create())
        //    {
        //        while (cursor.MoveNext())
        //        {
        //            if (labelList.Count == Utils.ArrayMaxSize)
        //                throw ch.Except($"Dataset row count exceeded the maximum count of {Utils.ArrayMaxSize}");
        //            labelList.Add(cursor.Label);
        //            if (hasWeights)
        //            {
        //                // Default weight = 1.
        //                if (float.IsNaN(cursor.Weight))
        //                    weightList.Add(1);
        //                else
        //                    weightList.Add(cursor.Weight);
        //            }
        //            if (hasGroup)
        //                cursorGroups.Add(cursor.Group);
        //        }
        //    }
        //    labels = labelList.ToArray();
        //    ConvertNaNLabels(ch, factory.Data, labels);
        //    numRow = labels.Length;
        //    ch.Check(numRow > 0, "Cannot use empty dataset.");
        //    weights = hasWeights ? weightList.ToArray() : null;
        //    groups = null;
        //    if (hasGroup)
        //    {
        //        List<int> groupList = new List<int>();
        //        int lastGroup = -1;
        //        for (int i = 0; i < numRow; ++i)
        //        {
        //            if (i == 0 || cursorGroups[i] != cursorGroups[i - 1])
        //            {
        //                groupList.Add(1);
        //                ++lastGroup;
        //            }
        //            else
        //                ++groupList[lastGroup];
        //        }
        //        groups = groupList.ToArray();
        //    }
        //}

        ///// <summary>
        ///// Convert Nan labels. Default way is converting them to zero.
        ///// </summary>
        //protected virtual void ConvertNaNLabels(IChannel ch, RoleMappedData data, float[] labels)
        //{
        //    for (int i = 0; i < labels.Length; ++i)
        //    {
        //        if (float.IsNaN(labels[i]))
        //            labels[i] = 0;
        //    }
        //}

        //private static bool MoveMany(FloatLabelCursor cursor, long count)
        //{
        //    for (long i = 0; i < count; ++i)
        //    {
        //        if (!cursor.MoveNext())
        //            return false;
        //    }
        //    return true;
        //}

        //private void GetFeatureValueDense(IChannel ch, FloatLabelCursor cursor, CategoricalMetaData catMetaData, IRandom rand, out float[] featureValues)
        //{
        //    if (catMetaData.CategoricalBoundaries != null)
        //    {
        //        featureValues = new float[catMetaData.NumCol];
        //        for (int i = 0; i < catMetaData.NumCol; ++i)
        //        {
        //            float fv = cursor.Features.Values[catMetaData.CategoricalBoundaries[i]];
        //            if (catMetaData.IsCategoricalFeature[i])
        //            {
        //                int hotIdx = catMetaData.CategoricalBoundaries[i] - 1;
        //                int nhot = 0;
        //                for (int j = catMetaData.CategoricalBoundaries[i]; j < catMetaData.CategoricalBoundaries[i + 1]; ++j)
        //                {
        //                    if (cursor.Features.Values[j] > 0)
        //                    {
        //                        // Reservoir Sampling.
        //                        nhot++;
        //                        var prob = rand.NextSingle();
        //                        if (prob < 1.0f / nhot)
        //                            hotIdx = j;
        //                    }
        //                }
        //                // All-Zero is category 0.
        //                fv = hotIdx - catMetaData.CategoricalBoundaries[i] + 1;
        //            }
        //            featureValues[i] = fv;
        //        }
        //    }
        //    else
        //    {
        //        featureValues = cursor.Features.Values;
        //    }
        //}

        //private void GetFeatureValueSparse(IChannel ch, FloatLabelCursor cursor,
        //    CategoricalMetaData catMetaData, IRandom rand, out int[] indices,
        //    out float[] featureValues, out int cnt)
        //{
        //    if (catMetaData.CategoricalBoundaries != null)
        //    {
        //        List<int> featureIndices = new List<int>();
        //        List<float> values = new List<float>();
        //        int lastIdx = -1;
        //        int nhot = 0;
        //        for (int i = 0; i < cursor.Features.Count; ++i)
        //        {
        //            float fv = cursor.Features.Values[i];
        //            int colIdx = cursor.Features.Indices[i];
        //            int newColIdx = catMetaData.OnehotIndices[colIdx];
        //            if (catMetaData.IsCategoricalFeature[newColIdx])
        //                fv = catMetaData.OnehotBias[colIdx] + 1;
        //            if (newColIdx != lastIdx)
        //            {
        //                featureIndices.Push(newColIdx);
        //                values.Push(fv);
        //                nhot = 1;
        //            }
        //            else
        //            {
        //                // Multi-hot.
        //                ++nhot;
        //                var prob = rand.NextSingle();
        //                if (prob < 1.0f / nhot)
        //                    values[values.Count - 1] = fv;
        //            }
        //            lastIdx = newColIdx;
        //        }
        //        indices = featureIndices.ToArray();
        //        featureValues = values.ToArray();
        //        cnt = featureIndices.Count;
        //    }
        //    else
        //    {
        //        indices = cursor.Features.Indices;
        //        featureValues = cursor.Features.Values;
        //        cnt = cursor.Features.Count;
        //    }
        //}

        // Maximum size of one-dimensional array.
        // See: https://msdn.microsoft.com/en-us/library/hh285054(v=vs.110).aspx
        private const int ArrayMaxSize = 0X7FEFFFFF;

        /// <summary>
        /// Resizes the array if necessary, to ensure that it has at least <paramref name="min"/> elements.
        /// </summary>
        /// <param name="array">The array to resize. Can be null.</param>
        /// <param name="min">The minimum number of items the new array must have.</param>
        /// <param name="keepOld">True means that the old array is preserved, if possible (Array.Resize is called). False
        /// means that a new array will be allocated.
        /// </param>
        /// <returns>The new size, that is no less than <paramref name="min"/>.</returns>
        private static int EnsureSize<T>(ref T[] array, int min, bool keepOld = true)
        {
            return EnsureSize(ref array, min, ArrayMaxSize, keepOld);
        }

        /// <summary>
        /// Resizes the array if necessary, to ensure that it has at least <paramref name="min"/> and at most <paramref name="max"/> elements.
        /// </summary>
        /// <param name="array">The array to resize. Can be null.</param>
        /// <param name="min">The minimum number of items the new array must have.</param>
        /// <param name="max">The maximum number of items the new array can have.</param>
        /// <param name="keepOld">True means that the old array is preserved, if possible (Array.Resize is called). False
        /// means that a new array will be allocated.
        /// </param>
        /// <returns>The new size, that is no less than <paramref name="min"/> and no more that <paramref name="max"/>.</returns>
        private static int EnsureSize<T>(ref T[] array, int min, int max, bool keepOld = true)
        {
            if (!(min <= max)) throw new ArgumentException(nameof(max), "min must not exceed max");
            // This code adapted from the private method EnsureCapacity code of List<T>.
            int size = array?.Length ?? 0;
            if (size >= min)
                return size;
            int newSize = size == 0 ? 4 : size * 2;
            // This constant taken from the internal code of system\array.cs of mscorlib.
            if ((uint)newSize > max)
                newSize = max;
            if (newSize < min)
                newSize = min;
            if (keepOld && size > 0)
                Array.Resize(ref array, newSize);
            else
                array = new T[newSize];
            return newSize;
        }

        /// <summary>
        /// Create a dataset from the sampling data.
        /// </summary>
        private Dataset CreateDatasetFromSamplingData(DataDense data,
                        CommonParameters cp,
                        DatasetParameters dp)
        {

            int numSampleRow = GetNumSampleRow(data.NumRows, data.NumColumns);

            var rand = new Random();
            double averageStep = (double)data.NumRows / numSampleRow;
            int totalIdx = 0;
            int sampleIdx = 0;
            double density = 1; // DetectDensity(factory);

            double[][] sampleValuePerColumn = new double[data.NumColumns][];
            int[][] sampleIndicesPerColumn = new int[data.NumColumns][];
            int[] nonZeroCntPerColumn = new int[data.NumColumns];
            int estimateNonZeroCnt = (int)(numSampleRow * density);
            estimateNonZeroCnt = Math.Max(1, estimateNonZeroCnt);
            for (int i = 0; i < data.NumColumns; i++)
            {
                nonZeroCntPerColumn[i] = 0;
                sampleValuePerColumn[i] = new double[estimateNonZeroCnt];
                sampleIndicesPerColumn[i] = new int[estimateNonZeroCnt];
            };
            //using (var cursor = factory.Create())
            var row = 0;
            {
                int step = 1;
                if (averageStep > 1)
                    step = rand.Next((int)(2 * averageStep - 1)) + 1;
                row += step;
                while (row < data.NumRows)
                {
                    //if (cursor.Features.IsDense)
                    {
                        //GetFeatureValueDense(ch, cursor, catMetaData, rand, out float[] featureValues);
                        var featureValues = data.Features[row];
                        for (int i = 0; i < featureValues.Length; ++i)
                        {
                            float fv = featureValues[i];
                            if (fv == 0)
                                continue;
                            int curNonZeroCnt = nonZeroCntPerColumn[i];
                            EnsureSize(ref sampleValuePerColumn[i], curNonZeroCnt + 1);
                            EnsureSize(ref sampleIndicesPerColumn[i], curNonZeroCnt + 1);
                            sampleValuePerColumn[i][curNonZeroCnt] = fv;
                            sampleIndicesPerColumn[i][curNonZeroCnt] = sampleIdx;
                            nonZeroCntPerColumn[i] = curNonZeroCnt + 1;
                        }
                    }
                    //else
                    //{
                    //    GetFeatureValueSparse(ch, cursor, catMetaData, rand, out int[] featureIndices, out float[] featureValues, out int cnt);
                    //    for (int i = 0; i < cnt; ++i)
                    //    {
                    //        int colIdx = featureIndices[i];
                    //        float fv = featureValues[i];
                    //        if (fv == 0)
                    //            continue;
                    //        int curNonZeroCnt = nonZeroCntPerColumn[colIdx];
                    //        Utils.EnsureSize(ref sampleValuePerColumn[colIdx], curNonZeroCnt + 1);
                    //        Utils.EnsureSize(ref sampleIndicesPerColumn[colIdx], curNonZeroCnt + 1);
                    //        sampleValuePerColumn[colIdx][curNonZeroCnt] = fv;
                    //        sampleIndicesPerColumn[colIdx][curNonZeroCnt] = sampleIdx;
                    //        nonZeroCntPerColumn[colIdx] = curNonZeroCnt + 1;
                    //    }
                    //}
                    totalIdx += step;
                    ++sampleIdx;
                    if (numSampleRow == sampleIdx || data.NumRows == totalIdx)
                        break;
                    averageStep = (double)(data.NumRows - totalIdx) / (numSampleRow - sampleIdx);
                    step = 1;
                    if (averageStep > 1)
                        step = rand.Next((int)(2 * averageStep - 1)) + 1;

                    row += step;
                }
            }
            var dataset = new Dataset( sampleValuePerColumn
                                     , sampleIndicesPerColumn
                                     , data.NumColumns
                                     , nonZeroCntPerColumn
                                     , sampleIdx
                                     , data.NumRows
                                     , cp
                                     , dp
                                     , data.Labels
                                     , data.Weights
                                     , data.Groups
                                     );
            return dataset;
        }

        /// <summary>
        /// Create a dataset from the sampling data.
        /// </summary>
        private Dataset CreateDatasetFromSamplingData2(DataDense data,
                        CommonParameters cp,
                        DatasetParameters dp)
        {
            var dataset = new Dataset( data.Features
                                     , data.NumColumns
                                     , cp
                                     , dp
                                     , data.Labels
                                     , data.Weights
                                     , data.Groups
                                     );
            return dataset;
        }

        /// <summary>
        /// Load dataset. Use row batch way to reduce peak memory cost.
        /// </summary>
        private void LoadDataset(DataDense data, Dataset dataset, int batchSize)
        {
            int numRows = data.NumRows;
            int numCols = data.NumColumns;
            batchSize = Math.Max(batchSize, numCols);
            int numElem = 0;
            int curRowCount = 0;

            //double density = DetectDensity(factory);
            //if (density >= 0.5)
            {
                // number of rows to batch up
                int batchRow = Math.Min(data.NumRows, Math.Max(1, batchSize / numCols));

                float[] features = new float[numCols * batchRow];

                for(int i=0; i < numRows; i++)
                {
                    data.Features[i].CopyTo(features, numElem);
                    numElem += numCols;
                    ++curRowCount;
                    if (batchRow == curRowCount)
                    {
                        Debug.Assert(numElem == curRowCount * numCols);
                        dataset.PushRows(features, curRowCount, numCols, i+1 - curRowCount);
                        curRowCount = 0;
                        numElem = 0;
                    }
                }
                if (curRowCount > 0)
                    dataset.PushRows(features, curRowCount, numCols, numRows - curRowCount);
            }
            
            // sparse??
            //else
            //{
            //    int esimateBatchRow = (int)(batchSize / (catMetaData.NumCol * density));
            //    esimateBatchRow = Math.Max(1, esimateBatchRow);
            //    float[] features = new float[batchSize];
            //    int[] indices = new int[batchSize];
            //    int[] indptr = new int[esimateBatchRow + 1];
            //
            //    using (var cursor = factory.Create())
            //    {
            //        while (cursor.MoveNext())
            //        {
            //            ch.Assert(totalRowCount < numRow);
            //            // Need push rows to LightGBM.
            //            if (numElem + cursor.Features.Count > features.Length)
            //            {
            //                // Mini batch size is greater than size of one row.
            //                // So, at least we have the data of one row.
            //                ch.Assert(curRowCount > 0);
            //                Utils.EnsureSize(ref indptr, curRowCount + 1);
            //                indptr[curRowCount] = numElem;
            //                // PushRows is run by multi-threading inside, so lock here.
            //                lock (LightGbmShared.LockForMultiThreadingInside)
            //                {
            //                    dataset.PushRows(indptr, indices, features,
            //                        curRowCount + 1, numElem, catMetaData.NumCol, totalRowCount - curRowCount);
            //                }
            //                curRowCount = 0;
            //                numElem = 0;
            //            }
            //            Utils.EnsureSize(ref indptr, curRowCount + 1);
            //            indptr[curRowCount] = numElem;
            //            CopyToCsr(ch, cursor, indices, features, catMetaData, rand, ref numElem);
            //            ++totalRowCount;
            //            ++curRowCount;
            //        }
            //        ch.Assert(totalRowCount == numRow);
            //        if (curRowCount > 0)
            //        {
            //            Utils.EnsureSize(ref indptr, curRowCount + 1);
            //            indptr[curRowCount] = numElem;
            //            // PushRows is run by multi-threading inside, so lock here.
            //            lock (LightGbmShared.LockForMultiThreadingInside)
            //            {
            //                dataset.PushRows(indptr, indices, features, curRowCount + 1,
            //                    numElem, catMetaData.NumCol, totalRowCount - curRowCount);
            //            }
            //        }
            //    }
            //}
        }

        //private void CopyToArray(float[][] data, float[] features, ref int numElem)
        //{
        //    if (catMetaData.CategoricalBoundaries != null)
        //    {
        //        if (cursor.Features.IsDense)
        //        {
        //            GetFeatureValueDense(ch, cursor, catMetaData, rand, out float[] featureValues);
        //            for (int i = 0; i < catMetaData.NumCol; ++i)
        //                features[numElem + i] = featureValues[i];
        //            numElem += catMetaData.NumCol;
        //        }
        //        else
        //        {
        //            GetFeatureValueSparse(ch, cursor, catMetaData, rand, out int[] indices, out float[] featureValues, out int cnt);
        //            int lastIdx = 0;
        //            for (int i = 0; i < cnt; i++)
        //            {
        //                int slot = indices[i];
        //                float fv = featureValues[i];
        //                Contracts.Assert(slot >= lastIdx);
        //                while (lastIdx < slot)
        //                    features[numElem + lastIdx++] = 0.0f;
        //                Contracts.Assert(lastIdx == slot);
        //                features[numElem + lastIdx++] = fv;
        //            }
        //            while (lastIdx < catMetaData.NumCol)
        //                features[numElem + lastIdx++] = 0.0f;
        //            numElem += catMetaData.NumCol;
        //        }
        //    }
        //    else
        //    {
        //        cursor.Features.CopyTo(features, numElem, 0.0f);
        //        numElem += catMetaData.NumCol;
        //    }
        //}

        //    private void CopyToCsr(IChannel ch, FloatLabelCursor cursor,
        //        int[] indices, float[] features, CategoricalMetaData catMetaData, IRandom rand, ref int numElem)
        //    {
        //        int numValue = cursor.Features.Count;
        //        if (numValue > 0)
        //        {
        //            ch.Assert(indices.Length >= numElem + numValue);
        //            ch.Assert(features.Length >= numElem + numValue);

        //            if (cursor.Features.IsDense)
        //            {
        //                GetFeatureValueDense(ch, cursor, catMetaData, rand, out float[] featureValues);
        //                for (int i = 0; i < catMetaData.NumCol; ++i)
        //                {
        //                    float fv = featureValues[i];
        //                    if (fv == 0)
        //                        continue;
        //                    features[numElem] = fv;
        //                    indices[numElem] = i;
        //                    ++numElem;
        //                }
        //            }
        //            else
        //            {
        //                GetFeatureValueSparse(ch, cursor, catMetaData, rand, out int[] featureIndices, out float[] featureValues, out int cnt);
        //                for (int i = 0; i < cnt; ++i)
        //                {
        //                    int colIdx = featureIndices[i];
        //                    float fv = featureValues[i];
        //                    if (fv == 0)
        //                        continue;
        //                    features[numElem] = fv;
        //                    indices[numElem] = colIdx;
        //                    ++numElem;
        //                }
        //            }
        //        }
        //    }

        //    private static double DefaultLearningRate(int numRow, bool useCat, int totalCats)
        //    {
        //        if (useCat)
        //        {
        //            if (totalCats < 1e6)
        //                return 0.1;
        //            else
        //                return 0.15;
        //        }
        //        else if (numRow <= 100000)
        //            return 0.2;
        //        else
        //            return 0.25;
        //    }

        //    private static int DefaultNumLeaves(int numRow, bool useCat, int totalCats)
        //    {
        //        if (useCat && totalCats > 100)
        //        {
        //            if (totalCats < 1e6)
        //                return 20;
        //            else
        //                return 30;
        //        }
        //        else if (numRow <= 100000)
        //            return 20;
        //        else
        //            return 30;
        //    }

        //    protected static int DefaultMinDataPerLeaf(int numRow, int numLeaves, int numClass)
        //    {
        //        if (numClass > 1)
        //        {
        //            int ret = numRow / numLeaves / numClass / 10;
        //            ret = Math.Max(ret, 5);
        //            ret = Math.Min(ret, 50);
        //            return ret;
        //        }
        //        else
        //        {
        //            return 20;
        //        }
        //    }

        private static int GetNumSampleRow(int numRow, int numCol)
        {
            // Default is 65536.
            int ret = 1 << 16;
            // If have many features, use more sampling data.
            if (numCol >= 100000)
                ret *= 4;
            ret = Math.Min(ret, numRow);
            return ret;
        }

        //    private protected abstract TPredictor CreatePredictor();

        //    /// <summary>
        //    /// This function will be called before training. It will check the label/group and add parameters for specific applications.
        //    /// </summary>
        //    protected abstract void CheckAndUpdateParametersBeforeTraining(IChannel ch,
        //        RoleMappedData data, float[] labels, int[] groups);

    }
}
