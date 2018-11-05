// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Diagnostics;
using System.Linq;
using LightGBMNet.Interface;
using LightGBMNet.FastTree;

namespace LightGBMNet.Training
{
    
    public class DataDense
    {
        /// <summary>
        /// Array of rows of feature vectors.
        /// </summary>
        public float[][] Features;
        /// <summary>
        /// Array of labels corresponding to each row in Features.
        /// </summary>
        public float[] Labels;
        /// <summary>
        /// (Optional) Array of training weights corresponding to each row in Features.
        /// </summary>
        public float[] Weights;
        /// <summary>
        /// Array of 
        /// (Must only be specified when ObjectiveType = LambdaRank)
        /// </summary>
        public int[] Groups;

        public int NumRows => Features.Length;
        public int NumColumns => Features[0].Length;

        public DataDense() { }

        public void Validate()
        {
            if (Features == null) throw new Exception("Features is null");
            if (Labels == null) throw new Exception("Labels is null");

            var numRows = NumRows;
            if (Labels.Length != numRows) throw new Exception("Number of Features must match number of Labels");
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

    public class Datasets : IDisposable
    {
        public CommonParameters Common { get; }
        public DatasetParameters Dataset { get; }

        public Dataset Training { get; set; } = null;
        public Dataset Validation { get; set; } = null;

        public Datasets(CommonParameters cp, DatasetParameters dp, DataDense trainData, DataDense validData)
        {
            Common = cp;
            Dataset = dp;

            Training = LoadTrainingData(trainData);
            if (validData != null)
                Validation = LoadValidationData(Training, validData);
        }

        public void Dispose()
        {
            Training?.Dispose();
            Validation?.Dispose();
            Training = null;
            Validation = null;
        }

        private Dataset LoadTrainingData(DataDense trainData)
        {
            if (trainData == null) throw new ArgumentNullException(nameof(trainData));
            trainData.Validate();

            // TODO
          //if (Args.Learning.Objective == ObjectiveType.LambdaRank && trainData.Groups == null)
          //    throw new Exception("Require Groups data for ObjectiveType.LambdaRank");

            // TODO: not parallelised, better off to concat data and pass in as a single matrix?
            Dataset dtrain = CreateDatasetFromSamplingData2(trainData, Common, Dataset);

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

            // TODO
            //if (Args.Learning.Objective == ObjectiveType.LambdaRank && validData.Groups == null)
            //    throw new Exception("Require Groups data for ObjectiveType.LambdaRank");

            //Dataset dvalid = new Dataset(dtrain, validData.NumRows, validData.Labels, validData.Weights, validData.Groups);
            // Push rows into dataset.
            //LoadDataset(validData, dvalid, Args.Dataset.BatchSize);

            var dvalid = new Dataset( validData.Features
                                    , validData.NumColumns
                                    , Common
                                    , Dataset
                                    , validData.Labels
                                    , validData.Weights
                                    , validData.Groups
                                    , dtrain
                                    );
            return dvalid;
        }

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
        [Obsolete]
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
        [Obsolete]
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
        [Obsolete]
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
            var dataset = new Dataset(sampleValuePerColumn
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
            var dataset = new Dataset(data.Features
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
        [Obsolete]
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

                for (int i = 0; i < numRows; i++)
                {
                    data.Features[i].CopyTo(features, numElem);
                    numElem += numCols;
                    ++curRowCount;
                    if (batchRow == curRowCount)
                    {
                        Debug.Assert(numElem == curRowCount * numCols);
                        dataset.PushRows(features, curRowCount, numCols, i + 1 - curRowCount);
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

        [Obsolete]
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

    }

    /// <summary>
    /// Base class for all training with LightGBM.
    /// </summary>
    public abstract class TrainerBase<TOutput> : IDisposable
    {
        public abstract PredictionKind PredictionKind { get; }
        private protected abstract IPredictorWithFeatureWeights<TOutput> CreatePredictor();

        public MetricParameters Metric { get; set; }
        public ObjectiveParameters Objective { get; set; }
        public LearningParameters Learning { get; set; }

        // Store _featureCount and _trainedEnsemble to construct predictor.
        private protected int FeatureCount;
        private protected Ensemble TrainedEnsemble;

        private Booster Booster { get; set; } = null;

        public string GetModelString() => Booster.GetModelString();

        private protected bool AverageOutput => (Learning.Boosting == BoostingType.RandomForest);

        private protected TrainerBase(LearningParameters lp, ObjectiveParameters op, MetricParameters mp)
        {
            Learning = lp;
            Objective = op;
            Metric = mp;
            //ParallelTraining = Args.ParallelTrainer != null ? Args.ParallelTrainer.CreateComponent(env) : new SingleTrainer();
            //InitParallelTraining();
        }

        public void Dispose()
        {
            Booster?.Dispose();
            Booster = null;

            DisposeParallelTraining();
        }

        private Parameters GetParameters(Datasets data)
        {
            var args = new Parameters
            {
                Common = data.Common,
                Dataset = data.Dataset,
                Metric = Metric,
                Objective = Objective,
                Learning = Learning
            };
            return args;
        }

        /// <summary>
        /// Generates files that can be used to run training with lightgbm.exe.
        ///  - train.conf: contains training parameters
        ///  - train.bin: training data
        ///  - valid.bin: validation data (if provided)
        /// Command line: lightgbm.exe config=train.conf
        /// </summary>
        /// <param name="data"></param>
        public void ToCommandLineFiles(Datasets data, string destinationDir = @"c:\temp")
        {
            var pms = GetParameters(data);

            var kvs = pms.ToDict();
            kvs.Add("output_model", Path.Combine(destinationDir, "LightGBM_model.txt"));

            var datafile = Path.Combine(destinationDir, "train.bin");
            if (File.Exists(datafile)) File.Delete(datafile);
            data.Training.SaveBinary(datafile);
            kvs.Add("data", datafile);

            if (data.Validation != null)
            {
                datafile = Path.Combine(destinationDir, "valid.bin");
                if (File.Exists(datafile)) File.Delete(datafile);
                data.Validation.SaveBinary(datafile);
                kvs.Add("valid", datafile);
            }

            using (var file = new StreamWriter(Path.Combine(destinationDir, "train.conf")))
            {
                foreach (var kv in kvs)
                    file.WriteLine($"{kv.Key} = {kv.Value}");
            }

        }

        public IPredictorWithFeatureWeights<TOutput> Train(Datasets data)
        {
            Booster?.Dispose();
            Booster = null;

            var args = GetParameters(data);
            Booster = TrainCore(args, data.Training, data.Validation);

            (var model, var argsout) = Booster.GetModel();
            TrainedEnsemble = model;
            FeatureCount = data.Training.NumFeatures;

            // check parameter strings
            var strIn  = args.ToString();
            var strOut = argsout.ToString();
            if (strIn != strOut)
                throw new Exception($"Parameters differ:\n{strIn}\n{strOut}");

            var predictor = CreatePredictor();            
            return predictor;
        }

        /// <summary>
        /// Evaluates the native LightGBM model on the given feature vector
        /// </summary>
        /// <param name="predictType"></param>
        /// <param name="row"></param>
        /// <returns></returns>
        public double [] Evaluate(Booster.PredictType predictType, float[] row)
        {
            if (Booster == null) throw new Exception("Model has not be trained");
            var rslt = Booster.PredictForMat(predictType, row); // Booster.PredictType.Normal
            return rslt;
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


        private Booster TrainCore(Parameters args, Dataset dtrain, Dataset dvalid = null)
        {
            if (dtrain == null) throw new ArgumentNullException(nameof(dtrain));

            // For multi class, the number of labels is required.
            if (!(PredictionKind != PredictionKind.MultiClassClassification || Objective.NumClass > 1))
                throw new Exception("LightGBM requires the number of classes to be specified in the parameters.");

            return WrappedLightGbmTraining.Train(args, dtrain, dvalid: dvalid);
        }

    }

    public static class PredictorPersist
    {
        public static void Save(IPredictorWithFeatureWeights<double> pred, BinaryWriter writer)
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
            else
                throw new Exception("Unknown IPredictorWithFeatureWeights type");
        }

        public static IPredictorWithFeatureWeights<double> Load(BinaryReader reader)
        {
            var flag = reader.ReadInt32();
            if (flag == 0)
                return BinaryPredictor.Create(reader);
            else if (flag == 1)
                return new RegressionPredictor(reader);
            else if (flag == 2)
            {
                var pred = PredictorPersist.Load(reader);
                var calib = CalibratorPersist.Load(reader);
                return new CalibratedPredictor(pred, calib);
            }
            else if (flag == 3)
                return RankingPredictor.Create(reader);
            else
                throw new FormatException("Invalid IPredictorWithFeatureWeights flag");
        }

        public static void Save(IPredictorWithFeatureWeights<VBuffer<double>> input, BinaryWriter writer)
        {
            var pred = input as OvaPredictor;
            if (pred == null) throw new Exception("Unexpected predictor type");
            writer.Write(pred.IsSoftMax);
            writer.Write(pred.Predictors.Length);
            foreach (var p in pred.Predictors)
                PredictorPersist.Save(p, writer);
        }

        public static IPredictorWithFeatureWeights<VBuffer<double>> LoadMulti(BinaryReader reader)
        {
            var isSoftMax = reader.ReadBoolean();
            var len = reader.ReadInt32();
            var predictors = new IPredictorWithFeatureWeights<double>[len];
            for (var i = 0; i < len; i++)
                predictors[i] = PredictorPersist.Load(reader);
            return OvaPredictor.Create(isSoftMax, predictors);
        }

    }
}
