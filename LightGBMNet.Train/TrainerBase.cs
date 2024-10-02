// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using LightGBMNet.Tree;

namespace LightGBMNet.Train
{    

    public class Datasets : IDisposable
    {
        public CommonParameters Common { get; }
        public DatasetParameters Dataset { get; }

        public Dataset Training { get; set; } = null;
        public Dataset Validation { get; set; } = null;

        public Datasets(CommonParameters cp, DatasetParameters dp, Dataset trainData, Dataset validData)
        {
            Common = cp;
            Dataset = dp;
            Training = trainData;
            Validation = validData;
        }

        public Datasets(CommonParameters cp, DatasetParameters dp, DataDense trainData, DataDense validData)
        {
            Common = cp;
            Dataset = dp;

            Training = LoadTrainingData(trainData);
            if (validData != null)
                Validation = LoadValidationData(Training, validData);
        }

        public Datasets(CommonParameters cp, DatasetParameters dp, DataSparse trainData, DataSparse validData)
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

            // TODO: not parallelised, better off to concat data and pass in as a single matrix?
            Dataset dtrain = CreateDatasetFromSamplingData(trainData, Common, Dataset);
            return dtrain;
        }

        private Dataset LoadTrainingData(DataSparse trainData)
        {
            if (trainData == null) throw new ArgumentNullException(nameof(trainData));
            trainData.Validate();

            // TODO: not parallelised, better off to concat data and pass in as a single matrix?
            Dataset dtrain = CreateDatasetFromSamplingData(trainData, Common, Dataset);
            return dtrain;
        }

        private Dataset LoadValidationData(Dataset dtrain, DataDense validData)
        {
            if (validData == null) throw new ArgumentNullException(nameof(validData));
            validData.Validate();

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

        private Dataset LoadValidationData(Dataset dtrain, DataSparse validData)
        {
            if (validData == null) throw new ArgumentNullException(nameof(validData));
            validData.Validate();

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

        /// <summary>
        /// Create a dataset from the sampling data.
        /// </summary>
        private Dataset CreateDatasetFromSamplingData(DataDense data,
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
        /// Create a dataset from the sampling data.
        /// </summary>
        private Dataset CreateDatasetFromSamplingData(DataSparse data,
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

    }
    
    public class Predictors<TOutput> : IDisposable
    {
        /// <summary>
        /// Managed predictor
        /// </summary>
        public IPredictorWithFeatureWeights<TOutput> Managed { get; }
        public IVectorisedPredictorWithFeatureWeights<TOutput> Native { get; }

        public Predictors(IPredictorWithFeatureWeights<TOutput> managed, IVectorisedPredictorWithFeatureWeights<TOutput> native)
        {
            Managed = managed;
            Native = native;
        }

        #region IDisposable
        public void Dispose()
        {
            if (Native != null)
                Native.Dispose();
        }
        #endregion
    }

    /// <summary>
    /// Base class for all training with LightGBM.
    /// </summary>
    public abstract class TrainerBase<TOutput> : IDisposable
    {
        public abstract PredictionKind PredictionKind { get; }
        private protected abstract IPredictorWithFeatureWeights<TOutput> CreateManagedPredictor();
        private protected abstract IVectorisedPredictorWithFeatureWeights<TOutput> CreateNativePredictor();
        public ObjectiveParameters Objective { get; set; }
        public LearningParameters Learning { get; set; }

        // Store _featureCount and _trainedEnsemble to construct predictor.
        private protected int FeatureCount;
        private protected Ensemble TrainedEnsemble;

        protected Booster Booster { get; set; } = null;
        protected Datasets Datasets { get; set; } = null;

        public Dictionary<int, double> TrainMetrics { get; } = new Dictionary<int, double>();
        public Dictionary<int, double> ValidMetrics { get; } = new Dictionary<int, double>();

        public string GetModelString() => Booster.GetModelString();

        private protected bool AverageOutput => (Learning.Boosting == BoostingType.RandomForest);

        private protected TrainerBase(LearningParameters lp, ObjectiveParameters op)
        {
            Learning = lp;
            Objective = op;

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

        public Predictors<TOutput> Train( Datasets data
                                        , Func<int, double> learningRateSchedule = null     // optional: learning rate as a function of iteration (zero-based)
                                        , bool createNativePredictor = true
                                        )
        {
            // For multi class, the number of labels is required.
            if (!(PredictionKind != PredictionKind.MultiClassClassification || Objective.NumClass > 1))
                throw new Exception("LightGBM requires the number of classes to be specified in the parameters for multi-class classification.");

            if (PredictionKind == PredictionKind.Ranking)
            {
                if (data.Training.GetGroups() == null)
                    throw new Exception("Require Groups training data for ObjectiveType.LambdaRank");
                if (data.Validation != null && data.Validation.GetGroups() == null)
                    throw new Exception("Require Groups validation data for ObjectiveType.LambdaRank");
            }

            TrainMetrics.Clear();
            ValidMetrics.Clear();
            Booster?.Dispose();
            Booster = null;

            Datasets = data;

            var args = GetParameters(data);
            Booster = Train(args, data.Training, data.Validation, TrainMetrics, ValidMetrics, learningRateSchedule);

            (var model, var argsout, var numFeatures) = Ensemble.GetModelFromString(Booster.GetModelString());
            TrainedEnsemble = model;
            FeatureCount = data.Training.NumFeatures;

            // check parameter strings
            if (learningRateSchedule != null)
                argsout.Learning.LearningRate = args.Learning.LearningRate;
            // if both ForceColWise and ForceRowWise are false on the input, LightGBM appears to set one of them to be true on the output?
            if (!args.Learning.ForceColWise && !args.Learning.ForceRowWise)
            {
                argsout.Learning.ForceColWise = false;
                argsout.Learning.ForceRowWise = false;
            }
            // for some reason this parameter is not returned in the output model
            argsout.Objective.MetricFreq = args.Objective.MetricFreq;

            var strIn  = args.ToString();
            var strOut = argsout.ToString();
            if (strIn != strOut)
                throw new Exception($"Parameters differ:\n{strIn}\n{strOut}");

            var managed = CreateManagedPredictor();
            var native = createNativePredictor ? CreateNativePredictor() : null;
            return new Predictors<TOutput>(managed, native);
        }

        /// <summary>
        /// Continue training current model, optionally with a new training dataset.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="learningRateSchedule"></param>
        /// <returns></returns>
        public Predictors<TOutput> ContinueTraining( Dataset trainingData = null
                                                   , Func<int, double> learningRateSchedule = null     // optional: learning rate as a function of iteration (zero-based)
                                                   , bool createNativePredictor = true
                                                   )
        {
            if (Booster == null)
                throw new Exception("No existing booster to train.");
            if (Datasets == null)
                throw new Exception("No existing data sets.");

            if (trainingData != null)
            {
                Datasets.Training = trainingData;
                Booster.ResetTrainingData(trainingData);
            }

            // For multi class, the number of labels is required.
            if (!(PredictionKind != PredictionKind.MultiClassClassification || Objective.NumClass > 1))
                throw new Exception("LightGBM requires the number of classes to be specified in the parameters for multi-class classification.");

            if (PredictionKind == PredictionKind.Ranking)
            {
                if (Datasets.Training.GetGroups() == null)
                    throw new Exception("Require Groups training data for ObjectiveType.LambdaRank");
                if (Datasets.Validation != null && Datasets.Validation.GetGroups() == null)
                    throw new Exception("Require Groups validation data for ObjectiveType.LambdaRank");
            }

            // NOTE: existing metrics cleared
            TrainMetrics.Clear();
            ValidMetrics.Clear();

            var args = GetParameters(Datasets);
            // TODO: HOW TO RESET VALIDATION DATA???
            Train(args, Booster, (Datasets.Validation != null), TrainMetrics, ValidMetrics, learningRateSchedule);

            (var model, var argsout, var numFeatures) = Ensemble.GetModelFromString(Booster.GetModelString());
            TrainedEnsemble = model;
            FeatureCount = Datasets.Training.NumFeatures;

            // check parameter strings
            if (learningRateSchedule != null)
                argsout.Learning.LearningRate = args.Learning.LearningRate;
            // if both ForceColWise and ForceRowWise are false on the input, LightGBM appears to set one of them to be true on the output?
            if (!args.Learning.ForceColWise && !args.Learning.ForceRowWise)
            {
                argsout.Learning.ForceColWise = false;
                argsout.Learning.ForceRowWise = false;
            }
            // for some reason this parameter is not returned in the output model
            argsout.Objective.MetricFreq = args.Objective.MetricFreq;

            var strIn = args.ToString();
            var strOut = argsout.ToString();
            if (strIn != strOut)
                throw new Exception($"Parameters differ:\n{strIn}\n{strOut}");

            var managed = CreateManagedPredictor();
            var native = createNativePredictor ? CreateNativePredictor() : null;
            return new Predictors<TOutput>(managed, native);
        }


        /// <summary>
        /// Evaluates the native LightGBM model on the given feature vector
        /// </summary>
        /// <param name="predictType"></param>
        /// <param name="row"></param>
        /// <returns></returns>
        public double [] Evaluate(Booster.PredictType predictType, float[] row, int startIteration, int numIteration)
        {
            if (Booster == null) throw new Exception("Model has not been trained");
            var rslt = Booster.PredictForMat(predictType, row, startIteration, numIteration);
            return rslt;
        }

        /// <summary>
        /// Evaluates the native LightGBM model on the given feature vector
        /// </summary>
        /// <param name="predictType"></param>
        /// <param name="row"></param>
        /// <returns></returns>
        public double[,] Evaluate(Booster.PredictType predictType, float[][] rows, int startIteration, int numIterations)
        {
            if (Booster == null) throw new Exception("Model has not been trained");
            var rslt = Booster.PredictForMatsMulti(predictType, rows, startIteration, numIterations);
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

        /// <summary>
        /// Train and return a booster.
        /// </summary>
        private static Booster Train( Parameters parameters
                                    , Dataset dtrain
                                    , Dataset dvalid
                                    , Dictionary<int, double> trainMetrics
                                    , Dictionary<int, double> validMetrics
                                    , Func<int, double> learningRateSchedule // optional: learning rate as a function of iteration (zero-based)
                                    )
        {
            if (dtrain == null) throw new ArgumentNullException(nameof(dtrain));

            // create Booster.
            Booster bst = new Booster(parameters, dtrain, dvalid);

            Train(parameters, bst, (dvalid != null), trainMetrics, validMetrics, learningRateSchedule);

            return bst;
        }

        /// <summary>
        /// Train an existing booster.
        /// </summary>
        private static void Train( Parameters parameters
                                 , Booster bst
                                 , bool haveValidationData
                                 , Dictionary<int, double> trainMetrics
                                 , Dictionary<int, double> validMetrics
                                 , Func<int, double> learningRateSchedule // optional: learning rate as a function of iteration (zero-based)
                                 )
        {
            if (bst == null) throw new ArgumentNullException(nameof(bst));

            // Disable early stopping if we don't have validation data.
            var numIteration = parameters.Learning.NumIterations;
            var earlyStoppingRound = parameters.Learning.EarlyStoppingRound;
            if (!haveValidationData && earlyStoppingRound > 0)
            {
                earlyStoppingRound = 0;
                if (parameters.Common.Verbosity >= VerbosityType.Error)
                    Console.WriteLine("Validation dataset not present, early stopping will be disabled.");
            }

            int bestIter = 0;
            double bestScore = double.MaxValue;
            double factorToSmallerBetter = 1.0;

            var metric = parameters.Objective.Metric;
            if (earlyStoppingRound > 0 && (metric == MetricType.Auc || metric == MetricType.Ndcg || metric == MetricType.Map))
                factorToSmallerBetter = -1.0;

            int evalFreq = parameters.Objective.MetricFreq;

            double validError = double.NaN;
            double trainError = double.NaN;
            int iter = 0;
            for (iter = 0; iter < numIteration; ++iter)
            {
                if (learningRateSchedule != null)
                {
                    var learningRate = learningRateSchedule.Invoke(iter);
                    bst.SetLearningRate(learningRate);
                }

                if (bst.Update())
                    break;

                if (earlyStoppingRound > 0)
                {
                    validError = bst.EvalValid();
                    validMetrics.Add(iter+1, validError);
                    if (validError * factorToSmallerBetter < bestScore)
                    {
                        bestScore = validError * factorToSmallerBetter;
                        bestIter = iter;
                    }
                    if (iter - bestIter >= earlyStoppingRound)
                    {
                        if (parameters.Common.Verbosity >= VerbosityType.Info)
                            Console.WriteLine($"Met early stopping, best iteration: {bestIter + 1}, best score: {bestScore / factorToSmallerBetter}");
                        break;
                    }
                }

                if ((iter + 1) % evalFreq == 0)
                {
                    trainError = bst.EvalTrain();
                    trainMetrics.Add(iter+1, trainError);
                    if (haveValidationData)
                    {
                        if (earlyStoppingRound == 0)
                        {
                            validError = bst.EvalValid();
                            validMetrics.Add(iter+1, validError);
                        }
                        if (parameters.Common.Verbosity >= VerbosityType.Info)
                            Console.WriteLine($"Iters: {iter+1}, Training Metric: {trainError}, Validation Metric: {validError}");
                    }
                }
            }

            // Add final metrics
            if (!trainMetrics.ContainsKey(iter))
                trainMetrics.Add(iter, bst.EvalTrain());

            if (haveValidationData && !validMetrics.ContainsKey(iter))
                validMetrics.Add(iter, bst.EvalValid());

            // Set the BestIteration.
            if (iter != numIteration && earlyStoppingRound > 0)
                bst.BestIteration = bestIter + 1;
        }
    }
}
