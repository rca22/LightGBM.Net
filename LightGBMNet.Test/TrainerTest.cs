using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Xunit.Abstractions;
using LightGBMNet.Tree;

namespace LightGBMNet.Train.Test
{
    public class TestBase
    {
        protected readonly ITestOutputHelper output;

        protected TestBase(ITestOutputHelper output)
        {
            this.output = output;
        }

        protected static void Compare(double x, double y)
        {
            if (x == y || double.IsNaN(x) && double.IsNaN(y))
                return;

            if (Math.Abs(x - y) / (1 + Math.Abs(x)) > 1e-6)
                throw new Exception($"Mismatch {x} vs {y} (error: {Math.Abs(x - y)})");
        }

        protected static BoostingType[] boostingTypes =
            new BoostingType[] {
                BoostingType.GBDT,
                BoostingType.RandomForest,
                BoostingType.Dart,
                BoostingType.Goss
            };

        protected static Parameters GenerateParameters(Random rand, ObjectiveType objective, int numColumns)
        {
            var pms = new Parameters();
            pms.Learning.NumIterations = rand.Next(1, 100);
            pms.Common.Verbosity = VerbosityType.Error;

            pms.Objective.Objective = objective;
            pms.Learning.Boosting = boostingTypes[rand.Next(boostingTypes.Length)];

            if (pms.Learning.Boosting == BoostingType.RandomForest)
            {
                pms.Learning.BaggingFreq = rand.Next(1, 10);
                pms.Learning.BaggingFraction = rand.Next(1, 99) / 100.0;
                pms.Learning.FeatureFraction = rand.Next(1, 99) / 100.0;
            }
            if (rand.Next(2) == 0) pms.Learning.LearningRate = rand.Next(1, 100) / 1e3;

            if (objective == ObjectiveType.MultiClass || objective == ObjectiveType.MultiClassOva)
                pms.Objective.NumClass = rand.Next(2, 4);

            if (objective == ObjectiveType.Binary || objective == ObjectiveType.MultiClassOva || objective == ObjectiveType.LambdaRank)
                if (rand.Next(2) == 0) pms.Objective.Sigmoid = rand.Next(1, 100) / 100.0;

            if (rand.Next(2) == 0) pms.Dataset.MaxBin = 64;
            if (rand.Next(2) == 0) pms.Dataset.MinDataInBin = rand.Next(1, 10);
            if (rand.Next(2) == 0) pms.Dataset.BinConstructSampleCnt = rand.Next(100, 1000);
            pms.Dataset.EnableBundle = (rand.Next(2) == 0);
            pms.Dataset.IsEnableSparse = (rand.Next(2) == 0);
            pms.Dataset.UseMissing = (rand.Next(2) == 0);
            if (rand.Next(2) == 0) pms.Dataset.MinDataInLeaf = rand.Next(1, 20);
            if (rand.Next(2) == 0) pms.Dataset.DataRandomSeed = rand.Next(1, 20);
            if (rand.Next(2) == 0) pms.Dataset.MonotoneConstraints = Enumerable.Range(0, numColumns).Select(x => rand.Next(2) - 1).ToArray();
            if (rand.Next(2) == 0) pms.Dataset.FeatureContri = Enumerable.Range(0, numColumns).Select(x => rand.Next(1, 100) / 100.0).ToArray();
            if (rand.Next(2) == 0) pms.Learning.EarlyStoppingRound = rand.Next(1, 20);
            if (rand.Next(2) == 0) pms.Common.LinearTree = true;
            if (!pms.Common.LinearTree)
                // disable GPU test, using native DLL compiled for CPU only
                if (false && rand.Next(2) == 0) pms.Common.DeviceType = DeviceType.GPU;

            pms.Objective.MetricFreq = rand.Next(1, 20);
            pms.Dataset.PreciseFloatParser = true;
            return pms;
        }


        protected static DataDense CreateRandomDenseData(
            Random rand,
            ref Dictionary<int, int> categorical,
            bool useMissing,
            int numColumns
            )
        {
            var numRows = rand.Next(100, 500);

            // from column index to number of classes
            if (categorical == null)
            {
                categorical = new Dictionary<int, int>();
                if (rand.Next(2) == 0)
                {
                    for (int j = 0; j < numColumns; j++)
                    {
                        if (rand.Next(10) == 0)
                            categorical.Add(j, rand.Next(2, 50));
                    }
                }
            }

            var scales = Enumerable.Range(0, numColumns).Select(x => Math.Pow(10.0, rand.Next(-10, 10))).ToArray();

            var rows = new float[numRows][];
            var weights = (rand.Next(2) == 0) ? new float[numRows] : null;
            for (int i = 0; i < numRows; ++i)
            {
                var row = new float[numColumns];
                for (int j = 0; j < row.Length; ++j)
                {
                    if (useMissing && rand.Next(50) == 0)
                    {
                        row[j] = float.NaN;
                    }
                    else
                    {
                        if (categorical.TryGetValue(j, out int numClass))
                            row[j] = rand.Next(numClass);
                        else
                            row[j] = (rand.Next(100) == 0) ? 0.0f : (float)(scales[j] * (rand.NextDouble() - 0.5));
                    }
                }
                rows[i] = row;
                if (weights != null) weights[i] = (float)rand.NextDouble();
            }

            var rslt = new DataDense
            {
                Features = rows,
                Weights = weights,
                Groups = null
            };
            return rslt;
        }

        protected static DataSparse Dense2Sparse(DataDense data)
        {
            return (data == null) ? null : new DataSparse()
            {
                Features = DatasetTest.Dense2Sparse(data.Features),
                Labels = data.Labels,
                Weights = data.Weights,
                Groups = data.Groups
            };
        }

        protected static DataDense CreateRandomDenseClassifyData(
            Random rand,
            int numClasses,
            ref Dictionary<int, int> categorical,
            bool useMissing,
            int numColumns
            )
        {
            var rslt = CreateRandomDenseData(rand, ref categorical, useMissing, numColumns);

            var labels = new float[rslt.NumRows];
            for (int i = 0; i < labels.Length; ++i)
                labels[i] = rand.Next(numClasses);

            rslt.Labels = labels;
            rslt.Validate();
            return rslt;
        }

        protected static DataDense CreateRandomDenseRegressionData(
            Random rand,
            ref Dictionary<int, int> categorical,
            bool useMissing,
            int numColumns = -1
            )
        {
            var rslt = CreateRandomDenseData(rand, ref categorical, useMissing, numColumns);
            var useSum = rand.Next(2) == 0;
            var scale = Math.Pow(10.0, rand.Next(-10, 10));
            var labels = new float[rslt.NumRows];
            for (int i = 0; i < labels.Length; ++i)
                labels[i] = useSum ? rslt.Features[i].Sum() : (float)(scale * (rand.NextDouble() - 0.5));

            rslt.Labels = labels;
            rslt.Validate();
            return rslt;
        }

    }

    public class TrainerTest : TestBase
    {
        private static readonly int Seed = (new Random()).Next();

        public TrainerTest(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TrainBinary()
        {
            var rand = new Random(Seed);
            for (int test = 0; test < 5; ++test)
            {
                int numColumns = rand.Next(1, 10);
                var pms = GenerateParameters(rand, ObjectiveType.Binary, numColumns);
              //pms.Common.NumThreads = 1;
                Dictionary<int, int> categorical = null;
                var trainData = CreateRandomDenseClassifyData(rand, 2, ref categorical, pms.Dataset.UseMissing, numColumns);
                var validData = (pms.Learning.EarlyStoppingRound > 0 || rand.Next(2) == 0) ? CreateRandomDenseClassifyData(rand, 2, ref categorical, pms.Dataset.UseMissing, numColumns) : null;
                // The output cannot be monotone with respect to categorical features
                if (categorical != null && pms.Dataset.MonotoneConstraints.Length > 0)
                {
                    foreach (int i in categorical.Keys)
                        pms.Dataset.MonotoneConstraints[i] = 0;
                }
                pms.Dataset.CategoricalFeature = categorical.Keys.ToArray();

                var learningRateSchedule = (rand.Next(2) == 0) ? (Func<int, double>)null : (iter => pms.Learning.LearningRate * Math.Pow(0.99, iter));

                try
                {
                    using (var datasets = (rand.Next(2) == 0) ? new Datasets(pms.Common, pms.Dataset, trainData, validData) :
                                                                new Datasets(pms.Common, pms.Dataset, Dense2Sparse(trainData), Dense2Sparse(validData)))
                    using (var trainer = new BinaryTrainer(pms.Learning, pms.Objective))
                    {
                        //trainer.ToCommandLineFiles(datasets);

                        var datasets2 = (rand.Next(2) == 0) ? null : datasets.Training.GetSubset(Enumerable.Range(0, datasets.Training.NumRows/2).ToArray());

                        var model = trainer.Train(datasets, learningRateSchedule);
                        {
                            if (datasets2 != null)
                            {
                                model.Dispose();
                                model = trainer.ContinueTraining(datasets2, learningRateSchedule);
                            }

                            model.Managed.MaxThreads = rand.Next(1, Environment.ProcessorCount);

                            // possibly use subset of trees
                            var numIterations = -1;
                            if (rand.Next(2) == 0)
                            {
                                numIterations = rand.Next(1, model.Managed.MaxNumTrees);
                                model.Managed.MaxNumTrees = numIterations;
                                model.Native.MaxNumTrees = numIterations;
                            }

                            CalibratedPredictor model2 = null;
                            using (var ms = new System.IO.MemoryStream())
                            using (var writer = new System.IO.BinaryWriter(ms))
                            using (var reader = new System.IO.BinaryReader(ms))
                            {
                                PredictorPersist.Save(model.Managed, writer);
                                ms.Position = 0;
                                model2 = PredictorPersist.Load<double>(reader, false) as CalibratedPredictor;
                                Assert.Equal(ms.Position, ms.Length);
                            }

                            BinaryNativePredictor model2native = null;
                            using (var ms = new System.IO.MemoryStream())
                            using (var writer = new System.IO.BinaryWriter(ms))
                            using (var reader = new System.IO.BinaryReader(ms))
                            {
                                NativePredictorPersist.Save(model.Native, writer);
                                ms.Position = 0;
                                model2native = NativePredictorPersist.Load<double>(reader) as BinaryNativePredictor;
                                Assert.Equal(ms.Position, ms.Length);
                            }

                            var rawscore2s = trainer.Evaluate(Booster.PredictType.RawScore, trainData.Features, 0, numIterations);
                            Assert.Equal(trainData.Features.Length, rawscore2s.GetLength(0));
                            Assert.Equal(1, rawscore2s.GetLength(1));

                            var output3s = trainer.Evaluate(Booster.PredictType.Normal, trainData.Features, 0, numIterations);
                            Assert.Equal(trainData.Features.Length, output3s.GetLength(0));
                            Assert.Equal(1, output3s.GetLength(1));

                            var output3natives = model.Native.GetOutputs(trainData.Features, 0, numIterations);
                            Assert.Equal(trainData.Features.Length, output3s.Length);

                            for (int i = 0; i < trainData.Features.Length; i++)
                            {
                                var row = trainData.Features[i];

                                double output = 0;
                                var input = new VBuffer<float>(row.Length, row);
                                model.Managed.GetOutput(ref input, ref output, 0, numIterations);
                                Assert.True(output >= 0);
                                Assert.True(output <= 1);

                                double output2 = 0;
                                model2.GetOutput(ref input, ref output2, 0, numIterations);
                                Assert.Equal(output, output2);

                                // check raw score against native booster object
                                var rawscore = 0.0;
                                (model.Managed as CalibratedPredictor).SubPredictor.GetOutput(ref input, ref rawscore, 0, numIterations);
                                var rawscore2 = trainer.Evaluate(Booster.PredictType.RawScore, row, 0, numIterations);
                                Assert.Single(rawscore2);
                                Assert.Equal(rawscore2[0], rawscore2s[i, 0]);
                                var isRf = (pms.Learning.Boosting == BoostingType.RandomForest);
                                Compare(isRf ? rawscore * model.Managed.MaxNumTrees : rawscore, rawscore2[0]);

                                var output3 = trainer.Evaluate(Booster.PredictType.Normal, row, 0, numIterations);
                                Assert.Single(output3);
                                Assert.Equal(output3[0], output3s[i, 0]);
                                Assert.Equal(output3[0], output3natives[i]);
                                Compare(output, output3[0]);

                                double outputNative = 0;
                                model.Native.GetOutput(ref input, ref outputNative, 0, numIterations);
                                Assert.Equal(outputNative, output3[0]);

                                // need to use Compare here, since model does not perfectly round trip numbers.
                                // for example 0.00092811701636191693 is read back in as 0.00092811701636191682
                                model2native.GetOutput(ref input, ref outputNative, 0, numIterations);
                                //if (outputNative != output3[0])
                                //{
                                //    var s1 = (model.Native as BinaryNativePredictor).Booster.GetModelString();
                                //    var s2 = model2native.Booster.GetModelString();
                                //    for (var j=0; j < Math.Min(s1.Length, s2.Length); j++)
                                //        if (s1[j] != s2[j])
                                //        {
                                //            var d1 = s1.Substring(j, 10);
                                //            var d2 = s2.Substring(j, 10);
                                //            Console.WriteLine($"Strings differ at index {j}: {d1} vs {d2}");
                                //        }
                                //}
                                Assert.Equal(outputNative, output3[0]);

                                //Console.WriteLine(trainer.GetModelString());
                                //throw new Exception($"Output mismatch {output} vs {output3[0]} (error: {Math.Abs(output - output3[0])}) input: {String.Join(", ", row)}");
                            }

                            var normalise = rand.Next(2) == 0;
                            var getSplits = rand.Next(2) == 0;
                            var gains = model.Managed.GetFeatureWeights(normalise, getSplits);
                            var gainsNative = model.Native.GetFeatureWeights(normalise, getSplits);
                            Assert.Equal(gains.Count, gainsNative.Count);
                            foreach (var kv in gains)
                            {
                                Assert.True(0 <= kv.Key && kv.Key < trainData.NumColumns);
                                Assert.True(0.0 <= kv.Value);
                                Compare(kv.Value, gainsNative[kv.Key]);
                            }

                            if (!getSplits && !normalise)
                            {
                                var totGain1 = gains.Values.Sum();
                                var totGain2 = Enumerable.Range(0, trainData.NumColumns).SelectMany(i => model.Managed.GetFeatureGains(i)).Sum();
                                Compare(totGain1, totGain2);
                            }
                        }

                        if (datasets2 != null) datasets2.Dispose();
                        if (model != null) model.Dispose();
                    }
                }
                catch (Exception e)
                {
                    throw new Exception($"Failed: {Seed} #{test} {pms}", e);
                }
            }
        }

        [Fact]
        public void TrainMultiClass()
        {
            var rand = new Random(Seed);
            for (int test = 0; test < 5; ++test)
            {
                int numColumns = rand.Next(1, 10);
                var objective = (rand.Next(2) == 0) ? ObjectiveType.MultiClass : ObjectiveType.MultiClassOva;
                var pms = GenerateParameters(rand, objective, numColumns);

                Dictionary<int, int> categorical = null;
                var trainData = CreateRandomDenseClassifyData(rand, pms.Objective.NumClass, ref categorical, pms.Dataset.UseMissing, numColumns);
                var validData = (pms.Learning.EarlyStoppingRound > 0 || rand.Next(2) == 0) ? CreateRandomDenseClassifyData(rand, pms.Objective.NumClass, ref categorical, pms.Dataset.UseMissing, numColumns) : null;
                // The output cannot be monotone with respect to categorical features
                if (categorical != null && pms.Dataset.MonotoneConstraints.Length > 0)
                {
                    foreach (int i in categorical.Keys)
                        pms.Dataset.MonotoneConstraints[i] = 0;
                }
                pms.Dataset.CategoricalFeature = categorical.Keys.ToArray();

                var learningRateSchedule = (rand.Next(2) == 0) ? (Func<int, double>)null : (iter => pms.Learning.LearningRate * Math.Pow(0.99, iter));

                //if (test != 1) continue;

                try
                {
                    using (var datasets = (rand.Next(2) == 0) ? new Datasets(pms.Common, pms.Dataset, trainData, validData) :
                                                                new Datasets(pms.Common, pms.Dataset, Dense2Sparse(trainData), Dense2Sparse(validData)))
                    using (var trainer = new MulticlassTrainer(pms.Learning, pms.Objective))
                    {
                        //trainer.ToCommandLineFiles(datasets);

                        var model = trainer.Train(datasets, learningRateSchedule);
                        model.Managed.MaxThreads = rand.Next(1, Environment.ProcessorCount);

                        // possibly use subset of trees
                        var numIterations = -1;
                        if (rand.Next(2) == 0)
                        {
                            numIterations = rand.Next(1, model.Managed.MaxNumTrees);
                            model.Managed.MaxNumTrees = numIterations;
                            model.Native.MaxNumTrees = numIterations;
                        }

                        OvaPredictor model2 = null;
                        using (var ms = new System.IO.MemoryStream())
                        using (var writer = new System.IO.BinaryWriter(ms))
                        using (var reader = new System.IO.BinaryReader(ms))
                        {
                            PredictorPersist.Save(model.Managed, writer);
                            ms.Position = 0;
                            model2 = PredictorPersist.Load<double[]>(reader, false) as OvaPredictor;
                            Assert.Equal(ms.Position, ms.Length);
                        }

                        MulticlassNativePredictor model2native = null;
                        using (var ms = new System.IO.MemoryStream())
                        using (var writer = new System.IO.BinaryWriter(ms))
                        using (var reader = new System.IO.BinaryReader(ms))
                        {
                            NativePredictorPersist.Save(model.Native, writer);
                            ms.Position = 0;
                            model2native = NativePredictorPersist.Load<double[]>(reader) as MulticlassNativePredictor;
                            Assert.Equal(ms.Position, ms.Length);
                        }

                        var rawscore3s = trainer.Evaluate(Booster.PredictType.RawScore, trainData.Features, 0, numIterations);
                        Assert.Equal(trainData.Features.Length, rawscore3s.GetLength(0));
                        Assert.Equal(pms.Objective.NumClass, rawscore3s.GetLength(1));

                        var output3s = trainer.Evaluate(Booster.PredictType.Normal, trainData.Features, 0, numIterations);
                        Assert.Equal(trainData.Features.Length, output3s.GetLength(0));
                        Assert.Equal(pms.Objective.NumClass, output3s.GetLength(1));

                        var output3natives = model.Native.GetOutputs(trainData.Features, 0, numIterations);
                        Assert.Equal(trainData.Features.Length, output3natives.Length);

                        for (var irow = 0; irow < trainData.Features.Length; irow++)
                        {
                            var row = trainData.Features[irow];
                            // check evaluation of managed model
                            double[] output = null;
                            var input = new VBuffer<float>(row.Length, row);
                            model.Managed.GetOutput(ref input, ref output, 0, numIterations);
                            foreach (var p in output)
                            {
                                Assert.True(p >= 0);
                                Assert.True(p <= 1);
                            }
                            Assert.Equal(1, output.Sum(), 5);
                            Assert.Equal(output.Length, pms.Objective.NumClass);

                            // compare with output of serialised model
                            double[] output2 = null;
                            model2.GetOutput(ref input, ref output2, 0, numIterations);
                            Assert.Equal(output.Length, output2.Length);
                            for (var i = 0; i < output.Length; i++)
                                Compare(output[i], output2[i]);

                            // check raw scores against native booster object
                            var isRf = (pms.Learning.Boosting == BoostingType.RandomForest);
                            var rawscores = (model.Managed as OvaPredictor).Predictors.Select(p =>
                            {
                                double outputi = 0;
                                if (p is CalibratedPredictor)
                                    (p as CalibratedPredictor).SubPredictor.GetOutput(ref input, ref outputi, 0, numIterations);
                                else
                                    p.GetOutput(ref input, ref outputi, 0, numIterations);
                                return (outputi, p.MaxNumTrees);
                            }).ToArray();
                            var rawscores3 = trainer.Evaluate(Booster.PredictType.RawScore, row, 0, numIterations);
                            Assert.Equal(pms.Objective.NumClass, rawscores.Length);
                            Assert.Equal(pms.Objective.NumClass, rawscores3.Length);
                            for (var i = 0; i < rawscores.Length; i++)
                            {
                                (var rawscore, var numTrees) = rawscores[i];
                                Compare(isRf ? rawscore * numTrees : rawscore, rawscores3[i]);
                                Assert.Equal(rawscores3[i], rawscore3s[irow, i]);
                            }
                            //Console.WriteLine(trainer.GetModelString());
                            //throw new Exception($"Raw score mismatch at row {irow}: {rawscores[i]} vs {rawscores3[i]} (error: {Math.Abs(rawscores[i] - rawscores3[i])}) input: {String.Join(", ", row)}");

                            double [] outputNative = null;
                            model.Native.GetOutput(ref input, ref outputNative, 0, numIterations);

                            double[] outputNative2 = null;
                            model2native.GetOutput(ref input, ref outputNative2, 0, numIterations);

                            // check probabilities against native booster object
                            var output3 = trainer.Evaluate(Booster.PredictType.Normal, row, 0, numIterations);
                            for (var i = 0; i < output3.Length; i++)
                            {
                                Assert.Equal(output3s[irow, i], output3[i]);
                                Compare(output3natives[irow][i], output3[i]); // need to use Compare as models do not perfectly round-trip
                                Compare(outputNative[i], output3[i]);
                                Compare(outputNative2[i], output3[i]);  // need to use Compare as models do not perfectly round-trip
                            }

                            if (objective == ObjectiveType.MultiClassOva)
                            {
                                // booster object doesn't return normalised probabilities for OVA
                                var sum = output3.Sum();
                                for (var i = 0; i < output3.Length; i++)
                                    output3[i] /= sum;
                            }
                            Assert.Equal(pms.Objective.NumClass, output3.Length);
                            for (var i = 0; i < output3.Length; i++)
                                Assert.Equal(output[i], output3[i], 3);
                        }

                        var normalise = rand.Next(2) == 0;
                        var getSplits = rand.Next(2) == 0;
                        var gains = model.Managed.GetFeatureWeights(normalise, getSplits);
                        var gainsNative = model.Native.GetFeatureWeights(normalise, getSplits);
                        Assert.Equal(gains.Count, gainsNative.Count);
                        foreach (var kv in gains)
                        {
                            Assert.True(0 <= kv.Key && kv.Key < trainData.NumColumns);
                            Assert.True(0.0 <= kv.Value);
                            Compare(kv.Value, gainsNative[kv.Key]);
                        }

                        if (!getSplits && !normalise)
                        {
                            var totGain1 = gains.Values.Sum();
                            var totGain2 = Enumerable.Range(0, trainData.NumColumns).SelectMany(i => model.Managed.GetFeatureGains(i)).Sum();
                            Compare(totGain1, totGain2);
                        }
                    }
                }
                catch (Exception e)
                {
                    throw new Exception($"Failed: {Seed} #{test} {pms}", e);
                }
            }
        }

        protected static bool CompareVals(double a, double b)
        {
            double err = Math.Abs(a - b) / (1 + Math.Abs(a));
            return (a == b || err < 1e-10); // add equality case for infinity
        }

        [Fact]
        public void TrainRegression()
        {
            var objectiveTypes =
                new ObjectiveType[] {
                        ObjectiveType.Regression,
                        ObjectiveType.RegressionL1,
                        ObjectiveType.Huber,
                        ObjectiveType.Fair,
                        ObjectiveType.Poisson,
                        ObjectiveType.Quantile,
                        ObjectiveType.Mape,
                        ObjectiveType.Gamma,
                        ObjectiveType.Tweedie
                        };

            // Seed =272889208
            // ----LightGBMNet.Train.PInvokeException : Check failed: (best_split_info.left_count) > (0) at c:\lightgbm\src\treelearner\serial_tree_learner.cpp, line 653.
            var rand = new Random(Seed);
            for (int test = 0; test < 5; ++test)
            {
                int numColumns = rand.Next(1, 10);
                var objective = objectiveTypes[rand.Next(objectiveTypes.Length)];
                var pms = GenerateParameters(rand, objective, numColumns);
              //pms.Common.NumThreads = 1;  // uncomment to debug
                if (pms.Common.LinearTree)
                {
                    if (pms.Objective.Objective == ObjectiveType.RegressionL1)
                        pms.Objective.Objective = ObjectiveType.Regression; // L1 not a valid objective
                }
                if (rand.Next(2) == 0) pms.Objective.RegSqrt = true;

                var learningRateSchedule = (rand.Next(2) == 0) ? (Func<int, double>)null : (iter => pms.Learning.LearningRate * Math.Pow(0.99, iter));

                try
                {
                    Dictionary<int, int> categorical = null;
                    var trainData = CreateRandomDenseRegressionData(rand, ref categorical, pms.Dataset.UseMissing, numColumns);
                    var validData = (pms.Learning.EarlyStoppingRound > 0 || rand.Next(2) == 0) ? CreateRandomDenseRegressionData(rand, ref categorical, pms.Dataset.UseMissing, numColumns) : null;
                    // The output cannot be monotone with respect to categorical features
                    if (categorical != null && pms.Dataset.MonotoneConstraints.Length > 0)
                    {
                        foreach (int i in categorical.Keys)
                            pms.Dataset.MonotoneConstraints[i] = 0;
                    }
                    if (objective == ObjectiveType.Mape || objective == ObjectiveType.Quantile || objective == ObjectiveType.RegressionL1)
                            pms.Dataset.MonotoneConstraints = Array.Empty<int>();
                    pms.Dataset.CategoricalFeature = categorical.Keys.ToArray();

                    // make labels positive for certain objective types
                    if (objective == ObjectiveType.Poisson ||
                        objective == ObjectiveType.Gamma ||
                        objective == ObjectiveType.Tweedie)
                    {
                        for (var i = 0; i < trainData.Labels.Length; i++)
                            trainData.Labels[i] = Math.Abs(trainData.Labels[i]);

                        if (validData != null)
                        {
                            for (var i = 0; i < validData.Labels.Length; i++)
                                validData.Labels[i] = Math.Abs(validData.Labels[i]);
                        }
                    }

                    // uncomment to select particular iteration
                    //if (test != 3)
                    //    continue;

                    using (var datasets = (rand.Next(2) == 0) ? new Datasets(pms.Common, pms.Dataset, trainData, validData) :
                                                                new Datasets(pms.Common, pms.Dataset, Dense2Sparse(trainData), Dense2Sparse(validData)))
                    using (var trainer = new RegressionTrainer(pms.Learning, pms.Objective))
                    {
                        //if (true)
                        //    trainer.ToCommandLineFiles(datasets);

                        var model = trainer.Train(datasets, learningRateSchedule);
                        model.Managed.MaxThreads = rand.Next(1, Environment.ProcessorCount);

                        // possibly use subset of trees
                        var numIterations = -1;
                        if (rand.Next(2) == 0)
                        {
                            numIterations = rand.Next(1, model.Managed.MaxNumTrees);
                            model.Managed.MaxNumTrees = numIterations;
                            model.Native.MaxNumTrees = numIterations;
                        }

                        IPredictorWithFeatureWeights<double> model2 = null;
                        using (var ms = new System.IO.MemoryStream())
                        using (var writer = new System.IO.BinaryWriter(ms))
                        using (var reader = new System.IO.BinaryReader(ms))
                        {
                            PredictorPersist.Save(model.Managed, writer);
                            ms.Position = 0;
                            model2 = PredictorPersist.Load<double>(reader, false);
                            Assert.Equal(ms.Position, ms.Length);
                        }

                        IPredictorWithFeatureWeights<double> model2native = null;
                        using (var ms = new System.IO.MemoryStream())
                        using (var writer = new System.IO.BinaryWriter(ms))
                        using (var reader = new System.IO.BinaryReader(ms))
                        {
                            NativePredictorPersist.Save(model.Native, writer);
                            ms.Position = 0;
                            model2native = NativePredictorPersist.Load<double>(reader);
                            Assert.Equal(ms.Position, ms.Length);
                        }

                        var output3s = trainer.Evaluate(Booster.PredictType.Normal, trainData.Features, 0, numIterations);
                        Assert.Equal(trainData.Features.Length, output3s.GetLength(0));
                        Assert.Equal(1, output3s.GetLength(1));

                        var output3natives = model.Native.GetOutputs(trainData.Features, 0, numIterations);
                        Assert.Equal(trainData.Features.Length, output3s.Length);

                        for (int i = 0; i < trainData.Features.Length; i++)
                        {
                            var row = trainData.Features[i];

                            double output = 0;
                            var input = new VBuffer<float>(row.Length, row);
                            model.Managed.GetOutput(ref input, ref output, 0, numIterations);
                            Assert.False(double.IsNaN(output));

                            double output2 = 0;
                            model2.GetOutput(ref input, ref output2, 0, numIterations);
                            Assert.Equal(output, output2);

                            var output3 = trainer.Evaluate(Booster.PredictType.Normal, row, 0, numIterations);
                            Assert.Single(output3);
                            Assert.Equal(output3[0], output3s[i, 0]);
                            
                            Assert.Equal(output3[0], output3natives[i]);
                          //if (!ok)
                          //{
                          //    for(var j=0; j<model.Managed.MaxNumTrees; j++)
                          //    {
                          //        var output3j = trainer.Evaluate(Booster.PredictType.Normal, row, j, 1);
                          //        var output3nativesj = model.Native.GetOutputs(new float[][] { row }, j, 1);
                          //        ok = CompareVals(output3j[0], output3nativesj[0]);
                          //        if (!ok)
                          //            Console.WriteLine($"Mismatch at tree {j}, {output3j[0]} vs {output3nativesj[i]}, diff {output3j[0] - output3nativesj[i]}");
                          //    }
                          //    Console.WriteLine();
                          //}
                          //Assert.True(ok);

                            var ok = CompareVals(output, output3[0]);
                            if (!ok)
                            {
                                for (var j = 0; j < model.Managed.MaxNumTrees; j++)
                                {
                                    double outputj = 0;
                                    model.Managed.GetOutput(ref input, ref outputj, j, 1);
                                    var output3j = trainer.Evaluate(Booster.PredictType.Normal, row, j, 1);
                                    ok = CompareVals(outputj, output3j[0]);
                                    if (!ok)
                                        Console.WriteLine($"Mismatch at tree {j}, {outputj} vs {output3j[0]}, diff {outputj- output3j[0]}");
                                }
                                Console.WriteLine();
                            }
                            Assert.True(ok);

                            double outputNative = 0;
                            model.Native.GetOutput(ref input, ref outputNative, 0, numIterations);
                            Assert.True(CompareVals(outputNative, output3[0])); // have to use CompareVals as model does not round-trip doubles

                            model2native.GetOutput(ref input, ref outputNative, 0, numIterations);
                            Assert.True(CompareVals(outputNative, output3[0])); // have to use CompareVals as model does not round-trip doubles
                        }

                        var normalise = rand.Next(2) == 0;
                        var getSplits = rand.Next(2) == 0;
                        var gains = model.Managed.GetFeatureWeights(normalise, getSplits);
                        var gainsNative = model.Native.GetFeatureWeights(normalise, getSplits);
                        Assert.Equal(gains.Count, gainsNative.Count);
                        foreach (var kv in gains)
                        {
                            Assert.True(0 <= kv.Key && kv.Key < trainData.NumColumns);
                            Assert.True(0.0 <= kv.Value);
                            Compare(kv.Value, gainsNative[kv.Key]);
                        }

                        if (!getSplits && !normalise)
                        {
                            var totGain1 = gains.Values.Sum();
                            var totGain2 = Enumerable.Range(0, trainData.NumColumns).SelectMany(i => model.Managed.GetFeatureGains(i)).Sum();
                            Compare(totGain1, totGain2);
                        }
                    }
                }
                catch (Exception e)
                {
                    throw new Exception($"Failed: {Seed} #{test} {pms}", e);
                }
            }
        }

        public static int [] GenGroups(Random random, int len)
        {
            var groups = new List<int>();
            var count = 0;
            for (int i = 0; i < len; i++)
            {
                count++;
                if (random.Next(10) == 0)
                {
                    groups.Add(count);
                    count = 0;
                }
            }
            if (count > 0)
                groups.Add(count);
            Assert.Equal(len, groups.Sum());
            return groups.ToArray();
        }

        [Fact]
        public void TrainRanking()
        {
            var rand = new Random(Seed);
            for (int test = 0; test < 5; ++test)
            {
                int numColumns = rand.Next(1, 10);
                var pms = GenerateParameters(rand, ObjectiveType.LambdaRank, numColumns);
                pms.Objective.EvalAt = new int[] { 5 };    // TODO: need at most one or get 'Expected at most one metric' error
                var numRanks = rand.Next(2, 4);

                Dictionary<int, int> categorical = null;
                var trainData = CreateRandomDenseClassifyData(rand, numRanks, ref categorical, pms.Dataset.UseMissing, numColumns);
                trainData.Groups = GenGroups(rand, trainData.NumRows);
                var validData = (pms.Learning.EarlyStoppingRound > 0 || rand.Next(2) == 0) ? CreateRandomDenseClassifyData(rand, numRanks, ref categorical, pms.Dataset.UseMissing, numColumns) : null;
                if (validData != null) validData.Groups = GenGroups(rand, validData.NumRows);
                // The output cannot be monotone with respect to categorical features
                if (categorical != null && pms.Dataset.MonotoneConstraints.Length > 0)
                {
                    foreach (int i in categorical.Keys)
                        pms.Dataset.MonotoneConstraints[i] = 0;
                }
                pms.Dataset.CategoricalFeature = categorical.Keys.ToArray();

                var learningRateSchedule = (rand.Next(2) == 0) ? (Func<int, double>)null : (iter => pms.Learning.LearningRate * Math.Pow(0.99, iter));

                try
                {
                    using (var datasets = (rand.Next(2) == 0) ? new Datasets(pms.Common, pms.Dataset, trainData, validData) :
                                                                new Datasets(pms.Common, pms.Dataset, Dense2Sparse(trainData), Dense2Sparse(validData)))
                    using (var trainer = new RankingTrainer(pms.Learning, pms.Objective))
                    {
                        var model = trainer.Train(datasets, learningRateSchedule);
                        model.Managed.MaxThreads = rand.Next(1, Environment.ProcessorCount);

                        // possibly use subset of trees
                        var numIterations = -1;
                        if (rand.Next(2) == 0)
                        {
                            numIterations = rand.Next(1, model.Managed.MaxNumTrees);
                            model.Managed.MaxNumTrees = numIterations;
                            model.Native.MaxNumTrees = numIterations;
                        }

                        RankingPredictor model2 = null;
                        using (var ms = new System.IO.MemoryStream())
                        using (var writer = new System.IO.BinaryWriter(ms))
                        using (var reader = new System.IO.BinaryReader(ms))
                        {
                            PredictorPersist.Save(model.Managed, writer);
                            ms.Position = 0;
                            model2 = PredictorPersist.Load<double>(reader, false) as RankingPredictor;
                            Assert.Equal(ms.Position, ms.Length);
                        }

                        RankingNativePredictor model2native = null;
                        using (var ms = new System.IO.MemoryStream())
                        using (var writer = new System.IO.BinaryWriter(ms))
                        using (var reader = new System.IO.BinaryReader(ms))
                        {
                            NativePredictorPersist.Save(model.Native, writer);
                            ms.Position = 0;
                            model2native = NativePredictorPersist.Load<double>(reader) as RankingNativePredictor;
                            Assert.Equal(ms.Position, ms.Length);
                        }

                        var output3s = trainer.Evaluate(Booster.PredictType.Normal, trainData.Features, 0, numIterations);
                        Assert.Equal(trainData.Features.Length, output3s.GetLength(0));
                        Assert.Equal(1, output3s.GetLength(1));

                        var output3natives = model.Native.GetOutputs(trainData.Features, 0, numIterations);
                        Assert.Equal(trainData.Features.Length, output3s.Length);

                        for (int i = 0; i < trainData.Features.Length; i++)
                        {
                            var row = trainData.Features[i];

                            double output = 0;
                            var input = new VBuffer<float>(row.Length, row);
                            model.Managed.GetOutput(ref input, ref output, 0, numIterations);
                            // TODO: NFI what output represents...

                            double output2 = 0;
                            model2.GetOutput(ref input, ref output2, 0, numIterations);
                            Compare(output, output2);

                            var output3 = trainer.Evaluate(Booster.PredictType.Normal, row, 0, numIterations);
                            Assert.Single(output3);
                            Assert.Equal(output3[0], output3s[i, 0]);
                            Compare(output3[0], output3natives[i]);
                            Compare(output, output3[0]);
                            //Console.WriteLine(trainer.GetModelString());
                            //throw new Exception($"Output mismatch {output} vs {output3[0]} (error: {Math.Abs(output - output3[0])}) input: {String.Join(", ", row)}");
                        }

                        var normalise = rand.Next(2) == 0;
                        var getSplits = rand.Next(2) == 0;
                        var gains = model.Managed.GetFeatureWeights(normalise, getSplits);
                        var gainsNative = model.Native.GetFeatureWeights(normalise, getSplits);
                        Assert.Equal(gains.Count, gainsNative.Count);
                        foreach (var kv in gains)
                        {
                            Assert.True(0 <= kv.Key && kv.Key < trainData.NumColumns);
                            Assert.True(0.0 <= kv.Value);
                            Compare(kv.Value, gainsNative[kv.Key]);
                        }

                        if (!getSplits && !normalise)
                        {
                            var totGain1 = gains.Values.Sum();
                            var totGain2 = Enumerable.Range(0, trainData.NumColumns).SelectMany(i => model.Managed.GetFeatureGains(i)).Sum();
                            Compare(totGain1, totGain2);
                        }
                    }
                }
                catch (Exception e)
                {
                    throw new Exception($"Failed: {Seed} #{test} {pms}", e);
                }
            }
        }

        //[Fact]
        protected void BenchmarkBinary()
        {
            var rand = new Random(Seed);
            for (int test = 0; test < 3; ++test)
            {
                for (int gpu = 0; gpu < 2; gpu++)
                {
                    int numColumns = 50 * (test + 1);
                    var pms = new Parameters();
                    pms.Objective.Objective = ObjectiveType.Binary;
                    pms.Dataset.MaxBin = 63;
                    pms.Learning.BaggingFraction = 1;
                    pms.Learning.BaggingFreq = 1;
                    pms.Learning.LearningRate = 1e-3;
                    pms.Learning.NumIterations = 10;
                    pms.Common.DeviceType = (gpu > 0) ? DeviceType.GPU : DeviceType.CPU;

                    var categorical = new Dictionary<int, int>();   // i.e., no cat
                    var trainData = CreateRandomDenseClassifyData(rand, 2, ref categorical, pms.Dataset.UseMissing, numColumns);
                    DataDense validData = null;
                    pms.Dataset.CategoricalFeature = categorical.Keys.ToArray();

                    try
                    {
                        using (var datasets = new Datasets(pms.Common, pms.Dataset, trainData, validData))
                        using (var trainer = new BinaryTrainer(pms.Learning, pms.Objective))
                        {
                            var timer = System.Diagnostics.Stopwatch.StartNew();
                            var model = trainer.Train(datasets);
                            var elapsed = timer.Elapsed;
                            output.WriteLine($"{pms.Common.DeviceType}: NumRows={trainData.NumRows} NumCols={numColumns} MaxNumTrees={model.Managed.MaxNumTrees} TrainTimeSecs={elapsed.TotalSeconds}");
                        }
                    }
                    catch (Exception e)
                    {
                        throw new Exception($"Failed: {Seed} #{test} {pms}", e);
                    }
                }
            }
        }

        [Fact]
        public void BenchmarkEval()
        {
            var rand = new Random(Seed);
            int numColumns = 100;
            var pms = new Parameters();
            pms.Objective.Objective = ObjectiveType.Binary;
            pms.Dataset.MaxBin = 63;
            pms.Learning.LearningRate = 1e-3;
            pms.Learning.NumIterations = 1000;
            pms.Common.DeviceType = DeviceType.CPU;

            var categorical = new Dictionary<int, int>();   // i.e., no cat
            var trainData = CreateRandomDenseClassifyData(rand, 2, ref categorical, pms.Dataset.UseMissing, numColumns);
            DataDense validData = null;
            pms.Dataset.CategoricalFeature = categorical.Keys.ToArray();

            using (var datasets = new Datasets(pms.Common, pms.Dataset, trainData, validData))
            using (var trainer = new BinaryTrainer(pms.Learning, pms.Objective))
            {
                var model = trainer.Train(datasets);
                output.WriteLine($"MaxNumTrees={model.Managed.MaxNumTrees}");

                var timer = System.Diagnostics.Stopwatch.StartNew();
                model.Native.GetOutputs(trainData.Features, 0, -1);
                var elapsed1 = timer.Elapsed;
                output.WriteLine($"EvalNativeMulti={elapsed1.TotalMilliseconds}");

                timer.Restart();
                foreach (var row in trainData.Features)
                    trainer.Evaluate(Booster.PredictType.Normal, row, 0, -1);
                var elapsed2 = timer.Elapsed;
                output.WriteLine($"EvalNativeSingle={elapsed2.TotalMilliseconds}");

                foreach (var maxThreads in new int[] { 1, 2, 4, 8, 16, 32, Environment.ProcessorCount })
                {
                    model.Managed.MaxThreads = maxThreads;
                    timer.Restart();
                    foreach (var row in trainData.Features)
                    {
                        double output = 0;
                        var input = new VBuffer<float>(row.Length, row);
                        model.Managed.GetOutput(ref input, ref output, 0, -1);
                    }
                    var elapsed3 = timer.Elapsed;
                    output.WriteLine($"MaxThreads={maxThreads} EvalManaged={elapsed3.TotalMilliseconds}");
                }
            }
        }

    }
}
