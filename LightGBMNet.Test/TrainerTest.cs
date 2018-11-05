using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using LightGBMNet.Train;
using LightGBMNet.Tree;

namespace LightGBMNet.Train.Test
{
    public class TrainerTest
    {
        private static BoostingType [] boostingTypes =
            new BoostingType[] {
                BoostingType.GBDT,
              //BoostingType.RandomForest,  // TODO: wtf
                BoostingType.Dart,
                BoostingType.Goss
            };

        private static Parameters GenerateParameters(Random rand, ObjectiveType objective, int numColumns)
        {
            var pms = new Parameters();
            pms.Learning.NumIterations = rand.Next(1, 100);
            pms.Common.Verbosity = VerbosityType.Error;

            pms.Learning.Objective = objective;
            pms.Learning.Boosting = boostingTypes[rand.Next(boostingTypes.Length)];

            if (pms.Learning.Boosting == BoostingType.RandomForest)
            {
                pms.Learning.BaggingFreq = rand.Next(1, 10);
                pms.Learning.BaggingFraction = rand.Next(1, 99) / 100.0;
                pms.Learning.FeatureFraction = rand.Next(1, 99) / 100.0;
            }

            if (objective == ObjectiveType.MultiClass || objective == ObjectiveType.MultiClassOva)
                pms.Objective.NumClass = rand.Next(2, 4);

            if (objective == ObjectiveType.Binary || objective == ObjectiveType.MultiClassOva || objective == ObjectiveType.LambdaRank)
                if (rand.Next(2) == 0) pms.Objective.Sigmoid = rand.Next(1,100) / 100.0;

            if (rand.Next(2) == 0) pms.Dataset.MaxBin = 64;
            if (rand.Next(2) == 0) pms.Dataset.MinDataInBin = rand.Next(1,10);
            if (rand.Next(2) == 0) pms.Dataset.BinConstructSampleCnt = rand.Next(100, 1000);
            pms.Dataset.EnableBundle = (rand.Next(2) == 0);
            pms.Dataset.IsEnableSparse = (rand.Next(2) == 0);
            pms.Dataset.UseMissing = (rand.Next(2) == 0);
            if (rand.Next(2) == 0) pms.Dataset.SparseThreshold = rand.Next(1, 100) / 100.0;
            if (rand.Next(2) == 0) pms.Dataset.MaxConflictRate = rand.Next(0, 100) / 100.0;
            if (rand.Next(2) == 0) pms.Dataset.MinDataInLeaf = rand.Next(1, 20);
            if (rand.Next(2) == 0) pms.Dataset.DataRandomSeed = rand.Next(1, 20);
            if (rand.Next(2) == 0) pms.Dataset.MonotoneConstraints = Enumerable.Range(0, numColumns).Select(x => rand.Next(2)-1).ToArray();
            if (rand.Next(2) == 0) pms.Dataset.FeatureContri = Enumerable.Range(0, numColumns).Select(x => rand.Next(1,100)/100.0).ToArray();
            if (rand.Next(2) == 0) pms.Learning.EarlyStoppingRound = rand.Next(1, 20);

            return pms;
        }


        public static DataDense CreateRandomDenseData(
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
                            categorical.Add(j, rand.Next(3,100));
                    }
                }
            }

            var rows = new float[numRows][];
            var weights = (rand.Next(2) == 0) ? new float[numRows] : null;
            for (int i = 0; i < numRows; ++i)
            {
                var row = new float[numColumns];
                for (int j = 0; j < row.Length; ++j)
                {
                    if (useMissing && rand.Next(50) == 0)
                    {
                        row[j] = Single.NaN;
                    }
                    else
                    {
                        if (categorical.TryGetValue(j, out int numClass))
                            row[j] = rand.Next(numClass);
                        else
                            row[j] = (rand.Next(100) == 0) ? 0.0f : (float)(rand.NextDouble() - 0.5);
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

        public static DataDense CreateRandomDenseClassifyData(
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

        public static DataDense CreateRandomDenseRegressionData(
            Random rand,
            ref Dictionary<int, int> categorical,
            bool useMissing,
            int numColumns = -1
            )
        {
            var rslt = CreateRandomDenseData(rand, ref categorical, useMissing, numColumns);

            var labels = new float[rslt.NumRows];
            for (int i = 0; i < labels.Length; ++i)
                labels[i] = (float)(rand.NextDouble() - 0.5);

            rslt.Labels = labels;
            rslt.Validate();
            return rslt;
        }

#if false
        private static unsafe float NextDown(float x)
        {
            uint temp = *(uint*)&x;
            temp--;
            return *(float*)&temp;
        }

        private static unsafe float NextUp(float x)
        {
            uint temp = *(uint*)&x;
            temp++;
            return *(float*)&temp;
        }

        [Fact]
        public unsafe void RemoveMe()
        {
            float secondBiggest = NextDown(Single.MaxValue);
            float secondSmallest = NextDown(Single.MinValue);

            float secondBiggest2 = 0;
            float secondSmallest2 = 0;

            uint temp = 0x7f7f_fffe;
            secondBiggest2 = *(float*)&temp;

            temp = 0xff7f_fffe;
            secondSmallest2 = *(float*)&temp;

            //temp = 0x0000_0000;
            ////temp = 0x8000_0000;    // get nan and a negative
            //float zero = *(float*)&temp;
            //Assert.True(zero == 0.0);
            //var up = NextUp(zero);
            //var dn = NextDown(zero);
            
            Assert.True(secondBiggest > 0);
            Assert.True(secondBiggest < Single.MaxValue);
            Assert.Equal(secondBiggest, secondBiggest2);

            Assert.True(secondSmallest < 0);
            Assert.True(secondSmallest > Single.MinValue);
            Assert.Equal(secondSmallest, secondSmallest2);
        }
#endif

        private static readonly int Seed = (new Random()).Next();

        [Fact]
        public void TrainBinary()
        {
            var rand = new Random(Seed);
            for (int test = 0; test < 5; ++test)
            {
                int numColumns = rand.Next(1, 10);
                var pms = GenerateParameters(rand, ObjectiveType.Binary, numColumns);
                Dictionary<int, int> categorical = null;
                var trainData = CreateRandomDenseClassifyData(rand, 2, ref categorical, pms.Dataset.UseMissing, numColumns);
                var validData = (pms.Learning.EarlyStoppingRound > 0 || rand.Next(2) == 0) ? CreateRandomDenseClassifyData(rand, 2, ref categorical, pms.Dataset.UseMissing, numColumns) : null;
                pms.Dataset.CategoricalFeature = categorical.Keys.ToArray();

                try
                {
                    using (var datasets = new Datasets(pms.Common, pms.Dataset, trainData, validData))
                    using (var trainer = new BinaryTrainer(pms.Learning, pms.Objective, pms.Metric))
                    {
                        var model = trainer.Train(datasets);

                        CalibratedPredictor model2 = null;
                        using (var ms = new System.IO.MemoryStream())
                        using (var writer = new System.IO.BinaryWriter(ms))
                        using (var reader = new System.IO.BinaryReader(ms))
                        {
                            PredictorPersist.Save(model, writer);
                            ms.Position = 0;
                            model2 = PredictorPersist.Load<double>(reader) as CalibratedPredictor;
                            Assert.Equal(ms.Position, ms.Length);
                        }

                        foreach (var row in trainData.Features)
                        {
                            double output = 0;
                            var input = new VBuffer<float>(row.Length, row);
                            model.GetOutput(ref input, ref output);
                            Assert.True(output >= 0);
                            Assert.True(output <= 1);

                            double output2 = 0;
                            model2.GetOutput(ref input, ref output2);
                            Assert.Equal(output, output2);

                            var output3 = trainer.Evaluate(Booster.PredictType.Normal, row);
                            Assert.Single(output3);
                            if (Math.Abs(output - output3[0]) / (1 + Math.Abs(output)) > 1e-6)
                            {
                                Console.WriteLine(trainer.GetModelString());
                                throw new Exception($"Output mismatch {output} vs {output3[0]} (error: {Math.Abs(output - output3[0])}) input: {String.Join(", ", row)}");
                            }
                        }

                        var gains = model.GetFeatureWeights();
                        foreach(var kv in gains)
                        {
                            Assert.True(0 <= kv.Key && kv.Key < trainData.NumColumns);
                            Assert.True(0.0 <= kv.Value);
                        }
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
                pms.Dataset.CategoricalFeature = categorical.Keys.ToArray();

                //if (test != 1) continue;

                try
                {
                    using (var datasets = new Datasets(pms.Common, pms.Dataset, trainData, validData))
                    using (var trainer = new MulticlassTrainer(pms.Learning, pms.Objective, pms.Metric))
                    {
                        //trainer.ToCommandLineFiles(datasets);

                        var model = trainer.Train(datasets);

                        OvaPredictor model2 = null;
                        using (var ms = new System.IO.MemoryStream())
                        using (var writer = new System.IO.BinaryWriter(ms))
                        using (var reader = new System.IO.BinaryReader(ms))
                        {
                            PredictorPersist.Save(model, writer);
                            ms.Position = 0;
                            model2 = PredictorPersist.Load<VBuffer<double>>(reader) as OvaPredictor;
                            Assert.Equal(ms.Position, ms.Length);
                        }

                        for (var irow=0; irow < trainData.Features.Length; irow++)
                        {
                            var row = trainData.Features[irow];
                            // check evaluation of managed model
                            VBuffer<double> output = default;
                            var input = new VBuffer<float>(row.Length, row);
                            model.GetOutput(ref input, ref output);
                            foreach (var p in output.Values)
                            {
                                Assert.True(p >= 0);
                                Assert.True(p <= 1);
                            }
                            Assert.Equal(1, output.Values.Sum(), 5);
                            Assert.Equal(output.Values.Length, pms.Objective.NumClass);

                            // compare with output of serialised model
                            VBuffer<double> output2 = default;
                            model2.GetOutput(ref input, ref output2);
                            Assert.Equal(output.Count, output2.Count);
                            Assert.Equal(output.Length, output2.Length);
                            Assert.Equal(output.Values, output2.Values);
                            Assert.Equal(output.Indices, output2.Indices);

                            // check raw scores against native booster object
                            var rawscores = (model as OvaPredictor).Predictors.Select(p => 
                            {
                                double outputi = 0;
                                if (p is CalibratedPredictor)
                                    (p as CalibratedPredictor).SubPredictor.GetOutput(ref input, ref outputi);
                                else
                                    p.GetOutput(ref input, ref outputi);
                                return outputi;
                            }).ToArray();
                            var rawscores3 = trainer.Evaluate(Booster.PredictType.RawScore, row);
                            Assert.Equal(pms.Objective.NumClass, rawscores.Length);
                            Assert.Equal(pms.Objective.NumClass, rawscores3.Length);
                            for (var i = 0; i < rawscores.Length; i++)
                                if (Math.Abs(rawscores[i] - rawscores3[i]) / (1 + Math.Abs(rawscores[i])) > 1e-6)
                                {
                                    Console.WriteLine(trainer.GetModelString());
                                    throw new Exception($"Raw score mismatch at row {irow}: {rawscores[i]} vs {rawscores3[i]} (error: {Math.Abs(rawscores[i] - rawscores3[i])}) input: {String.Join(", ", row)}");
                                }

                            // check probabilities against native booster object
                            var output3 = trainer.Evaluate(Booster.PredictType.Normal, row);
                            if (objective == ObjectiveType.MultiClassOva)
                            {
                                // booster object doesn't return normalised probabilities for OVA
                                var sum = output3.Sum();
                                for (var i = 0; i < output3.Length; i++)
                                    output3[i] /= sum;
                            }
                            Assert.Equal(pms.Objective.NumClass, output3.Length);
                            for (var i = 0; i < output3.Length; i++)
                                Assert.Equal(output.Values[i], output3[i], 3);
                        }

                        var gains = model.GetFeatureWeights();
                        foreach (var kv in gains)
                        {
                            Assert.True(0 <= kv.Key && kv.Key < trainData.NumColumns);
                            Assert.True(0.0 <= kv.Value);
                        }
                    }
                }
                catch (Exception e)
                {
                    throw new Exception($"Failed: {Seed} #{test} {pms}", e);
                }
            }
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

            var rand = new Random(Seed);
            for (int test = 0; test < 5; ++test)
            {
                int numColumns = rand.Next(1, 10);
                var objective = objectiveTypes[rand.Next(objectiveTypes.Length)];
                var pms = GenerateParameters(rand, objective, numColumns);
                if (rand.Next(2) == 0) pms.Objective.RegSqrt = true;

                try
                {
                    Dictionary<int, int> categorical = null;
                    var trainData = CreateRandomDenseRegressionData(rand, ref categorical, pms.Dataset.UseMissing, numColumns);
                    var validData = (pms.Learning.EarlyStoppingRound > 0 || rand.Next(2) == 0) ? CreateRandomDenseRegressionData(rand, ref categorical, pms.Dataset.UseMissing, numColumns) : null;
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

                    using (var datasets = new Datasets(pms.Common, pms.Dataset, trainData, validData))
                    using (var trainer = new RegressionTrainer(pms.Learning, pms.Objective, pms.Metric))
                    {
                        if (true)
                            trainer.ToCommandLineFiles(datasets);

                        var model = trainer.Train(datasets);

                        IPredictorWithFeatureWeights<double> model2 = null;
                        using (var ms = new System.IO.MemoryStream())
                        using (var writer = new System.IO.BinaryWriter(ms))
                        using (var reader = new System.IO.BinaryReader(ms))
                        {
                            PredictorPersist.Save(model, writer);
                            ms.Position = 0;
                            model2 = PredictorPersist.Load<double>(reader);
                            Assert.Equal(ms.Position, ms.Length);
                        }

                        foreach (var row in trainData.Features)
                        {
                            double output = 0;
                            var input = new VBuffer<float>(row.Length, row);
                            model.GetOutput(ref input, ref output);
                            Assert.False(Double.IsNaN(output));

                            double output2 = 0;
                            model2.GetOutput(ref input, ref output2);
                            Assert.Equal(output, output2);

                            var output3 = trainer.Evaluate(Booster.PredictType.Normal, row);
                            Assert.Single(output3);
                            if (Math.Abs(output - output3[0]) / (1 + Math.Abs(output)) > 1e-6)
                            {
                                Console.WriteLine(trainer.GetModelString());
                                throw new Exception($"Output mismatch {output} vs {output3[0]} (error: {Math.Abs(output - output3[0])}) input: {String.Join(", ", row)}");
                            }
                        }

                        var gains = model.GetFeatureWeights();
                        foreach (var kv in gains)
                        {
                            Assert.True(0 <= kv.Key && kv.Key < trainData.NumColumns);
                            Assert.True(0.0 <= kv.Value);
                        }
                    }
                }
                catch (Exception e)
                {
                    throw new Exception($"Failed: {Seed} #{test} {pms}", e);
                }
            }
        }

        private int [] GenGroups(Random random, int len)
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
                pms.Metric.EvalAt = new int[] { 5 };    // TODO: need at most one or get 'Expected at most one metric' error
                var numRanks = rand.Next(2, 4);

                Dictionary<int, int> categorical = null;
                var trainData = CreateRandomDenseClassifyData(rand, numRanks, ref categorical, pms.Dataset.UseMissing, numColumns);
                trainData.Groups = GenGroups(rand, trainData.NumRows);
                var validData = (pms.Learning.EarlyStoppingRound > 0 || rand.Next(2) == 0) ? CreateRandomDenseClassifyData(rand, numRanks, ref categorical, pms.Dataset.UseMissing, numColumns) : null;
                if (validData != null) validData.Groups = GenGroups(rand, validData.NumRows);
                pms.Dataset.CategoricalFeature = categorical.Keys.ToArray();

                try
                {
                    using (var datasets = new Datasets(pms.Common, pms.Dataset, trainData, validData))
                    using (var trainer = new RankingTrainer(pms.Learning, pms.Objective, pms.Metric))
                    {
                        var model = trainer.Train(datasets);

                        RankingPredictor model2 = null;
                        using (var ms = new System.IO.MemoryStream())
                        using (var writer = new System.IO.BinaryWriter(ms))
                        using (var reader = new System.IO.BinaryReader(ms))
                        {
                            PredictorPersist.Save(model, writer);
                            ms.Position = 0;
                            model2 = PredictorPersist.Load<double>(reader) as RankingPredictor;
                            Assert.Equal(ms.Position, ms.Length);
                        }

                        foreach (var row in trainData.Features)
                        {
                            double output = 0;
                            var input = new VBuffer<float>(row.Length, row);
                            model.GetOutput(ref input, ref output);
                            // TODO: NFI what output represents...

                            double output2 = 0;
                            model2.GetOutput(ref input, ref output2);
                            Assert.Equal(output, output2);

                            var output3 = trainer.Evaluate(Booster.PredictType.Normal, row);
                            Assert.Single(output3);
                            if (Math.Abs(output - output3[0]) / (1 + Math.Abs(output)) > 1e-6)
                            {
                                Console.WriteLine(trainer.GetModelString());
                                throw new Exception($"Output mismatch {output} vs {output3[0]} (error: {Math.Abs(output - output3[0])}) input: {String.Join(", ", row)}");
                            }
                        }

                        var gains = model.GetFeatureWeights();
                        foreach (var kv in gains)
                        {
                            Assert.True(0 <= kv.Key && kv.Key < trainData.NumColumns);
                            Assert.True(0.0 <= kv.Value);
                        }
                    }
                }
                catch (Exception e)
                {
                    throw new Exception($"Failed: {Seed} #{test} {pms}", e);
                }
            }
        }
    }
}
