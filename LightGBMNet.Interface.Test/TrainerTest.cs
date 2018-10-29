using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using LightGBMNet.Training;
using LightGBMNet.FastTree;

namespace LightGBMNet.Interface.Test
{
    public class TrainerTest
    {

        public static DataDense CreateRandomDenseClassifyData(System.Random rand, int numClasses, int numColumns = -1)
        {
            var numRows = rand.Next(100, 500);
            if (numColumns == -1)
                numColumns = rand.Next(1, 10);

            var rows = new float[numRows][];
            var labels = new float[numRows];
            var weights = (rand.Next(2) == 0) ? new float[numRows] : null;
            for (int i = 0; i < numRows; ++i)
            {
                var row = new float[numColumns];
                for (int j = 0; j < row.Length; ++j)
                    row[j] = (float) (rand.NextDouble() - 0.5);
                rows[i] = row;
                labels[i] = rand.Next(numClasses);
                if (weights != null) weights[i] = (float) rand.NextDouble();
            }

            var rslt = new DataDense();
            rslt.Features = rows;
            rslt.Labels = labels;
            rslt.Weights = weights;
            rslt.Groups = null;
            rslt.Validate();
            return rslt;
        }

        public static DataDense CreateRandomDenseRegressionData(System.Random rand, int numColumns = -1)
        {
            var numRows = rand.Next(100, 500);
            if (numColumns == -1)
                numColumns = rand.Next(1, 10);

            var rows = new float[numRows][];
            var labels = new float[numRows];
            var weights = (rand.Next(2) == 0) ? new float[numRows] : null;
            for (int i = 0; i < numRows; ++i)
            {
                var row = new float[numColumns];
                for (int j = 0; j < row.Length; ++j)
                    row[j] = (float)(rand.NextDouble() - 0.5);
                rows[i] = row;
                labels[i] = (float)(rand.NextDouble() - 0.5);
                if (weights != null) weights[i] = (float)rand.NextDouble();
            }

            var rslt = new DataDense();
            rslt.Features = rows;
            rslt.Labels = labels;
            rslt.Weights = weights;
            rslt.Groups = null;
            rslt.Validate();
            return rslt;
        }

        [Fact]
        public void TrainBinary()
        {
            var rand = new System.Random();
            for (int test = 0; test < 5; ++test)
            {
                var pms = new Parameters();
                pms.Core.Objective = ObjectiveType.Binary;
                pms.Core.NumIterations = rand.Next(1, 100);
                pms.Common.Verbosity = VerbosityType.Error;
                var trainer = new BinaryTrainer(pms);
                var trainData = CreateRandomDenseClassifyData(rand, 2);
                var validData = (rand.Next(2) == 0) ? CreateRandomDenseClassifyData(rand, 2, trainData.NumColumns) : null;
                var model = trainer.Train(trainData, validData);

                // TODO
                //using (var ms = new System.IO.MemoryStream())
                //{
                //    using (var writer = new System.IO.BinaryWriter(ms))
                //    {
                //        model.
                //    }
                //}

                foreach (var row in trainData.Features)
                {
                    float output = 0;
                    var input = new VBuffer<float>(row.Length, row);
                    model.GetOutput(ref input, ref output);
                    Assert.True(output >= 0);
                    Assert.True(output <= 1);
                }
            }
        }

        [Fact]
        public void TrainMultiClass()
        {
            var rand = new System.Random();
            for (int test = 0; test < 5; ++test)
            {
                var pms = new Parameters();
                //pms.Core.Objective = (rand.Next(2) == 0) ? ObjectiveType.MultiClass : ObjectiveType.MultiClassOva;
                pms.Core.Objective = ObjectiveType.MultiClass; // UNDO
                //pms.Objective.Sigmoid = 0.5; // UNDO
                pms.Core.NumIterations = rand.Next(1,100);
                pms.Objective.NumClass = rand.Next(2,4); // UNDO
                pms.Common.Verbosity = VerbosityType.Error;
                var trainer = new MulticlassTrainer(pms);
                var trainData = CreateRandomDenseClassifyData(rand, pms.Objective.NumClass);
                var validData = (rand.Next(2) == 0) ? CreateRandomDenseClassifyData(rand, pms.Objective.NumClass, trainData.NumColumns) : null;
                var model = trainer.Train(trainData, validData);

                // TODO
                //using (var ms = new System.IO.MemoryStream())
                //{
                //    using (var writer = new System.IO.BinaryWriter(ms))
                //    {
                //        model.
                //    }
                //}

                foreach (var row in trainData.Features)
                {
                    VBuffer<float> output = default;
                    var input = new VBuffer<float>(row.Length, row);
                    model.GetOutput(ref input, ref output);
                    Assert.Equal(output.Values.Length, pms.Objective.NumClass);
                    foreach (var p in output.Values)
                    {
                        Assert.True(p >= 0);
                        Assert.True(p <= 1);
                    }
                    Assert.Equal(1, output.Values.Sum(), 5);
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

            var rand = new System.Random();
            for (int test = 0; test < 5; ++test)
            {
                var pms = new Parameters();
                pms.Core.Objective = objectiveTypes[rand.Next(objectiveTypes.Length)];
                pms.Core.NumIterations = rand.Next(1, 100);
                pms.Common.Verbosity = VerbosityType.Error;
                var trainer = new RegressionTrainer(pms);
                var trainData = CreateRandomDenseRegressionData(rand);
                var validData = (rand.Next(2) == 0) ? CreateRandomDenseRegressionData(rand, trainData.NumColumns) : null;

                // make labels positive for certain objective types
                if (pms.Core.Objective == ObjectiveType.Poisson ||
                    pms.Core.Objective == ObjectiveType.Gamma ||
                    pms.Core.Objective == ObjectiveType.Tweedie)
                {
                    for (var i = 0; i < trainData.Labels.Length; i++)
                        trainData.Labels[i] = Math.Abs(trainData.Labels[i]);

                    if (validData != null)
                    {
                        for (var i = 0; i < validData.Labels.Length; i++)
                            validData.Labels[i] = Math.Abs(validData.Labels[i]);
                    }
                }

                var model = trainer.Train(trainData, validData);

                // TODO
                //using (var ms = new System.IO.MemoryStream())
                //{
                //    using (var writer = new System.IO.BinaryWriter(ms))
                //    {
                //        model.
                //    }
                //}

                foreach (var row in trainData.Features)
                {
                    float output = 0;
                    var input = new VBuffer<float>(row.Length, row);
                    model.GetOutput(ref input, ref output);
                    Assert.False(Double.IsNaN(output));
                }
            }
        }

    }
}
