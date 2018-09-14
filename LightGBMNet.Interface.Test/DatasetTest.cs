using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using LightGBMNet.Interface;

namespace LightGBMNet.Interface.Test
{
    internal class Check
    {
        public static void EqualArray<T>(T[] x, T[] y)
        {
            Assert.Equal(x.Length, y.Length);
            for (int i = 0; i < x.Length; ++i)
                Assert.Equal(x[i], y[i]);

        }
    }

    public class DatasetTest
    {
        public static Dataset CreateRandom(System.Random rand)
        {
            var numTotalRow = rand.Next(10, 50);
            var numSampleRow = rand.Next(5, numTotalRow);

            var numColumns = rand.Next(5, 10);
            var columns = new double[numColumns][];
            for (int i = 0; i < numColumns; ++i)
            {
                columns[i] = new double[numTotalRow];
                for (int j = 0; j < columns[i].Length; ++j) columns[i][j] = rand.NextDouble();
            }

            // select the sample indices
            var sampleIndices = new int[numColumns][];
            for (int i = 0; i < numColumns; ++i)
            {
                var sampleIndex = new int[numTotalRow];
                for (int j = 0; j < numSampleRow; ++j) sampleIndex[j] = j;
                sampleIndices[i] = sampleIndex;
            }

            var sizePerColumn = new int[numColumns];
            for (int i = 0; i < numColumns; ++i) sizePerColumn[i] = numTotalRow;


            var parameters = new Parameters();
            parameters.Learning.MinDataInLeaf = 1;
            parameters.IO.MinDataInBin = 1;

            var ds = new Dataset(columns,
                               sampleIndices,
                               numColumns,
                               sizePerColumn,
                               numSampleRow,
                               numTotalRow,
                               parameters);
            Assert.Equal(numTotalRow, ds.NumRows);
            Assert.Equal(numColumns, ds.NumFeatures);
            return ds;
        }

        [Fact]
        public void GetSetLabels()
        {
            var rand = new System.Random();
            for (int test = 0; test < 100; ++test)
                using (var ds = CreateRandom(rand))
                {
                    var rows = ds.NumRows;
                    var labels = new float[rows];
                    for (int i = 0; i < rows; ++i) labels[i] = (float)rand.NextDouble();
                    ds.SetLabel(labels);
                    var retLabels = ds.GetLabels();
                    Check.EqualArray(retLabels, labels);

                }
        }

        [Fact]
        public void GetSetWeights()
        {
            var rand = new System.Random();
            for (int test = 0; test < 100; ++test)
                using (var ds = CreateRandom(rand))
                {
                    var rows = ds.NumRows;
                    var weights = new float[rows];
                    for (int i = 0; i < weights.Length; ++i) weights[i] = (float) rand.NextDouble();
                    ds.SetWeights(weights);
                    var retWeights = ds.GetWeights();
                    Check.EqualArray(retWeights, weights);
                }
        }

        [Fact]
        public void GetSetFeatureNames()
        {
            var rand = new System.Random();
            for (int test = 0; test < 100; ++test)
                using (var ds = CreateRandom(rand))
                {
                    var cols = ds.NumFeatures;
                    var names = new string[cols];
                    for (int i = 0; i < names.Length; ++i)
                        names[i] = String.Format("name{0}", i);
                    ds.SetFeatureNames(names);

                    var outNames = ds.FeatureNames;
                    Check.EqualArray(outNames, names);
                }
        }

        [Fact]
        public void GetSetGroups()
        {
            var rand = new System.Random();
            for (int test = 0; test < 100; ++test)
                using (var ds = CreateRandom(rand))
                {
                    var rows = ds.NumRows;
                    var weights = new float[rows];
                    for (int i = 0; i < weights.Length; ++i) weights[i] = (float)rand.NextDouble();
                    ds.SetWeights(weights);
                    var retWeights = ds.GetWeights();
                    Check.EqualArray(retWeights, weights);
                }
        }

        [Fact]
        public void SaveLoad()
        {
            var rand = new System.Random();
            for (int test = 0; test < 100; ++test)
                using (var ds = CreateRandom(rand))
                {
                    var file = System.IO.Path.GetTempFileName();
                    var fileName = file.Substring(0, file.Length - 4) + ".bin";
                    try
                    {
                        ds.SaveBinary(fileName);

                        using (var ds2 = new Dataset(fileName, new Parameters()))
                        {
                            Assert.Equal(ds2.NumFeatures, ds.NumFeatures);
                            Assert.Equal(ds2.NumRows, ds.NumRows);
                        }
                    }
                    finally
                    {
                        if (System.IO.File.Exists(fileName))
                            System.IO.File.Delete(fileName);
                    }
                }
        }

        [Fact]
        public void GetSubset()
        {
            var rand = new System.Random();
            for (int test = 0; test < 100; ++test)
                using (var ds = CreateRandom(rand))
                {
                    var pms = new Parameters();
                    var rows = ds.NumRows;
                    var usedRowIndices = new List<int>(rows);
                    for(int i = 0; i < rows; ++i)
                    {
                        if (rand.NextDouble() > 0.5)
                            usedRowIndices.Add(i);
                    }
                    var numUsedRowIndices = usedRowIndices.Count;
                    using (var ds2 = ds.GetSubset(usedRowIndices.ToArray(), numUsedRowIndices))
                    {
                        Assert.Equal(ds2.NumFeatures, ds.NumFeatures);
                        Assert.Equal(ds2.NumRows, numUsedRowIndices);
                        //TODO: more rigorous testing?
                    }
                }
        }

    }
}
