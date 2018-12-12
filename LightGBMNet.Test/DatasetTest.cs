using System;
using System.Collections.Generic;
using Xunit;

namespace LightGBMNet.Train.Test
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
        public static SparseMatrix Dense2Sparse(float[,] matrix)
        {
            var nnz = 0;
            for (var i = 0; i < matrix.GetLength(0); i++)
                for (var j = 0; j < matrix.GetLength(1); j++)
                    if (matrix[i, j] != 0) nnz++;

            var rslt = new SparseMatrix(matrix.GetLength(0), matrix.GetLength(1), nnz);

            var idx = 0;
            for (var i = 0; i < matrix.GetLength(0); i++)
            {
                rslt.RowExtents[i] = idx;
                for (var j = 0; j < matrix.GetLength(1); j++)
                {
                    if (matrix[i, j] != 0)
                    {
                        rslt.Data[idx] = matrix[i, j];
                        rslt.ColumnIndices[idx] = j;
                        idx++;
                    }
                }
            }
            rslt.RowExtents[rslt.RowExtents.Length - 1] = idx;

            rslt.Validate();

            return rslt;
        }

        public static SparseMatrix Dense2Sparse(float[][] rows)
        {
            var nnz = 0;
            foreach (var row in rows)
                foreach (var value in row)
                    if (value != 0) nnz++;

            var rslt = new SparseMatrix(rows.Length, rows[0].Length, nnz);

            var idx = 0;
            for (var i = 0; i < rows.Length; i++)
            {
                var row = rows[i];
                rslt.RowExtents[i] = idx;
                for (var j = 0; j < row.Length; j++)
                {
                    if (row[j] != 0)
                    {
                        rslt.Data[idx] = row[j];
                        rslt.ColumnIndices[idx] = j;
                        idx++;
                    }
                }
            }
            rslt.RowExtents[rslt.RowExtents.Length - 1] = idx;

            rslt.Validate();

            return rslt;
        }

        public static Dataset CreateRandom(Random rand)
        {
            var numTotalRow = rand.Next(100, 500);
            var numColumns = rand.Next(1, 10);

            var cp = new CommonParameters();
            var dp = new DatasetParameters
            {
                MinDataInLeaf = 1,
                MinDataInBin = 1
            };

            if (rand.Next(3) == 0)
            {
                var rows = new float[numTotalRow][];
                for (int i = 0; i < numTotalRow; ++i)
                {
                    var row = new float[numColumns];
                    for (int j = 0; j < row.Length; ++j)
                        row[j] = (float) rand.NextDouble();
                    rows[i] = row;
                }

                var ds = new Dataset(rows, numColumns, cp, dp);
                Assert.Equal(numTotalRow, ds.NumRows);
                Assert.Equal(numColumns, ds.NumFeatures);
                return ds;
            }
            else if (rand.Next(2) == 0)
            {
                var data = new float[numTotalRow, numColumns];
                for (int i = 0; i < numTotalRow; ++i)
                    for (int j = 0; j < numColumns; ++j)
                        if (rand.Next(3) == 0) data[i,j] = (float)rand.NextDouble();

                var dataSparse = Dense2Sparse(data);

                var ds = new Dataset(dataSparse, numColumns, cp, dp);
                Assert.Equal(numTotalRow, ds.NumRows);
                Assert.Equal(numColumns, ds.NumFeatures);
                return ds;
            }
            else
            {
                var numSampleRow = rand.Next(5, numTotalRow);

                var columns = new double[numColumns][];
                for (int i = 0; i < numColumns; ++i)
                {
                    var col = new double[numTotalRow];
                    for (int j = 0; j < col.Length; ++j)
                        col[j] = rand.NextDouble();
                    columns[i] = col;
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

                var ds = new Dataset(columns,
                                   sampleIndices,
                                   numColumns,
                                   sizePerColumn,
                                   numSampleRow,
                                   numTotalRow,
                                   cp,
                                   dp);
                Assert.Equal(numTotalRow, ds.NumRows);
                Assert.Equal(numColumns, ds.NumFeatures);
                return ds;
            }
        }

        [Fact]
        public void GetSetLabels()
        {
            var rand = new Random();
            for (int test = 0; test < 100; ++test)
                using (var ds = CreateRandom(rand))
                {
                    var rows = ds.NumRows;
                    var labels = new float[rows];
                    for (int i = 0; i < rows; ++i) labels[i] = (float)rand.NextDouble();
                    ds.SetLabels(labels);
                    var retLabels = ds.GetLabels();
                    Check.EqualArray(retLabels, labels);
                }
        }

        [Fact]
        public void GetSetWeights()
        {
            var rand = new Random();
            for (int test = 0; test < 100; ++test)
                using (var ds = CreateRandom(rand))
                {
                    var rows = ds.NumRows;
                    var weights = new float[rows];
                    for (int i = 0; i < weights.Length; ++i) weights[i] = (float) rand.NextDouble();
                    ds.SetWeights(weights);
                    var retWeights = ds.GetWeights();
                    Check.EqualArray(retWeights, weights);

                    if (test == 0)
                    {
                        ds.SetWeights(null);
                        Assert.Null(ds.GetWeights());
                    }
                }
        }

        [Fact]
        public void GetSetFeatureNames()
        {
            var rand = new Random();
            for (int test = 0; test < 100; ++test)
                using (var ds = CreateRandom(rand))
                {
                    var cols = ds.NumFeatures;
                    var names = new string[cols];
                    for (int i = 0; i < names.Length; ++i)
                        names[i] = string.Format("name{0}", i);
                    ds.SetFeatureNames(names);

                    var outNames = ds.FeatureNames;
                    Check.EqualArray(outNames, names);
                }
        }

        [Fact]
        public void GetSetGroups()
        {
            var rand = new Random();
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
            var rand = new Random();
            for (int test = 0; test < 100; ++test)
                using (var ds = CreateRandom(rand))
                {
                    var file = System.IO.Path.GetTempFileName();
                    var fileName = file.Substring(0, file.Length - 4) + ".bin";
                    try
                    {
                        ds.SaveBinary(fileName);

                        using (var ds2 = new Dataset(fileName, null, null))
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
            var rand = new Random();
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
