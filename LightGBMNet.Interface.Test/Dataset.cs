using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using LightGBMNet.Interface;

namespace LightGBMNet.Interface.Test
{
    public class DatasetTest
    {
        [Fact]
        public void Create()
        {
            var rand = new System.Random();
            
            var numTotalRow = rand.Next(10,50);
            var numSampleRow = rand.Next(5,numTotalRow);

            var numColumns = rand.Next(5, 10);
            var columns = new double[numColumns][];
            for (int i = 0; i < numColumns; ++i)
            {
                columns[i] = new double[numTotalRow];
                for (int j = 0; j < columns[i].Length; ++j) columns[i][j] = rand.NextDouble();
            }

            // select the sample indices
            var sampleIndex = new int[numSampleRow];
            for (int i = 0; i < numSampleRow; ++i) sampleIndex[i] = i;
            var sampleIndices = new int[numColumns][];
            for (int i = 0; i < numColumns; ++i) sampleIndices[i] = sampleIndex;

            var sizePerColumn = new int[numColumns];
            for (int i = 0; i < numColumns; ++i) sizePerColumn[i] = numTotalRow;

            var labels = new float[numTotalRow];
            for (int i = 0; i < numTotalRow; ++i) labels[i] = (float)i;
            using (var dataset =  new Dataset(columns,
                                              sampleIndices,
                                              numColumns,
                                              sizePerColumn,
                                              numSampleRow,
                                              numTotalRow,
                                              "min_data=1 min_data_in_bin=1",
                                              labels))
            {
                Assert.Equal(numTotalRow, dataset.GetNumRows());
                Assert.Equal(numColumns, dataset.GetNumCols());
            }
        }
    }
}
