using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using LightGBMNet.Tree;

namespace LightGBMNet.Train.Test
{
    public class NetworkConfigTest
    {
        //checks PInvoke calls only
        [Fact]
        public void FromConfig()
        {
            using (var nc = new NetworkConfig(new CommonParameters()))
            {
            }
        }

        //Checks PInvoke calls only
        [Fact]
        public void FromFunction()
        {
            void reduce(byte[] input, int inputSize, int typeSize,
                                         int[] blockStart, int[] blockLen, int numBlock,
                                         byte[] output, int outputSize, IntPtr reducer)
            {
            }

            void gather(byte[] input, int inputSize, int[] blockStart, int[] blockLen, int numBlock, byte[] output, int outputSize)
            {
            }

            using (var nc = new NetworkConfig(2, 4, reduce, gather))
            {
            }
        }
    }
}
