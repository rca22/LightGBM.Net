using System;
using System.Collections.Generic;
using Xunit;
using LightGBMNet.Interface;

namespace LightGBMNet.Interface.Test
{
    internal class Example
    {
        public int MyInt{ get; set; } = 1;

        public double MyDouble { get; set; } = 2.0;

        public float MyFloat { get; set; } = 3.0f;

        public long MyLong { get; set; } = 4;

        public string MyString { get; set; } = "hello";

        public Example()
        {
        }

        public static bool operator == (Example t, Example other)
        {
            return (t.MyInt    == other.MyInt &&
                    t.MyDouble == other.MyDouble &&
                    t.MyFloat  == other.MyFloat &&
                    t.MyLong   == other.MyLong &&
                    t.MyString == other.MyString);
        }

        public static bool operator != (Example t, Example other)
        {
            return !(t == other);
        }

        public override bool Equals(object obj)
        {
            if (obj is Example)
            {
                return this == (Example)obj;
            }
            else
                return false;
        }
    }

    public class ParamsHelperTest
    {
        [Fact]
        public void TestWriteDefault()
        {
            var helper = new ParamsHelper<Example>();
            var result = new Dictionary<string, string>();
            var e = new Example();
            var e2 = helper.FromParameters(result);
            Assert.Equal(e, e2);
        }

        [Fact]
        public void TestReadDefault()
        {
            var helper = new ParamsHelper<Example>();
            var result = new Dictionary<string, string>();
            var e = new Example();
            helper.AddParameters(e, result);
            Assert.Empty(result);
        }

        [Fact]
        public void TestRead()
        {
            var helper = new ParamsHelper<Example>();
            var result = new Dictionary<string, string>();
            var e = new Example();
            e.MyInt = 2;
            e.MyDouble = 3.5;
            e.MyFloat = 4.5f;
            e.MyLong = 5;
            e.MyString = "goodbye";

            helper.AddParameters(e, result);
            Assert.Equal(5, result.Count);
            Assert.True(result.ContainsKey("my_int"));
            Assert.True(result.ContainsKey("my_double"));
            Assert.True(result.ContainsKey("my_float"));
            Assert.True(result.ContainsKey("my_long"));
            Assert.True(result.ContainsKey("my_string"));
            Assert.Equal("2", result["my_int"]);
            Assert.Equal("3.5", result["my_double"]);
            Assert.Equal("4.5", result["my_float"]);
            Assert.Equal("5", result["my_long"]);
            Assert.Equal("goodbye", result["my_string"]);
        }

        [Fact]
        public void TestRoundTrip()
        {
            var helper = new ParamsHelper<Example>();
            var result = new Dictionary<string, string>();
            var e = new Example();
            e.MyInt = 2;
            e.MyDouble = 3.5;
            e.MyFloat = 4.5f;
            e.MyLong = 5;
            e.MyString = "goodbye";

            helper.AddParameters(e, result);
            var e2 = helper.FromParameters(result);
            Assert.Equal(e, e2);
        }
    }

    public class ParametersTest
    {
        [Fact]
        public void TestGPUParametersDefault()
        {
            var x = new GPUParameters();
            var result = new Dictionary<string, string>();
            x.AddParameters(result);
            Assert.Empty(result);
        }

        [Fact]
        public void TestNetworkParametersDefault()
        {
            var x = new NetworkParameters();
            var result = new Dictionary<string, string>();
            x.AddParameters(result);
            Assert.Empty(result);
        }

        [Fact]
        public void TestMetricParametersDefault()
        {
            var x = new MetricParameters();
            var result = new Dictionary<string, string>();
            x.AddParameters(result);
            Assert.Empty(result);
        }

        [Fact]
        public void TestObjectiveParametersDefault()
        {
            var x = new ObjectiveParameters();
            var result = new Dictionary<string, string>();
            x.AddParameters(result);
            Assert.Empty(result);
        }

        [Fact]
        public void TestIOParametersDefault()
        {
            var x = new IOParameters();
            var result = new Dictionary<string, string>();
            x.AddParameters(result);
            Assert.Empty(result);
        }

        [Fact]
        public void TestCoreParametersDefault()
        {
            var x = new CoreParameters();
            var result = new Dictionary<string, string>();
            x.AddParameters(result);
            Assert.Empty(result);
        }
        [Fact]

        public void TestLearningControlParametersDefault()
        {
            var x = new LearningControlParameters();
            var result = new Dictionary<string, string>();
            x.AddParameters(result);
            Assert.Empty(result);
        }
    }
}
