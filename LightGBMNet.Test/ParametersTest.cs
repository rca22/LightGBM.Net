using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using Xunit;

namespace LightGBMNet.Train.Test
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

        public override int GetHashCode()
        {
            return base.GetHashCode();
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
        public void TestDatasetParametersDefault()
        {
            var x = new DatasetParameters();
            var result = new Dictionary<string, string>();
            x.AddParameters(result);
            Assert.Empty(result);
        }

        [Fact]
        public void TestCommonParametersDefault()
        {
            var x = new CommonParameters();
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
        public void TestLearningParametersDefault()
        {
            var x = new LearningParameters();
            var result = new Dictionary<string, string>();
            x.AddParameters(result);
            Assert.Empty(result);
        }

        private static readonly int Seed = (new Random()).Next();

        [Fact]
        public void TestDatasetParametersEquality()
        {
            var rand = new Random(Seed);
            var x = GenerateRandom<DatasetParameters>(rand);
            var y = Clone(x);
            Assert.Equal(x, y);
            Assert.True(x.Equals(y));
            Assert.True(x == y);
            Assert.False(x != y);
            Assert.Equal(x.GetHashCode(), y.GetHashCode());

            Modify(y, rand);
            Assert.NotEqual(x, y);
            Assert.False(x.Equals(y));
            Assert.False(x == y);
            Assert.True(x != y);
            Assert.NotEqual(x.GetHashCode(), y.GetHashCode());
        }

        [Fact]
        public void TestLearningParametersEquality()
        {
            var rand = new Random(Seed);
            var x = GenerateRandom<LearningParameters>(rand);
            var y = Clone(x);
            Assert.Equal(x, y);
            Assert.True(x.Equals(y));
            Assert.True(x == y);
            Assert.False(x != y);
            Assert.Equal(x.GetHashCode(), y.GetHashCode());

            Modify(y, rand);
            Assert.NotEqual(x, y);
            Assert.False(x.Equals(y));
            Assert.False(x == y);
            Assert.True(x != y);
            Assert.NotEqual(x.GetHashCode(), y.GetHashCode());
        }

        [Fact]
        public void TestObjectiveParametersEquality()
        {
            var rand = new Random(Seed);
            var x = GenerateRandom<ObjectiveParameters>(rand);
            var y = Clone(x);
            Assert.Equal(x, y);
            Assert.True(x.Equals(y));
            Assert.True(x == y);
            Assert.False(x != y);
            Assert.Equal(x.GetHashCode(), y.GetHashCode());

            Modify(y, rand);
            Assert.NotEqual(x, y);
            Assert.False(x.Equals(y));
            Assert.False(x == y);
            Assert.True(x != y);
            Assert.NotEqual(x.GetHashCode(), y.GetHashCode());
        }

        [Fact]
        public void TestCommonParametersEquality()
        {
            var rand = new Random(Seed);
            var x = GenerateRandom<CommonParameters>(rand);
            var y = Clone(x);
            Assert.Equal(x, y);
            Assert.True(x.Equals(y));
            Assert.True(x == y);
            Assert.False(x != y);
            Assert.Equal(x.GetHashCode(), y.GetHashCode());

            Modify(y, rand);
            Assert.NotEqual(x, y);
            Assert.False(x.Equals(y));
            Assert.False(x == y);
            Assert.True(x != y);
            Assert.NotEqual(x.GetHashCode(), y.GetHashCode());
        }

        [Fact]
        public void TestParametersEquality()
        {
            var rand = new Random(Seed);
            var x = new Parameters
            {
                Common    = GenerateRandom<CommonParameters>(rand),
                Objective = GenerateRandom<ObjectiveParameters>(rand),
                Dataset   = GenerateRandom<DatasetParameters>(rand),
                Learning  = GenerateRandom<LearningParameters>(rand)
            };
            var y = new Parameters
            {
                Common    = Clone(x.Common),
                Objective = Clone(x.Objective),
                Dataset   = Clone(x.Dataset),
                Learning  = Clone(x.Learning)
            };
            Assert.Equal(x, y);
            Assert.True(x.Equals(y));
            Assert.True(x == y);
            Assert.False(x != y);
            Assert.Equal(x.GetHashCode(), y.GetHashCode());

            var flag = rand.Next(4);
            if (flag == 0) Modify(y.Common, rand);
            else if (flag == 1) Modify(y.Objective, rand);
            else if (flag == 2) Modify(y.Dataset, rand);
            else Modify(y.Learning, rand);

            Assert.NotEqual(x, y);
            Assert.False(x.Equals(y));
            Assert.False(x == y);
            Assert.True(x != y);
            Assert.NotEqual(x.GetHashCode(), y.GetHashCode());

        }

        internal static object GenerateEnum(Random r, Type t)
        {
            var values = Enum.GetValues(t);
            return values.GetValue(r.Next(values.Length));
        }

        internal static T GenerateRandom<T>(Random r) where T : new()
        {
            var rslt = new T();
            var props = typeof(T).GetProperties();
            foreach (var prop in props)
            {
                if (prop.CanWrite && r.Next(2) == 0)
                {
                    try
                    {
                        var typ = prop.PropertyType;
                        if (typ == typeof(Int32))
                            prop.SetValue(rslt, r.Next());
                        else if (typ == typeof(Int64))
                            prop.SetValue(rslt, (Int64)r.Next());
                        else if (typ == typeof(double))
                            prop.SetValue(rslt, r.NextDouble());
                        else if (typ == typeof(float))
                            prop.SetValue(rslt, (float)r.NextDouble());
                        else if (typ == typeof(string))
                            prop.SetValue(rslt, r.Next().ToString());
                        else if (typ == typeof(bool))
                            prop.SetValue(rslt, r.Next(2) == 0);
                        else if (typ == typeof(int[]))
                            prop.SetValue(rslt, Enumerable.Range(0, r.Next(10)).Select(x => r.Next()).ToArray());
                        else if (typ == typeof(double[]))
                            prop.SetValue(rslt, Enumerable.Range(0, r.Next(10)).Select(x => r.NextDouble()).ToArray());
                        else if (typ.IsEnum)
                            prop.SetValue(rslt, GenerateEnum(r, typ));
                        else
                            throw new Exception(String.Format("Unhandled parameter type {0}", typ));
                    }
                    catch (System.Reflection.TargetInvocationException e)
                    {
                        if (!(e.InnerException is ArgumentOutOfRangeException))
                            throw;
                    }
                }
            }
            return rslt;
        }

        internal static void Modify<T>(T src, Random r)
        {
            var props = typeof(T).GetProperties();
            var modified = false;
            while (!modified)
            {
                var prop = props[r.Next(props.Length)];
                {
                    if (prop.CanWrite)
                    {
                        try
                        {
                            var prev = prop.GetValue(src);
                            var typ = prop.PropertyType;
                            if (typ == typeof(Int32))
                                prop.SetValue(src, r.Next());
                            else if (typ == typeof(Int64))
                                prop.SetValue(src, (Int64)r.Next());
                            else if (typ == typeof(double))
                                prop.SetValue(src, r.NextDouble());
                            else if (typ == typeof(float))
                                prop.SetValue(src, (float)r.NextDouble());
                            else if (typ == typeof(string))
                                prop.SetValue(src, r.Next().ToString());
                            else if (typ == typeof(bool))
                                prop.SetValue(src, r.Next(2) == 0);
                            else if (typ == typeof(int[]))
                                prop.SetValue(src, Enumerable.Range(0, r.Next(10)).Select(x => r.Next()).ToArray());
                            else if (typ == typeof(double[]))
                                prop.SetValue(src, Enumerable.Range(0, r.Next(10)).Select(x => r.NextDouble()).ToArray());
                            else if (typ.IsEnum)
                                prop.SetValue(src, GenerateEnum(r, typ));
                            else
                                throw new Exception(String.Format("Unhandled parameter type {0}", typ));
                            var curr = prop.GetValue(src);
                            modified = !Equal(prev, curr);
                        }
                        catch (System.Reflection.TargetInvocationException e)
                        {
                            if (!(e.InnerException is ArgumentOutOfRangeException))
                                throw;
                        }
                    }
                }
            }
        }

        internal static bool Equal(object x, object y)
        {
            if (x == null && y == null) return true;
            if (x == null || y == null) return false;
            if (x.GetType() != y.GetType()) return false;
            if (x.GetType().IsArray) return EqualArrays((Array)x, (Array)x);
            return x.Equals(y);
        }

        internal static bool EqualArrays(Array x, Array y)
        {
            bool ok = x.Rank == y.Rank && x.Length == y.Length;
            int i = 0;
            while (ok && i < x.Length)
            {
                ok = x.GetValue(i).Equals(y.GetValue(i));
                i++;
            }
            return ok;
        }

        internal static Array CloneArray(Array src)
        {
            var dst = Array.CreateInstance(src.GetType().GetElementType(), src.Length);
            for (var i = 0; i < src.Length; i++)
                dst.SetValue(src.GetValue(i), i);
            return dst;
        }

        internal static T Clone<T>(T src) where T : new()
        {
            var rslt = new T();
            var props = typeof(T).GetProperties();
            foreach (var prop in props)
            {
                if (prop.CanWrite)
                {
                    try
                    {
                        var typ = prop.PropertyType;
                        if (typ == typeof(Int32) ||
                            typ == typeof(Int64) ||
                            typ == typeof(double) ||
                            typ == typeof(float) ||
                            typ == typeof(string) ||
                            typ == typeof(bool) ||
                            typ.IsEnum
                           )
                            prop.SetValue(rslt, prop.GetValue(src));
                        else if (typ.IsArray)
                            prop.SetValue(rslt, CloneArray((Array)prop.GetValue(src)));
                        else
                            throw new Exception(String.Format("Unhandled parameter type {0}", typ));
                    }
                    catch (Exception e)
                    {
                        throw new Exception($"Failed to clone property {prop.Name}", e);
                    }
                }
            }
            return rslt;
        }
    }
}
