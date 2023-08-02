using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LightGBMNet.Train;
using Xunit;

namespace LightGBMNet.Train.Test
{
    public class EnumHelperTest
    {

        [Fact]
        public void TestObjectiveType()
        {
            var vals = (ObjectiveType[])Enum.GetValues(typeof(ObjectiveType));
            foreach (var v in vals)
            {
                var s = EnumHelper.GetObjectiveString(v);
                var vBar = EnumHelper.ParseObjective(s);
                Assert.Equal(v, vBar);
            }
        }

        [Fact]
        public void TestBoostingType()
        {
            var vals = (BoostingType[])Enum.GetValues(typeof(BoostingType));
            foreach (var v in vals)
            {
                var s = EnumHelper.GetBoostingString(v);
                var vBar = EnumHelper.ParseBoosting(s);
                Assert.Equal(v, vBar);
            }
        }

        [Fact]
        public void TestDataSampleStrategyType()
        {
            var vals = (DataSampleStrategyType[])Enum.GetValues(typeof(DataSampleStrategyType));
            foreach (var v in vals)
            {
                var s = EnumHelper.GetDataSampleStrategyString(v);
                var vBar = EnumHelper.ParseDataSampleStrategy(s);
                Assert.Equal(v, vBar);
            }
        }

        [Fact]
        public void TestTreeLearnerType()
        {
            var vals = (TreeLearnerType[])Enum.GetValues(typeof(TreeLearnerType));
            foreach (var v in vals)
            {
                var s = EnumHelper.GetTreeLearnerString(v);
                var vBar = EnumHelper.ParseTreeLearner(s);
                Assert.Equal(v, vBar);
            }
        }

        [Fact]
        public void TestDeviceType()
        {
            var vals = (DeviceType[])Enum.GetValues(typeof(DeviceType));
            foreach (var v in vals)
            {
                var s = EnumHelper.GetDeviceString(v);
                var vBar = EnumHelper.ParseDevice(s);
                Assert.Equal(v, vBar);
            }
        }

        [Fact]
        public void TestMetricType()
        {
            var vals = (MetricType[])Enum.GetValues(typeof(MetricType));
            foreach (var v in vals)
            {
                var s = EnumHelper.GetMetricString(v);
                var vBar = EnumHelper.ParseMetric(s);
                Assert.Equal(v, vBar);
            }
        }

        [Fact]
        public void TestMonotoneConstraintsMethod()
        {
            var vals = (MonotoneConstraintsMethod[])Enum.GetValues(typeof(MonotoneConstraintsMethod));
            foreach (var v in vals)
            {
                var s = EnumHelper.GetMonotoneConstraintsMethodString(v);
                var vBar = EnumHelper.ParseMonotoneConstraintsMethod(s);
                Assert.Equal(v, vBar);
            }
        }

    }
}
