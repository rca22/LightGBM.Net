using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Reflection;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("LightGBMNet.Interface.Test")]

namespace LightGBMNet.Interface
{
    public enum TaskType : int
    {
        Train = 0,
        Predict = 1,
        ConvertModel = 2,
        Refit = 3
    }

    public enum ObjectiveType : int
    {
        // regression
        Regression = 0,
        RegressionL1 = 1,
        Huber = 2,
        Fair = 3,
        Poisson = 4,
        Quantile = 5,
        Mape = 6,
        Gamma = 7,
        Tweedie = 8,
        // binary
        Binary = 9,
        // multiclass
        MultiClass = 10,
        MultiClassOva = 11,
        // cross-entropy
        XEntropy = 12,
        XEntLambda = 13,
        // ranking
        LambdaRank = 14
    }

    public enum BoostingType : int
    {
        /// Traditional Gradient Boosting Decision Tree
        GBDT,
        /// Random Forest
        RandomForest,
        /// Dropouts meet Multiple Additive Regression Trees
        Dart,
        /// Gradient-based One-Side Sampling
        Goss
    }

    public enum TreeLearnerType
    {
        /// Single machine tree learner
        Serial,
        /// Feature parallel tree learner
        Feature,
        /// Data parallel tree learner
        Data,
        /// Voting parallel tree learner
        Voting
    }

    public enum DeviceType
    {
        CPU,
        GPU
    }

    public enum VerbosityType : int
    {
        Fatal = -1,
        Error = 0,
        Info = 1
    }

    public enum MetricType
    {
        /// <summary>
        /// Metric corresponding to specified objective will be used.
        /// </summary>
        DefaultMetric,
        /// <summary>
        /// No metric will be registered
        /// </summary>
        None,
        /// <summary>
        /// Mean absolute error
        /// </summary>
        Mae,
        /// <summary>
        /// Mean squared error
        /// </summary>
        Mse,
        /// <summary>
        /// Root mean squared error
        /// </summary>
        Rmse,
        /// <summary>
        /// Quantile regression
        /// </summary>
        Quantile,
        /// <summary>
        /// Mean absolute percentage error
        /// </summary>
        Mape,
        /// <summary>
        /// Huber loss
        /// </summary>
        Huber,
        /// <summary>
        /// Fair loss
        /// </summary>
        Fair,
        /// <summary>
        /// Negative log-likelihood for Poisson regression
        /// </summary>
        Poisson,
        /// <summary>
        /// Negative log-likelihood for Gamma regression
        /// </summary>
        Gamma,
        /// <summary>
        /// Residual deviance for Gamma regression
        /// </summary>
        GammaDeviance,
        /// <summary>
        /// Negative log-likelihood for Tweedie regression
        /// </summary>
        Tweedie,
        /// <summary>
        /// Normalised discounted cumulative gain
        /// </summary>
        Ndcg,
        /// <summary>
        /// Mean average precision
        /// </summary>
        Map,
        /// <summary>
        /// Area under the curve
        /// </summary>
        Auc,
        BinaryLogLoss,
        /// <summary>
        /// for one sample: 0 for correct classification, 1 for error classification
        /// </summary>
        BinaryError,
        /// <summary>
        /// log loss for multi-class classification
        /// </summary>
        MultiLogLoss,
        /// <summary>
        /// error rate for multi-class classification
        /// </summary>
        MultiError,
        /// <summary>
        /// Cross-entropy (with optional linear weights)
        /// </summary>
        XEntropy,
        /// <summary>
        /// "intensity-weighted" cross-entropy
        /// </summary>
        XentLambda,
        /// <summary>
        /// Kullback-Leibler divergence
        /// </summary>
        Kldiv
    };

    public static class EnumHelper
    {
        public static string GetMetricString(MetricType m)
        {
            switch (m)
            {
                case MetricType.DefaultMetric:
                    return "";
                case MetricType.None:
                    return "None";
                case MetricType.Mae:
                    return "l1";
                case MetricType.Mse:
                    return "l2";
                case MetricType.Rmse:
                    return "rmse";
                case MetricType.Quantile:
                    return "quantile";
                case MetricType.Mape:
                    return "mape";
                case MetricType.Huber:
                    return "huber";
                case MetricType.Fair:
                    return "fair";
                case MetricType.Poisson:
                    return "poission";
                case MetricType.Gamma:
                    return "gamma";
                case MetricType.GammaDeviance:
                    return "gamma_deviance";
                case MetricType.Tweedie:
                    return "tweedie";
                case MetricType.Ndcg:
                    return "ndcg";
                case MetricType.Map:
                    return "map";
                case MetricType.Auc:
                    return "auc";
                case MetricType.BinaryLogLoss:
                    return "binary_logloss";
                case MetricType.BinaryError:
                    return "binary_error";
                case MetricType.MultiLogLoss:
                    return "multi_logloss";
                case MetricType.MultiError:
                    return "multi_error";
                case MetricType.XEntropy:
                    return "xentropy";
                case MetricType.XentLambda:
                    return "xentlambda";
                case MetricType.Kldiv:
                    return "kldiv";
                default:
                    throw new System.Exception(String.Format("Unexpected value of MetricType {0}",m));
            }
        }

        public static MetricType ParseMetric(string x)
        {
            switch (x)
            {
                case "":
                    return MetricType.DefaultMetric;
                case "None":
                case "na":
                case "null":
                case "custom":
                    return MetricType.None;
                case "l1":
                case "mean_absolute_error":
                case "mae":
                case "regression_l1":
                    return MetricType.Mae;
                case "mean_squared_error":
                case "mse":
                case "regression_l2":
                case "regression":
                case "l2":
                    return MetricType.Mse;
                case "l2_root":
                case "root_mean_squared_error":
                case "rmse":
                    return MetricType.Rmse;
                case "quantile":
                    return MetricType.Quantile;
                case "mape":
                case "mean_absoluete_percentage_error":
                    return MetricType.Mape;
                case "huber":
                    return MetricType.Huber;
                case "fair":
                    return MetricType.Fair;
                case "poission":
                    return MetricType.Poisson;
                case "gamma":
                    return MetricType.Gamma;
                case "gamma_deviance":
                case "gamma-deviance"://seems to be a typo in main library
                    return MetricType.GammaDeviance;
                case "tweedie":
                    return MetricType.Tweedie;
                case "ndcg":
                case "lambdarank":
                    return MetricType.Ndcg;
                case "map":
                case "mean_average_precision":
                    return MetricType.Map;
                case "auc":
                    return MetricType.Auc;
                case "binary_logloss":
                case "binary":
                    return MetricType.BinaryLogLoss;
                case "binary_error":
                    return MetricType.BinaryError;
                case "multi_logloss":
                case "multiclass":
                case "softmax":
                case "multiclassova":
                case "multiclass_ova":
                case "ova":
                case "ovr":
                    return MetricType.MultiLogLoss;
                case "multi_error":
                    return MetricType.MultiError;
                case "xentropy":
                case "cross_entropy":
                    return MetricType.XEntropy;
                case "xentlambda":
                case "cross_entity_lambda":
                    return MetricType.XentLambda;
                case "kldiv":
                case "kullback_leibler":
                    return MetricType.Kldiv;
                default:
                    throw new System.Exception(String.Format("Unexpected value of MetricType {0}",x));
            }
        }

        public static string GetTaskString(TaskType t)
        {
            switch(t)
            {
                case TaskType.Train:        return "train";
                case TaskType.Predict:      return "predict";
                case TaskType.ConvertModel: return "convert_model";
                case TaskType.Refit:        return "refit";
                default:
                    throw new ArgumentException("TaskType");
            }
        }

        public static TaskType ParseTask(string x)
        {
            switch(x)
            {
                case "train":
                case "training":
                    return TaskType.Train;
                case "predict":
                case "prediction":
                case "test":
                    return TaskType.Predict;
                case "convert_model":
                    return TaskType.ConvertModel;
                case "refit":
                case "refit_tree":
                    return TaskType.Refit;
                default:
                    throw new ArgumentException("x");
            }
        }

        public static string GetObjectiveString(ObjectiveType o)
        {
            switch(o)
            {
                case ObjectiveType.Regression:   return "regression";
                case ObjectiveType.RegressionL1: return "regression_l1";
                case ObjectiveType.Huber:        return "huber";
                case ObjectiveType.Fair:         return "fair";
                case ObjectiveType.Poisson:      return "poisson";
                case ObjectiveType.Quantile:     return "quantile";
                case ObjectiveType.Mape:         return "mape";
                case ObjectiveType.Gamma:        return "gamma";
                case ObjectiveType.Tweedie:      return "tweedie";
                case ObjectiveType.Binary:       return "binary";
                case ObjectiveType.MultiClass:   return "multiclass";
                case ObjectiveType.MultiClassOva:return "multiclassova";
                case ObjectiveType.XEntropy:     return "xentropy";
                case ObjectiveType.XEntLambda:   return "xentlambda";
                case ObjectiveType.LambdaRank:   return "lambdarank";
                default:
                    throw new ArgumentException("ObjectiveType not recognised");
            }
        }

        public static ObjectiveType ParseObjective(string o)
        {
            switch(o)
            {
                case "regression":
                case "regression_l2":
                case "mean_squared_error":
                case "mse":
                case "l2_root":
                case "root_mean_squared_error":
                case "rmse":
                    return ObjectiveType.Regression;
                case "regression_l1":
                case "mean_absolute_error":
                case "mae":
                    return ObjectiveType.RegressionL1;
                case "huber":         return ObjectiveType.Huber;
                case "fair":          return ObjectiveType.Fair;
                case "poisson":       return ObjectiveType.Poisson;
                case "quantile":      return ObjectiveType.Quantile;
                case "mape":
                case "mean_absolute_percentage_error":
                    return ObjectiveType.Mape;
                case "gamma":         return ObjectiveType.Gamma;
                case "tweedie":       return ObjectiveType.Tweedie;
                case "binary":        return ObjectiveType.Binary;
                case "multiclass":
                case "softmax":
                    return ObjectiveType.MultiClass;
                case "multiclassova":
                case "multiclass_ova":
                case "ova":
                case "ovr":
                    return ObjectiveType.MultiClassOva;
                case "xentropy":
                case "cross_entropy":
                    return ObjectiveType.XEntropy;
                case "xentlambda":
                case "cross_entropy_lambda":
                    return ObjectiveType.XEntLambda;
                case "lambdarank":    return ObjectiveType.LambdaRank;
                default:
                    throw new ArgumentException("ObjectiveType not recognised");
            }
        }

        public static string GetBoostingString(BoostingType t)
        {
            switch(t)
            {
                case BoostingType.GBDT: return "gbdt";
                case BoostingType.Dart: return "dart";
                case BoostingType.RandomForest: return "rf";
                case BoostingType.Goss: return "goss";
                default:
                    throw new ArgumentException("BoostingType not found");
            }
        }

        public static BoostingType ParseBoosting(string x)
        {
            switch (x)
            {
                case "gbdt":
                case "gbrt":
                    return BoostingType.GBDT;
                case "rf":
                case "random_forest":
                    return BoostingType.RandomForest;
                case "dart":
                    return BoostingType.Dart;
                case "goss":
                    return BoostingType.Goss;
                default:
                    throw new ArgumentException("BoostingType not found");
            }
        }

        public static string GetTreeLearnerString(TreeLearnerType t)
        {
            switch(t)
            {
                case TreeLearnerType.Data: return "data";
                case TreeLearnerType.Feature: return "feature";
                case TreeLearnerType.Serial: return "serial";
                case TreeLearnerType.Voting: return "voting";
                default:
                    throw new ArgumentException("TreeLearnerType not recognised");
            }
        }

        public static TreeLearnerType ParseTreeLearner(string x)
        {
            switch (x)
            {
                case "serial":
                    return TreeLearnerType.Serial;
                case "feature":
                case "feature_parallel":
                    return TreeLearnerType.Feature;
                case "data":
                case "data_parallel":
                    return TreeLearnerType.Data;
                case "voting":
                case "voting_parallel":
                    return TreeLearnerType.Voting;
                default:
                    throw new ArgumentException("TreeLearnerType not recognised");
            }
        }

        public static string GetDeviceString(DeviceType d)
        {
            switch (d)
            {
                case DeviceType.CPU: return "cpu";
                case DeviceType.GPU: return "gpu";
                default:
                    throw new ArgumentException("DeviceType not recognised");
            }
        }

        public static DeviceType ParseDevice(string x)
        {
            switch(x)
            {
                case "cpu": return DeviceType.CPU;
                case "gpu": return DeviceType.GPU;
                default:
                    throw new ArgumentException("DeviceType not recognised");
            }
        }

        public static string GetVerbosityString(VerbosityType v)
        {
            switch(v)
            {
                case VerbosityType.Fatal: return "-1";
                case VerbosityType.Error: return "0";
                case VerbosityType.Info: return "1";
                default:
                    throw new ArgumentException("VerbosityType not recognised");
            }
        }

        public static VerbosityType ParseVerbosity(string x)
        {
            switch (x)
            {
                case "-1": return VerbosityType.Fatal;
                case "0" : return VerbosityType.Error;
                case "1" : return VerbosityType.Info;
                default:
                    throw new ArgumentException("VerbosityType not recognised");
            }
        }
    }

    public static class ParamsHelper
    {
        public static string JoinParameters(Dictionary<string, string> parameters)
        {
            Check.NonNull(parameters, nameof(parameters));

            if (parameters == null)
                throw new ArgumentNullException("parameters");
            var res = parameters.Select(keyVal => keyVal.Key + "=" + keyVal.Value);
            return string.Join(" ", res);
        }

        public static Dictionary<string,string> SplitParameters(string p)
        {
            Check.NonNull(p, nameof(p));

            var bits = p.Split(new char[] { '\t', '\n', '\r', ' ','=' }, StringSplitOptions.RemoveEmptyEntries);
            if (bits.Length % 2 == 1)
                throw new ArgumentException("Unable to parse persisted parameters", "p");
            var cnt = bits.Length / 2;
            var rslt = new Dictionary<string, string>(cnt);
            for (int i = 0; i < cnt; ++i)
                rslt.Add(bits[2 * i], bits[2 * i + 1]);
            return rslt;
        }
    }

    public class ParamsHelper<T>
         where T : class, new()
    {
        private Dictionary<PropertyInfo, Tuple<string,WriteFunction,string>> _propToArgNameAndDefault;
        
        private Dictionary<string, Tuple<PropertyInfo,ParseFunction>> _argToProp;

        private delegate object ParseFunction(string input);
        private delegate string WriteFunction(object input);

        private static ParseFunction CreateParseFunction(Type typ)
        {
            if (typ == typeof(Int32))
                return x => (object)Int32.Parse(x);
            else if (typ == typeof(Int64))
                return x => (object)Int64.Parse(x);
            else if (typ == typeof(double))
                return x => (object)Double.Parse(x);
            else if (typ == typeof(float))
                return x => (object)float.Parse(x);
            else if (typ == typeof(string))
                return x => (object)x;
            else if (typ == typeof(bool))
                return x => (object)Boolean.Parse(x);
            else if (typ == typeof(int[]))
                return x => x.Split(new char[',']).Select(Int32.Parse).ToArray();
            else if (typ == typeof(double[]))
                return x => x.Split(new char[',']).Select(Double.Parse).ToArray();
            else if (typ == typeof(MetricType))
                return x => (object)EnumHelper.ParseMetric(x);
            else if (typ == typeof(TaskType))
                return x => (object)EnumHelper.ParseTask(x);
            else if (typ == typeof(ObjectiveType))
                return x => (object)EnumHelper.ParseObjective(x);
            else if (typ == typeof(BoostingType))
                return x => (object)EnumHelper.ParseBoosting(x);
            else if (typ == typeof(TreeLearnerType))
                return x => (object)EnumHelper.ParseTreeLearner(x);
            else if (typ == typeof(DeviceType))
                return x => (object)EnumHelper.ParseDevice(x);
            else if (typ == typeof(VerbosityType))
                return x => (object)EnumHelper.ParseVerbosity(x);
            else
                throw new Exception(String.Format("Unhandled parameter type {0}", typ));
        }

        private static WriteFunction CreateWriteFunction(Type typ)
        {
            if (typ == typeof(Int32))
                return x => x.ToString();
            else if (typ == typeof(Int64))
                return x => x.ToString();
            else if (typ == typeof(double))
                return x => x.ToString();
            else if (typ == typeof(float))
                return x => x.ToString();
            else if (typ == typeof(string))
                return x => x.ToString();
            else if (typ == typeof(bool))
                return x => ((bool)x) ? "true" : "false";
            else if (typ == typeof(int[]))
                return x => String.Join(",", (x as int[]).Select(y => y.ToString()));
            else if (typ == typeof(double[]))
                return x => String.Join(",", (x as double[]).Select(y => y.ToString()));
            else if (typ == typeof(MetricType))
                return x => EnumHelper.GetMetricString((MetricType)x);
            else if (typ == typeof(TaskType))
                return x => EnumHelper.GetTaskString((TaskType)x);
            else if (typ == typeof(ObjectiveType))
                return x => EnumHelper.GetObjectiveString((ObjectiveType)x);
            else if (typ == typeof(BoostingType))
                return x => EnumHelper.GetBoostingString((BoostingType)x);
            else if (typ == typeof(TreeLearnerType))
                return x => EnumHelper.GetTreeLearnerString((TreeLearnerType)x);
            else if (typ == typeof(DeviceType))
                return x => EnumHelper.GetDeviceString((DeviceType)x);
            else if (typ == typeof(VerbosityType))
                return x => EnumHelper.GetVerbosityString((VerbosityType)x);
            else
                throw new Exception(String.Format("Unhandled parameter type {0}", typ));
        }

        // Converts a camel case name into an underscore separated lower-case name.
        private static string GetArgName(string name)
        {
            StringBuilder strBuf = new StringBuilder();
            bool first = true;
            foreach (char c in name)
            {
                if (char.IsUpper(c))
                {
                    if (first)
                        first = false;
                    else
                        strBuf.Append('_');
                    strBuf.Append(char.ToLower(c));
                }
                else
                    strBuf.Append(c);
            }
            return strBuf.ToString();
        }

        public ParamsHelper()
        {
            _propToArgNameAndDefault = new Dictionary<PropertyInfo, Tuple<string, WriteFunction, string>>();
            _argToProp = new Dictionary<string, Tuple<PropertyInfo,ParseFunction>>();

            var dflt = new T();
            var props = typeof(T).GetProperties();
            foreach (var prop in props)
            {
                var argName = GetArgName(prop.Name);
                var deft = prop.GetValue(dflt);
                var writer = CreateWriteFunction(prop.PropertyType);
                var parser = CreateParseFunction(prop.PropertyType);
                var sDeft = writer(deft);
                _propToArgNameAndDefault.Add(prop, Tuple.Create(argName, writer, sDeft));
                _argToProp.Add(argName, Tuple.Create(prop,parser));
            }
        }

        public void AddParameters(T input, Dictionary<string, string> result)
        {
            foreach (var propNameDefault in _propToArgNameAndDefault)
            {
                var prop = propNameDefault.Key;
                var name = propNameDefault.Value.Item1;
                var writer = propNameDefault.Value.Item2;
                var sDefault = propNameDefault.Value.Item3;
                // do not bother to persist values which are the default.
                var cand = writer(prop.GetValue(input));
                if (cand != sDefault)
                    result.Add(name, cand);
            }
        }

        public T FromParameters(Dictionary<string,string> pms)
        {
            var rslt = new T();
            Tuple<PropertyInfo,ParseFunction> pair = null;
            foreach (var pm in pms)
            {
                if (_argToProp.TryGetValue(pm.Key, out pair))
                {
                    var prop = pair.Item1;
                    var parser = pair.Item2;
                    var obj = parser(pm.Value);
                    prop.SetValue(rslt, obj);
                }
            }
            return rslt;
        }
    }

    public interface IParameters
    {

    }
    public abstract class ParametersBase<T> 
        where T : ParametersBase<T>, new()
    {
        static protected ParamsHelper<T> _helper = new ParamsHelper<T>();

        static public T FromParameters(Dictionary<string, string> pms)
        {
            return _helper.FromParameters(pms);
        }

        protected ParametersBase()
        {
        }

        abstract public void AddParameters(Dictionary<string, string> result);

        public string ParameterString()
        {
            var result = new Dictionary<string, string>();
            this.AddParameters(result);
            var cmpts = new List<string>(result.Count);
            foreach (var keyVal in result)
                cmpts.Add(keyVal.Key + "=" + keyVal.Value);
            return string.Join(" ", cmpts);
        }
    }

    public class DatasetParameters : ParametersBase<DatasetParameters>
    {
        #region Properties

        // Not a true parameter, just controls size of batches when copying data across to Dataset (TODO: deprecate)
        [Obsolete]
        public int BatchSize => 1 << 20;

        private int _maxBin = 255;
        /// <summary>
        /// Max number of bins that feature values will be bucketed in.
        /// Small number of bins may reduce training accuracy but may increase general power (deal with over-fitting)
        /// </summary>
        public int MaxBin
        {
            get { return _maxBin; }
            set
            {
                if (value <= 1)
                    throw new ArgumentOutOfRangeException("max_bin");
                _maxBin = value;
            }
        }

        private int _minDataInBin = 3;
        /// <summary>
        /// Minimal number of data inside one bin.
        /// Use this to avoid one-data-one-bin (potential over-fitting)
        /// </summary>
        public int MinDataInBin
        {
            get { return _minDataInBin; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("min_data_in_bin");
                _minDataInBin = value;
            }
        }

        private int _binConstructSampleCnt = 200000;
        /// <summary>
        /// Number of data that sampled to construct histogram bins.
        /// Setting this to larger value will give better training result, but will increase data loading time.
        /// </summary>
        public int BinConstructSampleCnt
        {
            get { return _binConstructSampleCnt; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("bin_construct_sample_cnt");
                _binConstructSampleCnt = value;
            }
        }

        /// <summary>
        /// Random seed for data partition in parallel learning (excluding the feature_parallel mode)
        /// </summary>
        public double DataRandomSeed { get; set; } = 1;

        /// <summary>
        /// Used for parallel learning (excluding the feature_parallel mode).
        /// True if training data are pre-partitioned, and different machines use different partitions.
        /// </summary>
        public bool PrePartition { get; set; } = false;

        /// <summary>
        /// Set this to false to disable Exclusive Feature Bundling (EFB), which is described in LightGBM: A Highly Efficient Gradient Boosting Decision Tree
        /// Note: disabling this may cause the slow training speed for sparse datasets
        /// </summary>
        public bool EnableBundle { get; set; } = true;

        private double _maxConflictRate = 0.0;
        /// <summary>
        /// Max conflict rate for bundles in EFB
        /// Set this to 0.0 to disallow the conflict and provide more accurate results
        /// Set this to a larger value to achieve faster speed
        /// </summary>
        public double MaxConflictRate
        {
            get { return _maxConflictRate; }
            set
            {
                if (value < 0.0 || value >= 1.0)
                    throw new ArgumentOutOfRangeException("max_conflict_rate");
                _maxConflictRate = value;
            }
        }

        /// <summary>
        /// Used to enable/disable sparse optimization
        /// </summary>
        public bool IsEnableSparse { get; set; } = true;

        private double _sparseThreshold = 0.8;
        /// <summary>
        /// The threshold of zero elements percentage for treating a feature as a sparse one.
        /// </summary>
        public double SparseThreshold
        {
            get { return _sparseThreshold; }
            set
            {
                if (value <= 0.0 || value > 1.0)
                    throw new ArgumentOutOfRangeException("sparse_threshold");
                _sparseThreshold = value;
            }
        }

        /// <summary>
        /// Set this to false to disable the special handle of missing value
        /// </summary>
        public bool UseMissing { get; set; } = true;

        // Not supported by managed trees
        // <summary>
        // Set this to true to treat all zero as missing values (including the unshown values in libsvm/sparse matrices)
        // Set this to false to use na for representing missing values
        // </summary>
        //public bool ZeroAsMissing { get; set; } = false;

        // These are only used when LightGBM loading directly from file
        //public bool TwoRound { get; set; } = false;
        //public bool Header { get; set; } = false;
        //public string LabelColumn { get; set; } = "";
        //public string WeightColumn { get; set; } = "";
        //public string GroupColumn { get; set; } = "";
        //public int [] IgnoreColumn { get; set; } = Array.Empty;

        /// <summary>
        /// Used to specify categorical features
        /// Use number for index, e.g.categorical_feature=0,1,2 means column_0, column_1 and column_2 are categorical features
        /// Note: only supports categorical with int type
        /// Note: index starts from 0 and it doesn't count the label column when passing type is int
        /// Note: all values should be less than Int32.MaxValue(2147483647)
        /// Note: using large values could be memory consuming.Tree decision rule works best when categorical features are presented by consecutive integers starting from zero
        /// Note: all negative values will be treated as missing values
        /// </summary>
        public int [] CategoricalFeature { get; set; } = Array.Empty<int>();

        // This is used when constructing the dataset when EnableBundle is true.
        private int _minDataInLeaf = 20;
        /// <summary>
        /// Minimal number of data in one leaf. Can be used to deal with over-fitting.
        /// </summary>
        public int MinDataInLeaf
        {
            get { return _minDataInLeaf; }
            set
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException("MinDataInLeaf");
                _minDataInLeaf = value;
            }
        }

        // https://xgboost.readthedocs.io/en/latest//tutorials/monotonic.html
        /// <summary>
        /// Used for constraints of monotonic features.
        /// 1 means increasing, -1 means decreasing, 0 means non-constraint
        /// You need to specify all features in order. For example, mc = -1,0,1 means decreasing for 1st feature, non-constraint for 2nd feature and increasing for the 3rd feature.
        /// </summary>
        public int [] MonotoneConstraints { get; set; } = Array.Empty<int>();

        /// <summary>
        /// Used to control feature's split gain, will use gain[i] = max(0, feature_contri[i]) * gain[i] to replace the split gain of i-th feature
        /// You need to specify all features in order
        /// </summary>
        public double [] FeatureContri { get; set; } = Array.Empty<double>();

        /// <summary>
        /// Max cache size in MB for historical histogram (< 0 means no limit)
        /// </summary>
        public double HistogramPoolSize { get; set; } = -1.0;

        // These are for CLI only
        //public string OutputModel { get; set; } = "LightGBM_model.txt";
        //public int SnapshotFreq { get; set; } = -1;
        //public string InputModel { get; set; } = "";
        //public string OutputResult { get; set; } = "LightGBM_predict_result.txt";
        //public string InitscoreFilename { get; set; } = "";
        //public string ValidDataInitscores { get; set; } = "";

        /// <summary>
        /// If true, LightGBM will save the dataset (including validation data) to a binary file. This speed ups the data loading for the next time
        /// </summary>
        public bool SaveBinary { get; set; } = false;

        /// <summary>
        /// Set this to true to enable autoloading from previous saved binary datasets
        /// Set this to false to ignore binary datasets
        /// </summary>
        public bool EnableLoadFromBinaryFile { get; set; } = true;

        // These are just for prediction task
        // public bool PredictRawScore { get; set; } = false;
        // public bool PredictLeafIndex { get; set; } = false;
        // public int NumIterationPredict { get; set; } = -1;
        // public bool PredEarlyStop { get; set; } = false;
        // public int PredEarlyStopFreq { get; set; } = 10;
        // public double PredEarlyStopMargin { get; set; } = 10.0;

        // These are just for convert_model task
        // public string ConvertModelLanguage { get; set; } = "";
        // public string ConvertModel { get; set; } = "gbdt_prediction.cpp";       

        #endregion

        public DatasetParameters() : base() { }

        public DatasetParameters(Dictionary<string, string> data)
        {
            FromParameters(data);
        }

        public override void AddParameters(Dictionary<string, string> result)
        {
            _helper.AddParameters(this, result);
        }

    }


    public class LearningParameters : ParametersBase<LearningParameters>
    {
        #region Properties
        // CLI only
        //public string Config { get; set; } = "";
        //public TaskType Task { get; set; } = TaskType.Train;
        //public string Data { get; set; } = "";
        //public string Valid { get; set; } = "";

        public ObjectiveType Objective { get; set; } = ObjectiveType.Regression;

        /// <summary>
        /// Refer to Parallel Learning Guide to get more details
        /// </summary>
        public TreeLearnerType TreeLearner { get; set; } = TreeLearnerType.Serial;

        public BoostingType Boosting { get; set; } = BoostingType.GBDT;

        /// <summary>
        /// Number of boosting iterations
        /// Note: internally, LightGBM constructs num_class * num_iterations trees for multi-class classification problems
        /// </summary>
        private int _numIterations = 100;
        public int NumIterations
        {
            get { return _numIterations; }
            set
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException("NumIterations");
                _numIterations = value;
            }
        }

        private double _learningRate = 0.1;
        /// <summary>
        /// Shrinkage rate
        /// In dart, it also affects on normalization weights of dropped trees
        /// </summary>
        public double LearningRate
        {
            get { return _learningRate; }
            set
            {
                if (value <= 0.0)
                    throw new ArgumentOutOfRangeException("LearningRate");
                _learningRate = value;
            }
        }

        private int _numLeaves = 31;
        /// <summary>
        /// Max number of leaves in one tree
        /// </summary>
        public int NumLeaves
        {
            get { return _numLeaves; }
            set
            {
                if (value < 2)
                    throw new ArgumentOutOfRangeException("NumLeaves");
                _numLeaves = value;
            }
        }

        /// <summary>
        /// Limit the max depth for tree model. This is used to deal with over-fitting when data is small. Tree still grows leaf-wise.
        /// (< 0 means no limit)
        /// </summary>
        public int MaxDepth { get; set; } = -1;

        private double _minSumHessianInLeaf = 1e-3;
        /// <summary>
        /// Minimal sum hessian in one leaf. Like min_data_in_leaf, it can be used to deal with over-fitting
        /// </summary>
        public double MinSumHessianInLeaf
        {
            get { return _minSumHessianInLeaf; }
            set
            {
                if (value < 0.0)
                    throw new ArgumentOutOfRangeException("MinSumHessianInLeaf");
                _minSumHessianInLeaf = value;
            }
        }

        private double _baggingFraction = 1.0;
        /// <summary>
        /// Like feature_fraction, but this will randomly select part of data without resampling
        /// Can be used to speed up training
        /// Can be used to deal with over-fitting
        /// Note: to enable bagging, bagging_freq should be set to a non zero value as well
        /// </summary>
        public double BaggingFraction
        {
            get { return _baggingFraction; }
            set
            {
                if (value <= 0.0 || value > 1.0)
                    throw new ArgumentOutOfRangeException("BaggingFraction");
                _baggingFraction = value;
            }
        }

        /// <summary>
        /// frequency for bagging
        /// 0 means disable bagging; k means perform bagging at every k iteration
        /// Note: to enable bagging, bagging_fraction should be set to value smaller than 1.0 as well
        /// </summary>
        public int BaggingFreq { get; set; } = 0;

        /// <summary>
        /// Random seed for bagging
        /// </summary>
        public int BaggingSeed { get; set; } = 3;

        private double _featureFraction = 1.0;
        /// <summary>
        /// LightGBM will randomly select part of features on each iteration if feature_fraction smaller than 1.0. For example, if you set it to 0.8, LightGBM will select 80% of features before training each tree
        /// Can be used to speed up training
        /// Can be used to deal with over-fitting
        /// </summary>
        public double FeatureFraction
        {
            get { return _featureFraction; }
            set
            {
                if (value <= 0.0 || value > 1.0)
                    throw new ArgumentOutOfRangeException("FeatureFraction");
                _featureFraction = value;
            }
        }

        /// <summary>
        /// Random seed for feature_fraction
        /// </summary>
        public int FeatureFractionSeed { get; set; } = 2;

        /// <summary>
        /// Will stop training if one metric of one validation data doesn't improve in last early_stopping_round rounds.
        /// (<= 0 means disable)
        /// </summary>
        public int EarlyStoppingRound { get; set; } = 0;

        /// <summary>
        /// Used to limit the max output of tree leaves
        /// <= 0 means no constraint
        /// The final max output of leaves is learning_rate* max_delta_step
        /// </summary>
        public double MaxDeltaStep { get; set; } = 0.0;

        private double _lambdaL1 = 0.0;
        /// <summary>
        /// L1 regularization
        /// </summary>
        public double LambdaL1
        {
            get { return _lambdaL1; }
            set
            {
                if (value < 0.0)
                    throw new ArgumentOutOfRangeException("LambdaL1");
                _lambdaL1 = value;
            }
        }

        private double _lambdaL2 = 0.0;
        /// <summary>
        /// L2 regularization
        /// </summary>
        public double LambdaL2
        {
            get { return _lambdaL2; }
            set
            {
                if (value < 0.0)
                    throw new ArgumentOutOfRangeException("LambdaL2");
                _lambdaL2 = value;
            }
        }

        private double _minGainToSplit = 0.0;
        /// <summary>
        /// The minimal gain to perform split
        /// </summary>
        public double MinGainToSplit
        {
            get { return _minGainToSplit; }
            set
            {
                if (value < 0.0)
                    throw new ArgumentOutOfRangeException("MinGainToSplit");
                _minGainToSplit = value;
            }
        }

        private double _dropRate = 0.1;
        /// <summary>
        /// Dropout rate: a fraction of previous trees to drop during the dropout
        /// Used only in dart
        /// </summary>
        public double DropRate
        {
            get { return _dropRate; }
            set
            {
                if (value < 0.0 || value > 1.0)
                    throw new ArgumentOutOfRangeException("DropRate");
                _dropRate = value;
            }
        }

        /// <summary>
        /// Max number of dropped trees during one boosting iteration
        /// Used only in dart
        /// (<=0 means no limit)
        /// </summary>
        public int MaxDrop { get; set; } = 50;

        // Dart only
        private double _skipDrop = 0.5;
        /// <summary>
        /// Probability of skipping the dropout procedure during a boosting iteration
        /// Used only in dart
        /// </summary>
        public double SkipDrop
        {
            get { return _skipDrop; }
            set
            {
                if (value < 0.0 || value > 1.0)
                    throw new ArgumentOutOfRangeException("SkipDrop");
                _skipDrop = value;
            }
        }

        /// <summary>
        /// Set this to true, if you want to use xgboost dart mode
        /// </summary>
        public bool XgboostDartMode { get; set; } = false;

        /// <summary>
        /// Set this to true, if you want to use uniform drop
        /// Used only in dart
        /// </summary>
        public bool UniformDrop { get; set; } = false;

        /// <summary>
        /// Random seed to choose dropping models
        /// Used only in dart
        /// </summary>
        public int DropSeed { get; set; } = 4;

        /// <summary>
        /// The retain ratio of large gradient data
        /// Used only in goss
        /// </summary>
        private double _topRate = 0.2;
        public double TopRate
        {
            get { return _topRate; }
            set
            {
                if (value < 0.0 || value > 1.0)
                    throw new ArgumentOutOfRangeException("TopRate");
                _topRate = value;
            }
        }

        /// <summary>
        /// The retain ratio of small gradient data
        /// Used only in goss
        /// </summary>
        private double _otherRate = 0.1;
        public double OtherRate
        {
            get { return _otherRate; }
            set
            {
                if (value < 0.0 || value > 1.0)
                    throw new ArgumentOutOfRangeException("OtherRate");
                _otherRate = value;
            }
        }

        private int _minDataPerGroup = 100;
        /// <summary>
        /// Minimal number of data per categorical group
        /// </summary>
        public int MinDataPerGroup
        {
            get { return _minDataPerGroup; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("MinDataPerGroup");
                _minDataPerGroup = value;
            }
        }

        private int _maxCatThreshold = 32;
        /// <summary>
        /// Limit the max threshold points in categorical features
        /// Used for the categorical features
        /// </summary>
        public int MaxCatThreshold
        {
            get { return _maxCatThreshold; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("MaxCatThreshold");
                _maxCatThreshold = value;
            }
        }

        private double _catL2 = 10.0;
        /// <summary>
        /// L2 regularization in categorical split
        /// Used for the categorical features
        /// </summary>
        public double CatL2
        {
            get { return _catL2; }
            set
            {
                if (value < 0.0)
                    throw new ArgumentOutOfRangeException("CatL2");
                _catL2 = value;
            }
        }

        private double _catSmooth = 10.0;
        /// <summary>
        /// This can reduce the effect of noises in categorical features, especially for categories with few data
        /// Used for the categorical features
        /// </summary>
        public double CatSmooth
        {
            get { return _catSmooth; }
            set
            {
                if (value < 0.0)
                    throw new ArgumentOutOfRangeException("CatSmooth");
                _catSmooth = value;
            }
        }

        private int _maxCatToOnehot = 4;
        /// <summary>
        /// When number of categories of one feature smaller than or equal to max_cat_to_onehot, one-vs-other split algorithm will be used
        /// </summary>
        public int MaxCatToOnehot
        {
            get { return _maxCatToOnehot; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("MaxCatToOnehot");
                _maxCatToOnehot = value;
            }
        }

        private int _topK = 20;
        /// <summary>
        /// Set this to larger value for more accurate result, but it will slow down the training speed
        /// Used in Voting parallel
        /// </summary>
        public int TopK
        {
            get { return _topK; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("TopK");
                _topK = value;
            }
        }

        // public string ForcedsplitsFilename { get; set; } = "";

        // Used only in refit task
        //private double _refitDecayRate = 0.9;
        //public double RefitDecayRate
        //{
        //    get { return _refitDecayRate; }
        //    set
        //    {
        //        if (value < 0.0 || value > 1.0)
        //            throw new ArgumentOutOfRangeException("RefitDecayRate");
        //        _refitDecayRate = value;
        //    }
        //}

        #endregion

        public LearningParameters() : base() { }

        public LearningParameters(Dictionary<string, string> data)
        {
            FromParameters(data);
        }

        public override void AddParameters(Dictionary<string, string> result)
        {
            _helper.AddParameters(this, result);
        }

    }

    public class ObjectiveParameters : ParametersBase<ObjectiveParameters>
    {
        #region Properties

        private int _numClass = 1;
        /// <summary>
        /// Used only in multi-class classification application
        /// </summary>
        public int NumClass
        {
            get { return _numClass; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("num_class");
                _numClass = value;
            }
        }

        /// <summary>
        /// Used only in binary application
        /// Set this to true if training data are unbalanced
        /// Note: this parameter cannot be used at the same time with scale_pos_weight, choose only one of them
        /// </summary>
        public bool IsUnbalance { get; set; } = false;

        private double _scalePosWeight = 1.0;
        /// <summary>
        /// Used only in binary application
        /// Weight of labels with positive class
        /// Note : this parameter cannot be used at the same time with is_unbalance, choose only one of them
        /// </summary>
        public double ScalePosWeight
        {
            get { return _scalePosWeight; }
            set
            {
                if (value <= 0.0)
                    throw new ArgumentOutOfRangeException("scale_pos_weight");
                _scalePosWeight = value;
            }
        }

        private double _sigmoid = 1.0;
        /// <summary>
        /// Parameter for the sigmoid function
        /// Used only in binary and multiclassova classification and in lambdarank applications
        /// </summary>
        public double Sigmoid
        {
            get { return _sigmoid; }
            set
            {
                if (value <= 0.0)
                    throw new ArgumentOutOfRangeException("sigmoid");
                _sigmoid = value;
            }
        }

        /// <summary>
        /// Adjusts initial score to the mean of labels for faster convergence
        /// Used only in regression, binary and cross-entropy applications
        /// </summary>
        public bool BoostFromAverage { get; set; } = true;

        /// <summary>
        /// Used only in regression application
        /// Used to fit sqrt(label) instead of original values and prediction result will be also automatically converted to prediction^2
        /// Might be useful in case of large-range labels
        /// </summary>
        public bool RegSqrt { get; set; } = false;

        private double _alpha = 0.9;
        /// <summary>
        /// Parameter for Huber loss and Quantile regression
        /// Used only in huber and quantile regression applications
        /// </summary>
        public double Alpha
        {
            get { return _alpha; }
            set
            {
                if (value <= 0.0)
                    throw new ArgumentOutOfRangeException("Alpha");
                _alpha = value;
            }
        }

        private double _fairC = 1.0;
        /// <summary>
        /// Parameter for Fair loss
        /// Used only in fair regression application
        /// </summary>
        public double FairC
        {
            get { return _fairC; }
            set
            {
                if (value <= 0.0)
                    throw new ArgumentOutOfRangeException("FairC");
                _fairC = value;
            }
        }

        private double _poissonMaxDeltaStep = 0.7;
        /// <summary>
        /// Parameter for Poisson regression to safeguard optimization
        /// Used only in poisson regression application
        /// </summary>
        public double PoissonMaxDeltaStep
        {
            get { return _poissonMaxDeltaStep; }
            set
            {
                if (value <= 0.0)
                    throw new ArgumentOutOfRangeException("PoissonMaxDeltaStep");
                _poissonMaxDeltaStep = value;
            }
        }

        private double _tweedieVariancePower = 1.5;
        /// <summary>
        /// Used only in tweedie regression application
        /// Used to control the variance of the tweedie distribution
        /// Set this closer to 2 to shift towards a Gamma distribution
        /// Set this closer to 1 to shift towards a Poisson distribution
        /// </summary>
        public double TweedieVariancePower
        {
            get { return _tweedieVariancePower; }
            set
            {
                if (value < 1.0 || value >= 2.0)
                    throw new ArgumentOutOfRangeException("TweedieVariancePower");
                _tweedieVariancePower = value;
            }
        }

        private int _maxPosition = 20;
        /// <summary>
        /// Optimizes NDCG at this position
        /// Used only in lambdarank application
        /// </summary>
        public int MaxPosition
        {
            get { return _maxPosition; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("max_position");
                _maxPosition = value;
            }
        }

        /// <summary>
        /// Used only in lambdarank application.
        /// Relevant gain for labels.
        /// For example, the gain of label 2 is 3 in case of default label gains.
        /// Default = 0,1,3,7,15,31,63,...,2^30-1
        /// </summary>
        public double[] LabelGain { get; set; } = _defLabelGain;

        #endregion

        private static double[] _defLabelGain;
        static ObjectiveParameters()
        {
            _defLabelGain = new double[31];
            for (int i = 0; i < 31; ++i)
                _defLabelGain[i] = (double)((1L << i) - 1L);
        }

        public ObjectiveParameters() : base()
        {
        }

        public ObjectiveParameters(Dictionary<string, string> data)
        {
            FromParameters(data);
        }

        public override void AddParameters(Dictionary<string, string> result)
        {
            _helper.AddParameters(this, result);
        }
    }

    /// <summary>
    /// Parameters describing the metric <see cref="https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst#metric-parameters"/>
    /// </summary>
    public class MetricParameters : ParametersBase<MetricParameters>
    {
        #region Properties

        private MetricType _metric = MetricType.DefaultMetric;
        /// <summary>
        /// Metric to be evaluated on the evaluation sets
        /// </summary>
        public MetricType Metric
        {
            get { return _metric; }
            set
            {
                if (!Enum.IsDefined(typeof(MetricType), (int)value))
                    throw new ArgumentOutOfRangeException("Metric invalid");
                _metric = value;
            }
        }

        // TODO: this is only used if you call the LightGBM Train method (instead we are calling BoosterUpdateOneIter in a loop)
        // Set to zero to disable printing of metrics
        private int _metricFreq = 1;
        /// <summary>
        /// Frequency for metric output
        /// </summary>
        public int MetricFreq
        {
            get { return _metricFreq; }
            set
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException("MetricFreq");
                _metricFreq = value;
            }
        }

        // CLI only
        // public bool IsProvideTrainingMetric { get; set; } = false;

        private static readonly int[] _defEvalAt = new int[] { 1, 2, 3, 4, 5 };
        /// <summary>
        /// NDCG and MAP evaluation positions.
        /// Used only with ndcg and map metrics.
        /// </summary>
        public int[] EvalAt { get; set; } = _defEvalAt;

        #endregion


        public MetricParameters()
        {
        }

        public MetricParameters(Dictionary<string, string> data)
        {
            FromParameters(data);
        }

        public override void AddParameters(Dictionary<string, string> result)
        {
            _helper.AddParameters(this, result);
        }
    }

    /// <summary>
    /// Contains both network, verbosity and threading parameters.
    /// </summary>
    public class CommonParameters : ParametersBase<CommonParameters>
    {
        /// <summary>
        /// This seed is used to generate other seeds, e.g. data_random_seed, feature_fraction_seed.
        /// Will be overridden, if you set other seeds.
        /// </summary>
        public int Seed { get; set; } = 0;

        // Note, persisted as an int because we expect the file to contain -1,0,or 1, not strings.
        public VerbosityType Verbosity { get; set; } = VerbosityType.Info;
        
        #region Network
        private int _numMachines = 1;
        /// <summary>
        /// The number of machines for parallel learning application.
        /// This parameter is needed to be set in both socket and mpi versions
        /// </summary>
        public int NumMachines
        {
            get { return _numMachines; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("NumMachines");
                _numMachines = value;
            }
        }

        private int _localListenPort = 12400;
        /// <summary>
        /// TCP listen port for local machines
        /// Note: don't forget to allow this port in firewall settings before training
        /// </summary>
        public int LocalListenPort
        {
            get { return _localListenPort; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("LocalListenPort");
                _localListenPort = value;
            }
        }

        private int _timeOut = 120;
        /// <summary>
        /// Socket time-out in minutes
        /// </summary>
        public int TimeOut
        {
            get { return _timeOut; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("TimeOut");
                _timeOut = value;
            }
        }

        /// <summary>
        /// Path of file that lists machines for this parallel learning application.
        /// Each line contains one IP and one port for one machine. The format is ip port (space as a separator)
        /// </summary>
        public string MachineListFilename { get; set; } = "";

        /// <summary>
        /// List of machines in the following format: ip1:port1,ip2:port2
        /// </summary>
        public string Machines { get; set; } = "";
        #endregion

        /// <summary>
        /// number of threads for LightGBM
        /// 0 means default number of threads in OpenMP
        /// for the best speed, set this to the number of real CPU cores, not the number of threads (most CPUs use hyper-threading to generate 2 threads per CPU core)
        /// do not set it too large if your dataset is small (for instance, do not use 64 threads for a dataset with 10,000 rows)
        /// be aware a task manager or any similar CPU monitoring tool might report that cores not being fully utilized. This is normal
        /// for parallel learning, do not use all CPU cores because this will cause poor performance for the network communication
        /// </summary>
        public int NumThreads { get; set; } = 0;

        /// <summary>
        /// Device for the tree learning, you can use GPU to achieve the faster learning
        /// Note: it is recommended to use the smaller max_bin (e.g. 63) to get the better speed up
        /// Note: for the faster speed, GPU uses 32-bit float point to sum up by default, so this may affect the accuracy for some tasks.You can set gpu_use_dp=true to enable 64-bit float point, but it will slow down the training
        /// Note: refer to Installation Guide to build LightGBM with GPU support
        /// </summary>
        public DeviceType DeviceType { get; set; } = DeviceType.CPU;

        #region GPU
        /// <summary>
        /// OpenCL platform ID. Usually each GPU vendor exposes one OpenCL platform.
        /// -1 means the system-wide default platform.
        /// </summary>
        public int GPUPlatformId { get; set; } = -1;

        /// <summary>
        /// OpenCL device ID in the specified platform. Each GPU in the selected platform has a unique device ID.
        /// -1 means the default device in the selected platform
        /// </summary>
        public int GPUDeviceId { get; set; } = -1;

        /// <summary>
        /// Set this to true to use double precision math on GPU (by default single precision is used).
        /// </summary>
        public bool GPUUseDP { get; set; } = false;
        #endregion

        public CommonParameters(Dictionary<string, string> data)
        {
            FromParameters(data);
        }

        public override void AddParameters(Dictionary<string, string> result)
        {
            _helper.AddParameters(this, result);
        }

        public CommonParameters()
        {
        }
    }
  

    public class Parameters
    {
        // these two impact dataset construction
        public CommonParameters Common { get; set; }
        public DatasetParameters Dataset { get; set; }

        // these don't
        public MetricParameters Metric { get; set; }
        public ObjectiveParameters Objective { get; set; }
        public LearningParameters Learning { get; set; }

        public Parameters()
        {
            Common = new CommonParameters();
            Dataset = new DatasetParameters();
            Metric = new MetricParameters();
            Objective = new ObjectiveParameters();
            Learning = new LearningParameters();
        }

        public Parameters(string v)
        {
            var dict = ParamsHelper.SplitParameters(v);
            Common = CommonParameters.FromParameters(dict);
            Dataset = DatasetParameters.FromParameters(dict);
            Metric = MetricParameters.FromParameters(dict);
            Objective = ObjectiveParameters.FromParameters(dict);
            Learning = LearningParameters.FromParameters(dict);
        }

        public override string ToString()
        {
            var dict = new Dictionary<string, string>();
            Common.AddParameters(dict);
            Dataset.AddParameters(dict);
            Metric.AddParameters(dict);
            Objective.AddParameters(dict);
            Learning.AddParameters(dict);
            return ParamsHelper.JoinParameters(dict);
        }
    }
}