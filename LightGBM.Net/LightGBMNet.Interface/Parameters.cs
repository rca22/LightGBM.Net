﻿using System;
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
        Regression = 0,
        RegressionL1 = 1,
        Huber = 2,
        Fair = 3,
        Poisson = 4,
        Quantile = 5,
        Mape = 6,
        Gamma = 7,
        Tweedie = 8,
        Binary = 9,
        MultiClass = 10,
        MultiClassOva = 11,
        XEntropy = 12,
        XEntLambda = 13,
        LambdaRank = 14
    }

    public enum BoostingType : int
    {
        GBDT,
        RandomForest,
        Dart,
        Goss
    }

    public enum TreeLearnerType
    {
        Serial,
        Feature,
        Data,
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
                case ObjectiveType.Quantile:     return "quartile";
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
                case "quartile":      return ObjectiveType.Quantile;
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

    public class CoreParameters : ParametersBase<CoreParameters>
    {
        #region Properties

        public string Config { get; set; } = "";

        public TaskType Task { get; set; } = TaskType.Train;

        public ObjectiveType Objective { get; set; } = ObjectiveType.Regression;

        public BoostingType Boosting { get; set; } = BoostingType.GBDT;

        public string Data { get; set; } = "";

        public string Valid { get; set; } = "";

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

        public TreeLearnerType TreeLearner { get; set; } = TreeLearnerType.Serial;

        public int NumThreads { get; set; } = 0;

        public DeviceType DeviceType { get; set; } = DeviceType.CPU;

        public int Seed { get; set; } = 0;

        #endregion

        public CoreParameters() : base() { }

        public CoreParameters(Dictionary<string, string> data)
        {
            FromParameters(data);
        }

        public override void AddParameters(Dictionary<string, string> result)
        {
            _helper.AddParameters(this, result);
        }
    }

    public class LearningControlParameters : ParametersBase<LearningControlParameters>
    {
        #region Properties

        public int MaxDepth { get; set; } = -1;

        private int _minDataInLeaf = 20;
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

        private double _minSumHessianInLeaf = 1e-3;
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

        private double _baggingFraction = 1e-3;
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

        public int BaggingFreq { get; set; } = 0;

        public int BaggingSeed { get; set; } = 3;

        private double _featureFraction = 1.0;
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

        public int FeatureFractionSeed { get; set; } = 2;

        public int EarlyStoppingRound { get; set; } = 0;

        public double MaxDeltaStep { get; set; } = 0.0;

        private double _lambdaL1 = 0.0;
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

        // Dart only
        private double _dropRate = 0.1;
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

        public int MaxDrop { get; set; } = 50;

        // Dart only
        private double _skipDrop = 0.5;
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

        // set to true to use Dart mode
        public bool XgboostDartMode { get; set; } = false;

        // dart only
        public bool UniformDrop { get; set; } = false;

        // dart only
        public int DropSeed { get; set; } = 4;

        // goss only
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

        // goss only
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

        //TODO multi-int
        public string MonotoneConstraints { get; set; } = "None";

        //TODO multi-double
        public string FeatureContri { get; set; } = "None";

        public string ForcedsplitsFilename { get; set; } = "";

        private double _refitDecayRate = 0.9;
        public double RefitDecayRate
        {
            get { return _refitDecayRate; }
            set
            {
                if (value < 0.0 || value > 1.0)
                    throw new ArgumentOutOfRangeException("RefitDecayRate");
                _refitDecayRate = value;
            }
        }

        #endregion

        public LearningControlParameters() : base() { }

        public LearningControlParameters(Dictionary<string, string> data)
        {
            FromParameters(data);
        }

        public override void AddParameters(Dictionary<string, string> result)
        {
            _helper.AddParameters(this, result);
        }

    }

    public class IOParameters : ParametersBase<IOParameters>
    {
        #region Properties

        // Note, persisted as an int because we expect the file to contain -1,0,or 1, not strings.
        public int Verbosity { get; set; } = (int)VerbosityType.Info;
        public void SetVerbosity(VerbosityType vt) { Verbosity = (int)vt; }

        private int _maxBin = 255;
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

        public double HistogramPoolSize { get; set; } = -1.0;

        public double DataRandomSeed { get; set; } = 1;

        public string OutputModel { get; set; } = "LightGBM_model.txt";

        public int SnapshotFreq { get; set; } = -1;

        public string InputModel { get; set; } = "";

        public string OutputResult { get; set; } = "LightGBM_predict_result.txt";

        public string InitscoreFilename { get; set; } = "";

        public string ValidDataInitscores { get; set; } = "";

        public bool EnableBundle { get; set; } = true;

        private double _maxConflictRate = 0.0;
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

        public bool IsEnableSparse { get; set; } = true;

        private double _sparseThreshold = 0.8;
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

        public bool UseMissing { get; set; } = true;

        public bool ZeroAsMissing { get; set; } = false;

        public bool TwoRound { get; set; } = false;

        public bool SaveBinary { get; set; } = false;

        public bool EnableLoadFromBinaryFile { get; set; } = true;

        public bool Header { get; set; } = false;

        public string LabelColumn { get; set; } = "";

        public string WeightColumn { get; set; } = "";

        public string GroupColumn { get; set; } = "";

        //TODO
        public string IgnoreColumn { get; set; } = "";

        //TODO
        public string CategoricalFeature { get; set; } = "";

        public bool PredictRawScore { get; set; } = false;

        public bool PredictLeafIndex { get; set; } = false;

        public int NumIterationPredict { get; set; } = -1;

        public bool PredEarlyStop { get; set; } = false;

        public int PredEarlyStopFreq { get; set; } = 10;

        public double PredEarlyStopMargin { get; set; } = 10.0;

        public string ConvertModelLanguage { get; set; } = "";

        public string ConvertModel { get; set; } = "gbdt_prediction.cpp";

        #endregion

        public IOParameters() : base() {}

        public IOParameters (Dictionary<string, string> data)
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

        public bool IsUnbalance { get; set; } = false;

        private double _scalePosWeight = 1.0;
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

        public bool BoostFromAverage { get; set; } = true;

        public bool RegSqrt { get; set; } = false;

        private double _alpha = 0.9;
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

        private int _metricFreq = 1;
        public int MetricFreq
        {
            get { return _metricFreq; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("MetricFreq");
                _metricFreq = value;
            }
        }

        public bool IsProvidingTrainingMetric { get; set; } = false;

        private static readonly int[] _defEvalAt = new int[] { 1, 2, 3, 4, 5 };
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

    public class NetworkParameters : ParametersBase<NetworkParameters>
    {
        private int _numMachines = 1;
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

        public string MachineListFilename { get; set; } = "";

        public string Machines { get; set; } = "";

        public NetworkParameters(Dictionary<string, string> data)
        {
            FromParameters(data);
        }

        public override void AddParameters(Dictionary<string, string> result)
        {
            _helper.AddParameters(this, result);
        }

        public NetworkParameters()
        {
        }
    }

    public class GPUParameters : ParametersBase<GPUParameters>
    {
        public int GPUPlatformId { get; set; } = -1;

        public int GPUDeviceId { get; set; } = -1;

        public bool GPUUseDP { get; set; } = false;

        public GPUParameters()
        {
        }

        public GPUParameters(Dictionary<string, string> data)
        {
            FromParameters(data);
        }

        public override void AddParameters(Dictionary<string, string> result)
        {
            _helper.AddParameters(this, result);
        }
    }

    public class Parameters
    {
        public GPUParameters GPU { get; set; }
        public NetworkParameters Network { get; set; }
        public MetricParameters Metric { get; set; }
        public ObjectiveParameters Objective { get; set; }
        public LearningControlParameters Learning { get; set; }
        public IOParameters IO { get; set; }
        public CoreParameters Core { get; set; }

        public Parameters()
        {
            GPU = new GPUParameters();
            Network = new NetworkParameters();
            Metric = new MetricParameters();
            Objective = new ObjectiveParameters();
            Learning = new LearningControlParameters();
            IO = new IOParameters();
            Core = new CoreParameters();
        }

        public Parameters(string v)
        {
            var dict = ParamsHelper.SplitParameters(v);
            GPU = GPUParameters.FromParameters(dict);
            Network = NetworkParameters.FromParameters(dict);
            Metric = MetricParameters.FromParameters(dict);
            Objective = ObjectiveParameters.FromParameters(dict);
            Learning = LearningControlParameters.FromParameters(dict);
            IO = IOParameters.FromParameters(dict);
            Core = CoreParameters.FromParameters(dict);
        }

        public override string ToString()
        {
            var dict = new Dictionary<string, string>();
            GPU.AddParameters(dict);
            Network.AddParameters(dict);
            Metric.AddParameters(dict);
            Objective.AddParameters(dict);
            Learning.AddParameters(dict);
            IO.AddParameters(dict);
            Core.AddParameters(dict);
            return ParamsHelper.JoinParameters(dict);
        }
    }
}