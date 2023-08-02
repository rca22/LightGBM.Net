using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("LightGBMNet.Train.Test")]

namespace LightGBMNet.Train
{
    //public enum TaskType : int
    //{
    //    Train = 0,
    //    Predict = 1,
    //    ConvertModel = 2,
    //    Refit = 3
    //}

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
        XEntropy = 12,      // TODO
        XEntLambda = 13,    // TODO
        // ranking
        LambdaRank = 14,
        RankXendcg = 15
    }

    public enum BoostingType : int
    {
        /// Traditional Gradient Boosting Decision Tree
        GBDT,
        /// Random Forest
        RandomForest,
        /// Dropouts meet Multiple Additive Regression Trees
        Dart
    }

    public enum DataSampleStrategyType : int
    {
        /// Randomly Bagging Sampling
        Bagging,
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
        GPU,
        CUDA
    }

    public enum MonotoneConstraintsMethod
    {
        /// basic, the most basic monotone constraints method. It does not slow the library at all, but over-constrains the predictions
        Basic,
        /// intermediate, a more advanced method, which may slow the library very slightly. However, this method is much less constraining than the basic method and should significantly improve the results
        Intermediate,
        /// an even more advanced method which may slow the library. However, this method is even less constraining than the intermediate method and should again significantly improve the results.
        Advanced
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
        /// <summary>
        /// Average precision score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html>
        /// </summary>
        AveragePrecision,
        /// <summary>
        /// Log loss <https://en.wikipedia.org/wiki/Cross_entropy>
        /// </summary>
        BinaryLogLoss,
        /// <summary>
        /// for one sample: 0 for correct classification, 1 for error classification
        /// </summary>
        BinaryError,
        /// AUC-mu
        AucMu,
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
                case MetricType.AveragePrecision:
                    return "average_precision";
                case MetricType.BinaryLogLoss:
                    return "binary_logloss";
                case MetricType.BinaryError:
                    return "binary_error";
                case MetricType.AucMu:
                    return "auc_mu";
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
                    throw new Exception(string.Format("Unexpected value of MetricType {0}",m));
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
                case "average_precision":
                    return MetricType.AveragePrecision;
                case "binary_logloss":
                case "binary":
                    return MetricType.BinaryLogLoss;
                case "binary_error":
                    return MetricType.BinaryError;
                case "auc_mu":
                    return MetricType.AucMu;
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
                case "cross_entropy_lambda":
                    return MetricType.XentLambda;
                case "kldiv":
                case "kullback_leibler":
                    return MetricType.Kldiv;
                default:
                    throw new Exception(string.Format("Unexpected value of MetricType {0}",x));
            }
        }

        //public static string GetTaskString(TaskType t)
        //{
        //    switch(t)
        //    {
        //        case TaskType.Train:        return "train";
        //        case TaskType.Predict:      return "predict";
        //        case TaskType.ConvertModel: return "convert_model";
        //        case TaskType.Refit:        return "refit";
        //        default:
        //            throw new ArgumentException("TaskType");
        //    }
        //}

        //public static TaskType ParseTask(string x)
        //{
        //    switch(x)
        //    {
        //        case "train":
        //        case "training":
        //            return TaskType.Train;
        //        case "predict":
        //        case "prediction":
        //        case "test":
        //            return TaskType.Predict;
        //        case "convert_model":
        //            return TaskType.ConvertModel;
        //        case "refit":
        //        case "refit_tree":
        //            return TaskType.Refit;
        //        default:
        //            throw new ArgumentException("x");
        //    }
        //}

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
                case ObjectiveType.RankXendcg:   return "rank_xendcg";
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
                case "rank_xendcg": return ObjectiveType.RankXendcg;
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
                default:
                    throw new ArgumentException("BoostingType not found");
            }
        }

        public static string GetDataSampleStrategyString(DataSampleStrategyType t)
        {
            switch (t)
            {
                case DataSampleStrategyType.Bagging: return "bagging";
                case DataSampleStrategyType.Goss: return "goss";
                default:
                    throw new ArgumentException("DataSampleStrategyType not found");
            }
        }

        public static DataSampleStrategyType ParseDataSampleStrategy(string x)
        {
            switch (x)
            {
                case "bagging":
                    return DataSampleStrategyType.Bagging;
                case "goss":
                    return DataSampleStrategyType.Goss;
                default:
                    throw new ArgumentException("DataSampleStrategyType not found");
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
                case DeviceType.CUDA: return "cuda";
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
                case "cuda": return DeviceType.CUDA;
                default:
                    throw new ArgumentException("DeviceType not recognised");
            }
        }
        
        public static string GetMonotoneConstraintsMethodString(MonotoneConstraintsMethod d)
        {
            switch (d)
            {
                case MonotoneConstraintsMethod.Basic: return "basic";
                case MonotoneConstraintsMethod.Intermediate: return "intermediate";
                case MonotoneConstraintsMethod.Advanced: return "advanced";
                default:
                    throw new ArgumentException("MonotoneConstraintsMethod not recognised");
            }
        }

        public static MonotoneConstraintsMethod ParseMonotoneConstraintsMethod(string x)
        {
            switch (x)
            {
                case "basic": return MonotoneConstraintsMethod.Basic;
                case "intermediate": return MonotoneConstraintsMethod.Intermediate;
                case "advanced": return MonotoneConstraintsMethod.Advanced;
                default:
                    throw new ArgumentException("MonotoneConstraintsMethod not recognised");
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
            if (typ == typeof(int))
                return x => (object)int.Parse(x);
            else if (typ == typeof(long))
                return x => (object)long.Parse(x);
            else if (typ == typeof(double))
                return x => (object)double.Parse(x);
            else if (typ == typeof(float))
                return x => (object)float.Parse(x);
            else if (typ == typeof(string))
                return x => (object)x;
            else if (typ == typeof(bool))
                return x =>
                    {
                        if (!bool.TryParse(x, out bool rslt))
                            rslt = Convert.ToBoolean(int.Parse(x));
                        return (object)rslt;
                    };
            else if (typ == typeof(int[]))
                return x => x.Split(new char[] { ',' }).Select(int.Parse).ToArray();
            else if (typ == typeof(double[]))
                return x => x.Split(new char[] { ',' }).Select(double.Parse).ToArray();
            else if (typ == typeof(MetricType))
                return x => (object)EnumHelper.ParseMetric(x);
          //else if (typ == typeof(TaskType))
          //    return x => (object)EnumHelper.ParseTask(x);
            else if (typ == typeof(ObjectiveType))
                return x => (object)EnumHelper.ParseObjective(x);
            else if (typ == typeof(BoostingType))
                return x => (object)EnumHelper.ParseBoosting(x);
            else if (typ == typeof(DataSampleStrategyType))
                return x => (object)EnumHelper.ParseDataSampleStrategy(x);
            else if (typ == typeof(TreeLearnerType))
                return x => (object)EnumHelper.ParseTreeLearner(x);
            else if (typ == typeof(DeviceType))
                return x => (object)EnumHelper.ParseDevice(x);
            else if (typ == typeof(VerbosityType))
                return x => (object)EnumHelper.ParseVerbosity(x);
            else if (typ == typeof(MonotoneConstraintsMethod))
                return x => (object)EnumHelper.ParseMonotoneConstraintsMethod(x);
            else
                throw new Exception(string.Format("Unhandled parameter type {0}", typ));
        }

        private static WriteFunction CreateWriteFunction(Type typ)
        {
            if (typ == typeof(int))
                return x => x.ToString();
            else if (typ == typeof(long))
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
                return x => string.Join(",", (x as int[]).Select(y => y.ToString()));
            else if (typ == typeof(double[]))
                return x => string.Join(",", (x as double[]).Select(y => y.ToString()));
            else if (typ == typeof(MetricType))
                return x => EnumHelper.GetMetricString((MetricType)x);
          //else if (typ == typeof(TaskType))
          //    return x => EnumHelper.GetTaskString((TaskType)x);
            else if (typ == typeof(ObjectiveType))
                return x => EnumHelper.GetObjectiveString((ObjectiveType)x);
            else if (typ == typeof(BoostingType))
                return x => EnumHelper.GetBoostingString((BoostingType)x);
            else if (typ == typeof(DataSampleStrategyType))
                return x => EnumHelper.GetDataSampleStrategyString((DataSampleStrategyType)x);
            else if (typ == typeof(TreeLearnerType))
                return x => EnumHelper.GetTreeLearnerString((TreeLearnerType)x);
            else if (typ == typeof(DeviceType))
                return x => EnumHelper.GetDeviceString((DeviceType)x);
            else if (typ == typeof(VerbosityType))
                return x => EnumHelper.GetVerbosityString((VerbosityType)x);
            else if (typ == typeof(MonotoneConstraintsMethod))
                return x => EnumHelper.GetMonotoneConstraintsMethodString((MonotoneConstraintsMethod)x);
            else
                throw new Exception(string.Format("Unhandled parameter type {0}", typ));
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

        private static bool Equal(object x, object y)
        {
            if (x == null && y == null) return true;
            if (x == null || y == null) return false;
            if (x.GetType() != y.GetType()) return false;
            if (x.GetType().IsArray) return EqualArrays((Array)x, (Array)x);
            return x.Equals(y);
        }

        private static bool EqualArrays(Array x, Array y)
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

        public bool Equal(T x, T y)
        {
            return _propToArgNameAndDefault.Keys.All(prop =>
                Equal(prop.GetValue(x), prop.GetValue(y))
                );
        }

#pragma warning disable IDE0041 // Use 'is null' check

        public int GetHashCode(T x)
        {
            unchecked
            {
                int hash = 13;
                foreach (var prop in _propToArgNameAndDefault.Keys)
                {
                    hash *= 7;
                    var value = prop.GetValue(x);
                    if (!ReferenceEquals(null, value)) {
                        if (prop.PropertyType.IsArray)
                            hash += ((IStructuralEquatable)value).GetHashCode(StructuralComparisons.StructuralEqualityComparer);
                        else
                            hash += value.GetHashCode();
                    }
                    //Console.WriteLine($"{prop.Name} {value.GetHashCode()} {hash}");
                }
                return hash;
            }
        }

        public T FromParameters(Dictionary<string,string> pms)
        {
            var rslt = new T();
            Tuple<PropertyInfo,ParseFunction> pair = null;
            var keys = pms.Keys.ToArray();
            foreach (var key in keys)
            {
                if (_argToProp.TryGetValue(key, out pair))
                {
                    var value = pms[key];
                    try
                    {
                        var prop = pair.Item1;
                        var parser = pair.Item2;
                        var obj = parser(value);
                        prop.SetValue(rslt, obj);
                        pms.Remove(key);
                    }
                    catch (Exception e)
                    {
                        throw new FormatException($"Cannot parse '{value}' for parameter {key}", e);
                    }
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

        /// <summary>
        /// Used to enable/disable sparse optimization
        /// </summary>
        public bool IsEnableSparse { get; set; } = true;

        /// <summary>
        /// Set this to false to disable the special handle of missing value
        /// </summary>
        public bool UseMissing { get; set; } = true;

        /// <summary>
        /// Use precise floating point number parsing for text parser (e.g. CSV, TSV, LibSVM input).
        /// Note: setting this to true may lead to much slower text parsing.
        /// </summary>
        public bool PreciseFloatParser { get; set; } = false;

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

        /// <summary>
        /// Set this to true to pre-filter the unsplittable features by min_data_in_leaf
        /// </summary>
        public bool FeaturePreFilter { get; set; } = true;

        /// <summary>
        /// Set this to true if data file is too big to fit in memory.
        /// By default, LightGBM will map data file to memory and load features from memory.This will provide faster data loading speed, but may cause run out of memory error when the data file is very big.
        /// Note: works only in case of loading data directly from file.
        /// </summary>
        public bool TwoRound { get; set; } = false;

        /// <summary>
        /// Set this to true if input data has header.
        /// Note: works only in case of loading data directly from file.
        /// </summary>
        public bool Header { get; set; } = false;

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

        #region Structural equality overrides
        public override int GetHashCode() => _helper.GetHashCode(this);

        public override bool Equals(object value)
        {
            if (ReferenceEquals(null, value)) return false;
            if (ReferenceEquals(this, value)) return true;
            if (value.GetType() != this.GetType()) return false;
            return _helper.Equal(this, (DatasetParameters)value);
        }

        public static bool operator ==(DatasetParameters t, DatasetParameters other)
        {
            if (!ReferenceEquals(null, t)) return t.Equals(other);
            else if (!ReferenceEquals(null, other)) return other.Equals(t);
            else return true;
        }

        public static bool operator !=(DatasetParameters t, DatasetParameters other)
        {
            return !(t == other);
        }
        #endregion
    }


    public class LearningParameters : ParametersBase<LearningParameters>
    {
        #region Properties
        // CLI only
        //public string Config { get; set; } = "";
        //public TaskType Task { get; set; } = TaskType.Train;
        //public string Data { get; set; } = "";
        //public string Valid { get; set; } = "";

        /// <summary>
        /// Refer to Parallel Learning Guide to get more details
        /// </summary>
        public TreeLearnerType TreeLearner { get; set; } = TreeLearnerType.Serial;

        public BoostingType Boosting { get; set; } = BoostingType.GBDT;

        public DataSampleStrategyType DataSampleStrategy { get; set; } = DataSampleStrategyType.Bagging;

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

        /// <summary>
        /// Set this to true to force col-wise histogram building
        /// </summary>
        public bool ForceColWise { get; set; } = false;

        /// <summary>
        /// Set this to true to force row-wise histogram building
        /// </summary>
        public bool ForceRowWise { get; set; } = false;

        private double _pos_bagging_fraction = 1.0;
        /// <summary>
        /// Used for imbalanced binary classification problem, will randomly sample #pos_samples * pos_bagging_fraction positive samples in bagging
        /// </summary>
        public double PosBaggingFraction
        {
            get { return _pos_bagging_fraction; }
            set
            {
                if (value <= 0.0 || value > 1.0)
                    throw new ArgumentOutOfRangeException("PosBaggingFraction");
                _pos_bagging_fraction = value;
            }
        }

        private double _neg_bagging_fraction = 1.0;
        /// <summary>
        /// Used for imbalanced binary classification problem, will randomly sample #neg_samples * neg_bagging_fraction positive samples in bagging
        /// </summary>
        public double NegBaggingFraction
        {
            get { return _neg_bagging_fraction; }
            set
            {
                if (value <= 0.0 || value > 1.0)
                    throw new ArgumentOutOfRangeException("NegBaggingFraction");
                _neg_bagging_fraction = value;
            }
        }

        private double _feature_fraction_bynode = 1.0;
        /// <summary>
        /// LightGBM will randomly select part of features on each tree node if feature_fraction_bynode smaller than 1.0. For example, if you set it to 0.8, LightGBM will select 80% of features at each tree node
        /// </summary>
        public double FeatureFractionBynode
        {
            get { return _feature_fraction_bynode; }
            set
            {
                if (value <= 0.0 || value > 1.0)
                    throw new ArgumentOutOfRangeException("FeatureFractionBynode");
                _feature_fraction_bynode = value;
            }
        }

        /// <summary>
        /// Use extremely randomized trees? if set to true, when evaluating node splits LightGBM will check only one randomly-chosen threshold for each feature.
        /// </summary>
        public bool ExtraTrees { get; set; } = false;

        private int _extra_seed = 6;
        /// <summary>
        /// Random seed for selecting thresholds when extra_trees is true
        /// </summary>
        public int ExtraSeed
        {
            get { return _extra_seed; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("ExtraSeed");
                _extra_seed = value;
            }
        }

        /// <summary>
        /// Set this to true, if you want to use only the first metric for early stopping
        /// </summary>
        public bool FirstMetricOnly { get; set; } = false;

        /// <summary>
        /// Monotone constraints method.  Used only if monotone_constraints is set.
        /// </summary>
        public MonotoneConstraintsMethod MonotoneConstraintsMethod { get; set; } = MonotoneConstraintsMethod.Basic;

        private double _monotone_penalty = 0.0;
        /// <summary>
        /// A penalization parameter X forbids any monotone splits on the first X (rounded down) level(s) of the tree.
        /// The penalty applied to monotone splits on a given depth is a continuous, increasing function the penalization parameter.
        /// </summary>
        public double MonotonePenalty
        {
            get { return _monotone_penalty; }
            set
            {
                if (value < 0.0)
                    throw new ArgumentOutOfRangeException("MonotonePenalty");
                _monotone_penalty = value;
            }
        }

        private double _refit_decay_rate = 0.9;
        /// <summary>
        /// Decay rate of refit task, will use leaf_output = refit_decay_rate * old_leaf_output + (1.0 - refit_decay_rate) * new_leaf_output to refit trees
        /// </summary>
        public double RefitDecayRate
        {
            get { return _refit_decay_rate; }
            set
            {
                if (value < 0.0 || value > 1.0)
                    throw new ArgumentOutOfRangeException("RefitDecayRate");
                _refit_decay_rate = value;
            }
        }

        private double _cegb_tradeoff = 1.0;
        /// <summary>
        /// Cost-effective gradient boosting multiplier for all penalties
        /// </summary>
        public double CegbTradeoff
        {
            get { return _cegb_tradeoff; }
            set
            {
                if (value < 0.0)
                    throw new ArgumentOutOfRangeException("CegbTradeoff");
                _cegb_tradeoff = value;
            }
        }

        private double _cegb_penalty_split = 0.0;
        /// <summary>
        /// Cost-effective gradient-boosting penalty for splitting a node
        /// </summary>
        public double CegbPenaltySplit
        {
            get { return _cegb_penalty_split; }
            set
            {
                if (value < 0.0)
                    throw new ArgumentOutOfRangeException("CegbPenaltySplit");
                _cegb_penalty_split = value;
            }
        }

        private double _path_smooth = 0.0;
        /// <summary>
        /// Controls smoothing applied to tree nodes, helps prevent overfitting on leaves with few samples. 
        /// If set to zero, no smoothing is applied.  If path_smooth > 0 then min_data_in_leaf must be at least 2.
        /// Larger values give stronger regularisation.
        /// The weight of each node is (n / path_smooth) * w + w_p / (n / path_smooth + 1), where n is the number of samples in the node, w is the optimal node weight to minimise the loss(approximately -sum_gradients / sum_hessians), and w_p is the weight of the parent node.
        /// Note that the parent output w_p itself has smoothing applied, unless it is the root node, so that the smoothing effect accumulates with the tree depth.
        /// </summary>
        public double PathSmooth
        {
            get { return _path_smooth; }
            set
            {
                if (value < 0.0)
                    throw new ArgumentOutOfRangeException("PathSmooth");
                _path_smooth = value;
            }
        }

        private double _linear_lambda = 0.0;
        /// <summary>
        /// Linear tree regularization, corresponds to the parameter lambda in Eq. 3 of Gradient Boosting with Piece-Wise Linear Regression Trees.
        /// </summary>
        public double LinearLambda
        {
            get { return _linear_lambda; }
            set
            {
                if (value < 0.0)
                    throw new ArgumentOutOfRangeException("LinearLambda");
                _linear_lambda = value;
            }
        }

        // TODO: add interaction_constraints

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

        #region Structural equality overrides
        public override int GetHashCode() => _helper.GetHashCode(this);

        public override bool Equals(object value)
        {
            if (ReferenceEquals(null, value)) return false;
            if (ReferenceEquals(this, value)) return true;
            if (value.GetType() != this.GetType()) return false;
            return _helper.Equal(this, (LearningParameters)value);
        }

        public static bool operator ==(LearningParameters t, LearningParameters other)
        {
            if (!ReferenceEquals(null, t)) return t.Equals(other);
            else if (!ReferenceEquals(null, other)) return other.Equals(t);
            else return true;
        }

        public static bool operator !=(LearningParameters t, LearningParameters other)
        {
            return !(t == other);
        }
        #endregion
    }

    public class ObjectiveParameters : ParametersBase<ObjectiveParameters>
    {
        #region Properties

        public ObjectiveType Objective { get; set; } = ObjectiveType.Regression;

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


        /// Parameters describing the metric <see cref="https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst#metric-parameters"/>

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

        private int _objective_seed = 5;
        /// <summary>
        /// Used only in rank_xendcg objective. Random seed for objectives, if random process is needed.
        /// </summary>
        public int ObjectiveSeed
        {
            get { return _objective_seed; }
            set
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException("ObjectiveSeed");
                _objective_seed = value;
            }
        }

        private int _lambdarank_truncation_level = 30;
        /// <summary>
        /// Used only in lambdarank application. Used for truncating the max DCG, refer to “truncation level” in the Sec. 3 of LambdaMART paper.
        /// </summary>
        public int LambdarankTruncationLevel
        {
            get { return _lambdarank_truncation_level; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("LambdarankTruncationLevel");
                _lambdarank_truncation_level = value;
            }
        }

        /// <summary>
        /// Used only in lambdarank application.
        /// Set this to true to normalize the lambdas for different queries, and improve the performance for unbalanced data.
        /// Set this to false to enforce the original lambdarank algorithm.
        /// </summary>
        public bool LambdarankNorm { get; set; } = true;

        private int _multi_error_top_k = 1;
        /// <summary>
        /// Used only with multi_error metric.
        /// Threshold for top-k multi-error metric.
        /// The error on each sample is 0 if the true class is among the top multi_error_top_k predictions, and 1 otherwise.
        /// More precisely, the error on a sample is 0 if there are at least num_classes - multi_error_top_k predictions strictly less than the prediction on the true class.
        /// When multi_error_top_k=1 this is equivalent to the usual multi-error metric.
        /// </summary>
        public int MultiErrorTopK
        {
            get { return _multi_error_top_k; }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("MultiErrorTopK");
                _multi_error_top_k = value;
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

        private static readonly double[] _defLabelGain;
        static ObjectiveParameters()
        {
            _defLabelGain = new double[31];
            for (int i = 0; i < 31; ++i)
                _defLabelGain[i] = (1L << i) - 1L;
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

        #region Structural equality overrides
        public override int GetHashCode() => _helper.GetHashCode(this);

        public override bool Equals(object value)
        {
            if (ReferenceEquals(null, value)) return false;
            if (ReferenceEquals(this, value)) return true;
            if (value.GetType() != this.GetType()) return false;
            return _helper.Equal(this, (ObjectiveParameters)value);
        }

        public static bool operator ==(ObjectiveParameters t, ObjectiveParameters other)
        {
            if (!ReferenceEquals(null, t)) return t.Equals(other);
            else if (!ReferenceEquals(null, other)) return other.Equals(t);
            else return true;
        }

        public static bool operator !=(ObjectiveParameters t, ObjectiveParameters other)
        {
            return !(t == other);
        }
        #endregion
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

        /// <summary>
        /// Setting this to ``true`` should ensure the stable results when using the same data and the same parameters (and different ``num_threads``)
        /// </summary>
        public bool Deterministic { get; set; } = false;

        /// <summary>
        /// Fit piecewise linear gradient boosting tree
        ///  * Tree splits are chosen in the usual way, but the model at each leaf is linear instead of constant
        ///  * The linear model at each leaf includes all the numerical features in that leaf’s branch
        ///  * Categorical features are used for splits as normal but are not used in the linear models
        ///  * Missing values should not be encoded as 0. Use np.nan for Python, NA for the CLI, and NA, NA_real_, or NA_integer_ for R
        ///  * It is recommended to rescale data before training so that features have similar mean and standard deviation
        ///  * Note: only works with CPU and serial tree learner
        ///  * Note: regression_l1 objective is not supported with linear tree boosting
        ///  * Note: setting linear_tree = true significantly increases the memory use of LightGBM
        /// </summary>
        public bool LinearTree { get; set; } = false;

        #region GPU
        /// <summary>
        /// OpenCL platform ID. Usually each GPU vendor exposes one OpenCL platform.
        /// -1 means the system-wide default platform.
        /// </summary>
        public int GpuPlatformId { get; set; } = -1;

        /// <summary>
        /// OpenCL device ID in the specified platform. Each GPU in the selected platform has a unique device ID.
        /// -1 means the default device in the selected platform
        /// </summary>
        public int GpuDeviceId { get; set; } = -1;

        /// <summary>
        /// Set this to true to use double precision math on GPU (by default single precision is used).
        /// </summary>
        public bool GpuUseDp { get; set; } = false;

        /// <summary>
        /// Number of GPUs
        /// </summary>
        public int NumGpu { get; set; } = 1;
        #endregion

        public CommonParameters(Dictionary<string, string> data)
        {
            FromParameters(data);
        }

        public CommonParameters() : base()
        {
        }

        public override void AddParameters(Dictionary<string, string> result)
        {
            _helper.AddParameters(this, result);
        }

        #region Structural equality overrides
        public override int GetHashCode() => _helper.GetHashCode(this);

        public override bool Equals(object value)
        {
            if (ReferenceEquals(null, value)) return false;
            if (ReferenceEquals(this, value)) return true;
            if (value.GetType() != this.GetType()) return false;
            return _helper.Equal(this, (CommonParameters)value);
        }

        public static bool operator ==(CommonParameters t, CommonParameters other)
        {
            if (!ReferenceEquals(null, t)) return t.Equals(other);
            else if (!ReferenceEquals(null, other)) return other.Equals(t);
            else return true;
        }

        public static bool operator !=(CommonParameters t, CommonParameters other)
        {
            return !(t == other);
        }
        #endregion
    }


    public class Parameters
    {
        // these two impact dataset construction
        public CommonParameters Common { get; set; }
        public DatasetParameters Dataset { get; set; }

        // these don't
        public ObjectiveParameters Objective { get; set; }
        public LearningParameters Learning { get; set; }

        public Parameters()
        {
            Common = new CommonParameters();
            Dataset = new DatasetParameters();
            Objective = new ObjectiveParameters();
            Learning = new LearningParameters();
        }

        public Parameters(string v)
        {
            var dict = ParamsHelper.SplitParameters(v);
            Common = CommonParameters.FromParameters(dict);
            Dataset = DatasetParameters.FromParameters(dict);
            Objective = ObjectiveParameters.FromParameters(dict);
            Learning = LearningParameters.FromParameters(dict);
        }

        public Dictionary<string,string> ToDict()
        {
            var dict = new Dictionary<string, string>();
            Common.AddParameters(dict);
            Dataset.AddParameters(dict);
            Objective.AddParameters(dict);
            Learning.AddParameters(dict);
            return dict;
        }

        public override string ToString()
        {
            return ParamsHelper.JoinParameters(ToDict());
        }

        #region Structural equality overrides
        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 13;
                hash = (hash * 7) + (!ReferenceEquals(null, Common) ? Common.GetHashCode() : 0);
                hash = (hash * 7) + (!ReferenceEquals(null, Dataset) ? Dataset.GetHashCode() : 0);
                hash = (hash * 7) + (!ReferenceEquals(null, Objective) ? Objective.GetHashCode() : 0);
                hash = (hash * 7) + (!ReferenceEquals(null, Learning) ? Learning.GetHashCode() : 0);
                return hash;
            }
        }

        public override bool Equals(object value)
        {
            if (ReferenceEquals(null, value)) return false;
            if (ReferenceEquals(this, value)) return true;
            if (value.GetType() != this.GetType()) return false;
            var that = (Parameters)value;
            return this.Common == that.Common &&
                   this.Dataset == that.Dataset &&
                   this.Objective == that.Objective &&
                   this.Learning == that.Learning;
        }

        public static bool operator ==(Parameters t, Parameters other)
        {
            if (!ReferenceEquals(null, t)) return t.Equals(other);
            else if (!ReferenceEquals(null, other)) return other.Equals(t);
            else return true;
        }

        public static bool operator !=(Parameters t, Parameters other)
        {
            return !(t == other);
        }
        #endregion

#pragma warning restore IDE0041 // Use 'is null' check

    }
}