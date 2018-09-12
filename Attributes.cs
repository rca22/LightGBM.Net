using System;
using System.Collections.Generic;
using System.Text;

namespace LightGBMNet.Interface
{
    public static class Check
    {
        public static void NonNull<T>(T x,string name) where T : class
        {
            if (x == null)
                throw new ArgumentException(name);
        }
    }
    /*
    /// <summary>
    /// An attribute used to annotate the valid range of a numeric input.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public sealed class RangeAttribute : Attribute
    {
        private object _min;
        private object _max;
        private object _inf;
        private object _sup;
        private Type _type;

        /// <summary>
        /// The target type of this range attribute, as determined by the type of
        /// the set range bound values.
        /// </summary>
        public Type Type => _type;

        /// <summary>
        /// An inclusive lower bound of the value.
        /// </summary>
        public object Min
        {
            get { return _min; }
            set
            {
                CheckType(value);
                if (_inf != null)
                    throw new ArgumentException("The minimum and infimum cannot be both set in a range attribute");
                Contracts.Check(_max == null || ((IComparable)_max).CompareTo(value) != -1,
                    "The minimum must be less than or equal to the maximum");
                Contracts.Check(_sup == null || ((IComparable)_sup).CompareTo(value) == 1,
                    "The minimum must be less than the supremum");
                _min = value;
            }
        }

        /// <summary>
        /// An inclusive upper bound of the value.
        /// </summary>
        public object Max
        {
            get { return _max; }
            set
            {
                CheckType(value);
                Contracts.Check(_sup == null,
                    "The maximum and supremum cannot be both set in a range attribute");
                Contracts.Check(_min == null || ((IComparable)_min).CompareTo(value) != 1,
                    "The maximum must be greater than or equal to the minimum");
                Contracts.Check(_inf == null || ((IComparable)_inf).CompareTo(value) == -1,
                    "The maximum must be greater than the infimum");
                _max = value;
            }
        }

        /// <summary>
        /// An exclusive lower bound of the value.
        /// </summary>
        public object Inf
        {
            get { return _inf; }
            set
            {
                CheckType(value);
                Contracts.Check(_min == null,
                    "The infimum and minimum cannot be both set in a range attribute");
                Contracts.Check(_max == null || ((IComparable)_max).CompareTo(value) == 1,
                    "The infimum must be less than the maximum");
                Contracts.Check(_sup == null || ((IComparable)_sup).CompareTo(value) == 1,
                    "The infimum must be less than the supremum");
                _inf = value;
            }
        }

        /// <summary>
        /// An exclusive upper bound of the value.
        /// </summary>
        public object Sup
        {
            get { return _sup; }
            set
            {
                CheckType(value);
                Contracts.Check(_max == null,
                    "The supremum and maximum cannot be both set in a range attribute");
                Contracts.Check(_min == null || ((IComparable)_min).CompareTo(value) == -1,
                    "The supremum must be greater than the minimum");
                Contracts.Check(_inf == null || ((IComparable)_inf).CompareTo(value) == -1,
                    "The supremum must be greater than the infimum");
                _sup = value;
            }
        }

        private void CheckType(object val)
        {
            Contracts.CheckValue(val, nameof(val));
            if (_type == null)
            {
                Contracts.Check(val is IComparable, "Type for range attribute must support IComparable");
                _type = val.GetType();
            }
            else
                Contracts.Check(_type == val.GetType(), "All Range attribute values must be of the same type");
        }

        public void CastToDouble()
        {
            _type = typeof(double);
            if (_inf != null)
                _inf = Convert.ToDouble(_inf);
            if (_min != null)
                _min = Convert.ToDouble(_min);
            if (_max != null)
                _max = Convert.ToDouble(_max);
            if (_sup != null)
                _sup = Convert.ToDouble(_sup);
        }

        public override string ToString()
        {
            string optionalTypeSpecifier = "";
            if (_type == typeof(double))
                optionalTypeSpecifier = "d";
            else if (_type == typeof(float))
                optionalTypeSpecifier = "f";

            var pieces = new List<string>();
            if (_inf != null)
                pieces.Add($"Inf = {_inf}{optionalTypeSpecifier}");
            if (_min != null)
                pieces.Add($"Min = {_min}{optionalTypeSpecifier}");
            if (_max != null)
                pieces.Add($"Max = {_max}{optionalTypeSpecifier}");
            if (_sup != null)
                pieces.Add($"Sup = {_sup}{optionalTypeSpecifier}");
            return $"[Range({string.Join(", ", pieces)})]";
        }
    }
    */
}
