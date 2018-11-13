// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;

namespace LightGBMNet.Tree
{       
    public interface ICalibrator
    {
        /// <summary> Given a classifier output, produce the transformed output</summary>		
        double Transform(double output);
    }

    public class CalibratedPredictor : IPredictorWithFeatureWeights<double>
    {
        public IPredictorWithFeatureWeights<double> SubPredictor { get; }
        public ICalibrator Calibrator { get; }
        public PredictionKind PredictionKind => SubPredictor.PredictionKind;

        public CalibratedPredictor(IPredictorWithFeatureWeights<double> predictor, ICalibrator calibrator)
        {
            SubPredictor = predictor;
            Calibrator = calibrator;
        }
        public int NumInputs => SubPredictor.NumInputs;
        public int MaxNumTrees {
            get => SubPredictor.MaxNumTrees;
            set => SubPredictor.MaxNumTrees = value;
        }

        public int MaxThreads
        {
            get => SubPredictor.MaxThreads;
            set => SubPredictor.MaxThreads = value;
        }

        public void GetOutput(ref VBuffer<float> features, ref double prob)
        {
            double score = 0;
            SubPredictor.GetOutput(ref features, ref score);
            prob = Calibrator.Transform(score);
        }

        public FeatureToGainMap GetFeatureWeights(bool normalise = false, bool splits = false)
        {
            return SubPredictor.GetFeatureWeights(normalise, splits);
        }
    }
    
    public sealed class PlattCalibrator : ICalibrator
    {
        public double ParamA { get; }

        public PlattCalibrator(double paramA)
        {
            ParamA = paramA;
        }

        public static PlattCalibrator Load(BinaryReader reader)
        {
            return new PlattCalibrator(reader.ReadDouble());
        }

        public void Save(BinaryWriter writer)
        {
            writer.Write(ParamA);
        }

        public double Transform(double output)
        {
            if (double.IsNaN(output))
                return output;
            return 1.0 / (1.0 + Math.Exp(ParamA * output));
        }

    }
    
    public sealed class ExponentialCalibrator : ICalibrator
    {
        private ExponentialCalibrator()
        {
        }
        static ExponentialCalibrator()
        {
        }

        public static ExponentialCalibrator Instance { get; } = new ExponentialCalibrator();

        public double Transform(double output)
        {
            if (double.IsNaN(output))
                return output;
            return Math.Exp(output);
        }
    }
       
    public sealed class SqrtCalibrator : ICalibrator
    {
        private SqrtCalibrator()
        {
        }
        static SqrtCalibrator()
        {
        }

        public static SqrtCalibrator Instance { get; } = new SqrtCalibrator();

        public double Transform(double output)
        {
            var square = output * output;
            return (output >= 0.0) ? square : -square;
        }
    }
    public static class CalibratorPersist
    {
        public static void Save(ICalibrator calibrator, BinaryWriter writer)
        {
            if (calibrator is PlattCalibrator platt)
            {
                writer.Write(0);
                platt.Save(writer);
            }
            else if (calibrator is ExponentialCalibrator)
            {
                writer.Write(1);
            }
            else if (calibrator is SqrtCalibrator)
            {
                writer.Write(2);
            }
            else
                throw new Exception("Unknown ICalibrator type");
        }

        public static ICalibrator Load(BinaryReader reader)
        {
            var flag = reader.ReadInt32();
            if (flag == 0)
                return PlattCalibrator.Load(reader);
            else if (flag == 1)
                return ExponentialCalibrator.Instance;
            else if (flag == 2)
                return SqrtCalibrator.Instance;
            else
                throw new FormatException("Invalid ICalibrator flag");
        }
    }

}
