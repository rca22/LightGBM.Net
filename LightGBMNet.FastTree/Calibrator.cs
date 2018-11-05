// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;

namespace LightGBMNet.FastTree
{       
    /// <summary>
    /// An interface for probability calibrators.
    /// </summary>
    public interface ICalibrator
    {
        /// <summary> Given a classifier output, produce the probability </summary>		
        double PredictProbability(double output);
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

        public void GetOutput(ref VBuffer<float> features, ref double prob)
        {
            double score = 0;
            SubPredictor.GetOutput(ref features, ref score);
            prob = Calibrator.PredictProbability(score);
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

        public double PredictProbability(double output)
        {
            if (double.IsNaN(output))
                return output;
            return PredictProbability(output, ParamA);
        }

        public static double PredictProbability(double output, double a)
        {
            return 1.0 / (1.0 + Math.Exp(a * output));
        }
    }
    
    public sealed class ExponentialCalibrator : ICalibrator
    {
        private static readonly ExponentialCalibrator instance = new ExponentialCalibrator();

        private ExponentialCalibrator()
        {
        }
        static ExponentialCalibrator()
        {
        }

        public static ExponentialCalibrator Instance
        {
            get
            {
                return instance;
            }
        }

        public double PredictProbability(double output)
        {
            if (double.IsNaN(output))
                return output;
            return Math.Exp(output);
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
            else if (calibrator is ExponentialCalibrator e)
            {
                writer.Write(1);
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
            else
                throw new FormatException("Invalid ICalibrator flag");
        }
    }

}
