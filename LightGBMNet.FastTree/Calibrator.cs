// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

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
        Float PredictProbability(Float output);
    }

    public class CalibratedPredictor : IPredictorWithFeatureWeights<Float>
    {
        public IPredictorWithFeatureWeights<Float> SubPredictor { get; }
        public ICalibrator Calibrator { get; }
        public PredictionKind PredictionKind => SubPredictor.PredictionKind;

        public CalibratedPredictor(IPredictorWithFeatureWeights<Float> predictor, ICalibrator calibrator)
        {
            SubPredictor = predictor;
            Calibrator = calibrator;
        }

        public void GetOutput(ref VBuffer<float> features, ref float prob)
        {
            float score = 0;
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
        public Double ParamA { get; }

        public PlattCalibrator(Double paramA)
        {
            ParamA = paramA;
        }

        private PlattCalibrator(BinaryReader reader)
        {
            ParamA = reader.ReadDouble();
        }

        public static PlattCalibrator Create(BinaryReader reader)
        {
            return new PlattCalibrator(reader);
        }

        public void Save(BinaryWriter writer)
        {
            SaveCore(writer);
        }

        private void SaveCore(BinaryWriter writer)
        {
            writer.Write(ParamA);
        }

        public Float PredictProbability(Float output)
        {
            if (Float.IsNaN(output))
                return output;
            return PredictProbability(output, ParamA);
        }

        public static Float PredictProbability(Float output, Double a)
        {
            return (Float)(1 / (1 + Math.Exp(a * output)));
        }
    }

    public sealed class ExponentialCalibrator : ICalibrator
    {
        public ExponentialCalibrator()
        {
        }

        public Float PredictProbability(Float output)
        {
            if (Float.IsNaN(output))
                return output;
            return (Float)Math.Exp(output);
        }
    }
}
