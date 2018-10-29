// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace LightGBMNet.FastTree
{
    using TScalarPredictor = IPredictorProducing<Float>;

    public sealed class OvaPredictor :
        IPredictorProducing<VBuffer<Float>>
        //PredictorBase<VBuffer<Float>>,
        //IValueMapper,
        //ICanSaveModel,
        //ICanSaveInSourceCode,
        //ICanSaveInTextFormat,
        //ISingleCanSavePfa
    {

        public TScalarPredictor[] Predictors { get; }
        public bool IsSoftMax { get; }

        public PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        public static OvaPredictor Create(bool isSoftMax, TScalarPredictor[] predictors)
        {
            return new OvaPredictor(isSoftMax, predictors);
        }

        private OvaPredictor(bool isSoftMax, TScalarPredictor[] predictors)
        {
            if ((predictors?.Length ?? 0) == 0)
                throw new ArgumentException("Predictors must be non-empty");
            IsSoftMax = isSoftMax;
            Predictors = predictors;
        }

        //private OvaPredictor(BinaryReader reader)
        //{
        //    int len = reader.ReadInt32();
        //    if (len <= 0) throw new FormatException();
        //    var predictors = new TScalarPredictor[len];
        //    LoadPredictors(predictors, reader);
        //    _impl = new ImplDist(predictors);
        //}
        //
        //public static OvaPredictor Create(BinaryReader reader)
        //{
        //    return new OvaPredictor(reader);
        //}

        //private static void LoadPredictors<TPredictor>(TPredictor[] predictors, BinaryReader reader)
        //    where TPredictor : class
        //{
        //    for (int i = 0; i < predictors.Length; i++)
        //        ctx.LoadModel<TPredictor, SignatureLoadModel>(env, out predictors[i], string.Format(SubPredictorFmt, i));
        //}
        //
        //protected void SaveCore(BinaryWriter writer)
        //{
        //    var preds = _impl.Predictors;
        //    writer.Write(preds.Length);
        //    for (int i = 0; i < preds.Length; i++)
        //        preds[i].Save(writer);
        //}

        public void GetOutput(ref VBuffer<Float> src, ref VBuffer<Float> dst)
        {
            var values = dst.Values;
            if ((values?.Length ?? 0) < Predictors.Length)
                values = new Float[Predictors.Length];

            var tmp = src;
            Parallel.For(0, Predictors.Length, i => Predictors[i].GetOutput(ref tmp, ref values[i]));

            if (IsSoftMax)
                Softmax(values, Predictors.Length);
            else
                Normalize(values, Predictors.Length);

            dst = new VBuffer<Float>(Predictors.Length, values, dst.Indices);
        }

        private static void Normalize(Float[] output, int count)
        {
            // Clamp to zero and normalize.
            Double sum = 0;
            for (int i = 0; i < count; i++)
            {
                var value = output[i];
                if (value >= 0)
                    sum += value;
                else
                    output[i] = 0;
            }

            if (sum > 0)
            {
                for (int i = 0; i < count; i++)
                    output[i] = (Float)(output[i] / sum);
            }
        }

        private static void Softmax(Float[] output, int count)
        {
            double wmax = output[0];
            for (int i = 1; i < count; ++i)
            {
                wmax = Math.Max(output[i], wmax);
            }
            double wsum = 0.0f;
            for (int i = 0; i < count; ++i)
            {
                output[i] = (Float) Math.Exp(output[i] - wmax);
                wsum += output[i];
            }
            for (int i = 0; i < count; ++i)
            {
                output[i] = (Float) (output[i] / wsum);
            }

            //// Clamp to zero and normalize.
            //Double sum = 0;
            //for (int i = 0; i < count; i++)
            //{
            //    var value = output[i];
            //    if (value >= 0)
            //        sum += value;
            //    else
            //        output[i] = 0;
            //}

            //if (sum > 0)
            //{
            //    for (int i = 0; i < count; i++)
            //        output[i] = (Float)(output[i] / sum);
            //}
        }

    }
}