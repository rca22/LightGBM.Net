// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using LightGBMNet.Interface;
using NLog;

namespace LightGBMNet.Training
{
    /// <summary>
    /// Helpers to train a booster with given parameters.
    /// </summary>
    internal static class WrappedLightGbmTraining
    {
        private static Logger logger = LogManager.GetLogger("Trainer");

        /// <summary>
        /// Train and return a booster.
        /// </summary>
        public static Booster Train(Parameters parameters, 
                                    Dataset dtrain, 
                                    Dataset dvalid = null, 
                                    int numIteration = 100,
                                    int earlyStoppingRound = 0)
        {
            // create Booster.
            Booster bst = new Booster(parameters, dtrain, dvalid);

            // Disable early stopping if we don't have validation data.
            if (dvalid == null && earlyStoppingRound > 0)
            {
                earlyStoppingRound = 0;
                logger.Warn("Validation dataset not present, early stopping will be disabled.");
            }

            int bestIter = 0;
            double bestScore = double.MaxValue;
            double factorToSmallerBetter = 1.0;

            var metric = parameters.Metric.Metric;
            if (earlyStoppingRound > 0 && (metric == MetricType.Auc || metric == MetricType.Ndcg || metric == MetricType.Map))
                factorToSmallerBetter = -1.0;

            const int evalFreq = 50;

            double validError = double.NaN;
            double trainError = double.NaN;
            int iter = 0;
            for (iter = 0; iter < numIteration; ++iter)
            {
                if (bst.Update())
                    break;

                if (earlyStoppingRound > 0)
                {
                    validError = bst.EvalValid();
                    if (validError * factorToSmallerBetter < bestScore)
                    {
                        bestScore = validError * factorToSmallerBetter;
                        bestIter = iter;
                    }
                    if (iter - bestIter >= earlyStoppingRound)
                    {
                        logger.Info($"Met early stopping, best iteration: {bestIter + 1}, best score: {bestScore / factorToSmallerBetter}");
                        break;
                    }
                }
                if ((iter + 1) % evalFreq == 0)
                {
                    trainError = bst.EvalTrain();
                    if (dvalid != null)
                    {
                        if (earlyStoppingRound == 0)
                            validError = bst.EvalValid();
                        logger.Warn($"Eval {iter}, Training Error {trainError}, Validation Error {validError}");
                    }
                }
            }
            // Set the BestIteration.
            if (iter != numIteration && earlyStoppingRound > 0)
            {
                bst.BestIteration = bestIter + 1;
            }
            return bst;
        }
    }
}
