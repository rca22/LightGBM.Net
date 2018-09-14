using System;
using System.Collections.Generic;
using System.Text;

namespace LightGBMNet.Interface
{
    /// <summary>
    /// Use the IDisposable paradigm to ensure that network configuration is cleaned up
    /// after use.
    /// </summary>
    public sealed class NetworkConfig : IDisposable
    {
        public NetworkConfig (NetworkParameters pms)
        {
            PInvokeException.Check(PInvoke.NetworkInit(pms.Machines, pms.LocalListenPort, pms.TimeOut, pms.NumMachines),
                                   nameof(PInvoke.NetworkInit));
        }

        public NetworkConfig(   int numMachines,
                                int rank,
                                ReduceScatterFunction reduce,
                                AllGatherFunction gather)
        {
            if (numMachines < 1)
                throw new ArgumentOutOfRangeException("numMachines");

            PInvokeException.Check(PInvoke.NetworkInitWithFunctions(numMachines, rank, reduce, gather),
                                   nameof(PInvoke.NetworkInitWithFunctions));
        }

        #region IDisposable
        public void Dispose()
        {
            PInvokeException.Check(PInvoke.NetworkFree(), nameof(PInvoke.NetworkFree));
        }
        #endregion
    }
}
