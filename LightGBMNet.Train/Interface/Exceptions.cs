using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;

namespace LightGBMNet.Train
{
    class PInvokeException : System.Exception
    {
        public readonly string FunctionName;
        public readonly int ErrorCode;

        PInvokeException (int code, string msg, string fnName) : base(msg)
        {
            FunctionName = fnName;
            ErrorCode = code;
        }

        public static void Check(int res, string fnName)
        {
            if (res != 0)
            {
                var ptr = PInvoke.GetLastError();
                string msg = Marshal.PtrToStringAnsi(ptr);
                throw new PInvokeException(res, msg, fnName);
            }
        }
    }
}
