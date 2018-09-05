using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;

namespace LightGBMNet.Interface
{
    class PInvokeException : System.Exception
    {
        public string FunctionName = "";
        public int ErrorCode = 0;

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
                throw new PInvokeException(res, fnName, msg);
            }
        }
    }
}
