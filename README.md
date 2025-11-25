# LightGBM.Net
.Net wrapper for [LightGBM](https://github.com/Microsoft/LightGBM/)

|Package|NuGet|
|---|---|
|LightGBMNet.Train|[![NuGet version](https://img.shields.io/nuget/v/LightGBMNet.Train.svg)](https://www.nuget.org/packages/LightGBMNet.Train)|
|LightGBMNet.Tree|[![NuGet version](https://img.shields.io/nuget/v/LightGBMNet.Tree.svg)](https://www.nuget.org/packages/LightGBMNet.Tree)|

* Native LightGBM binaries in NuGet package are compiled for Intel/AMD 64 bit processors with Visual Studio 2022 (requires corresponding Visual C++ 2022 redistributable package to be installed).
* Training generates both a simple wrapper around the native LightGBM ensemble, and a corresponding 100% managed tree ensemble implementation.
* Managed implementation is fully self-contained in `LightGBMNet.Tree` (a .NET 9 assembly), with rigorous unit testing to ensure native and managed models generate identical outputs.
* See [training unit tests](https://github.com/rca22/LightGBM.Net/blob/master/LightGBMNet.Test/TrainerTest.cs) for usage examples.
* GPU support:
    * Compile native LightGBM DLL/EXE files for your GPU configuration as per the [instructions here](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#build-gpu-version).
    * Copy the native LightGBM files with GPU support over the CPU only versions in the `native` directory (or over those found in the `build\x64` directory of the NuGet package)
    * Set `DeviceType` parameter to `GPU`.
