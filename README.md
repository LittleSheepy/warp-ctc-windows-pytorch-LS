# PyTorch bindings for Warp-ctc on windows

在windows上编译部署成功了

环境：

python 3.8.8

torch                   1.8.1+cu102

setuptools              52.0.0.post20210125

##  相对于项目https://gitee.com/zhangxilong191203/warp-ctc修改的地方:
1.src/ctc_entrypoint.cu  把ctc_entrypoint.cpp 改为 #include "ctc_entrypoint.cpp"

2.include/ctc.h 添加 __declspec(dllexport)

原：
```C++
ctcStatus_t compute_ctc_loss(const float* const activations,
                             float* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
```
改为：
```C++
__declspec(dllexport) ctcStatus_t compute_ctc_loss(const float* const activations,
                             float* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
```
3.pytorch_binding/src/binding.cpp
原：
```C++
extern THCState* state;
```
改为：
```C++
THCState *state = at::globalContext().lazyInitCUDA();
```
4.pytorch_binding/setup.py
原：
```python
warp_ctc_path = "../build"

if platform.system() == 'Darwin':
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

if "WARP_CTC_PATH" in os.environ:
    warp_ctc_path = os.environ["WARP_CTC_PATH"]
if not os.path.exists(os.path.join(warp_ctc_path, "libwarpctc" + lib_ext)):
    print(("Could not find libwarpctc.so in {}.\n"
           "Build warp-ctc and set WARP_CTC_PATH to the location of"
           " libwarpctc.so (default is '../build')").format(warp_ctc_path))
    sys.exit(1)
```
改为：
```python
warp_ctc_path = "../build/Release"

if platform.system() == 'Darwin':
    lib_ext = ".dylib"
elif platform.system() == "Windows":
    lib_ext = ".dll"
else:
    lib_ext = ".so"
    
if "WARP_CTC_PATH" in os.environ:
    warp_ctc_path = os.environ["WARP_CTC_PATH"]
if not os.path.exists(os.path.join(warp_ctc_path, "warpctc" + lib_ext)):
    print(("Could not find libwarpctc.so in {}.\n"
           "Build warp-ctc and set WARP_CTC_PATH to the location of"
           " libwarpctc.so (default is '../build')").format(warp_ctc_path))
    sys.exit(1)
```













参考项目：

https://github.com/SeanNaren/warp-ctc

https://gitee.com/zhangxilong191203/warp-ctc

