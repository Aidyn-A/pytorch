#pragma once

#include <cuda.h>
#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#else
namespace c10 {
namespace cuda {
namespace win32 {
#include <c10/util/win32-headers.h>
}
}
}
#endif

#include <c10/util/Exception.h>

namespace c10 {
namespace cuda {
#ifndef C10_MOBILE
class C10_CUDA_API CUDADriverAPI {
 public:
#ifndef _WIN32
  CUDADriverAPI() {
#if defined(__APPLE__)
    std::string libcaffe2_nvrtc = "libcaffe2_nvrtc.dylib";
#else // Linux
    std::string libcaffe2_nvrtc = "libcaffe2_nvrtc.so";
#endif
    handle = dlopen(libcaffe2_nvrtc.c_str(), RTLD_LOCAL | RTLD_NOW);
    if (!handle) {
      TORCH_WARN_ONCE("Error in dlopen: ", dlerror());
    } else {
      _c10_hasPrimaryContext = (_cuDevicePrimaryCtxGetState)dlsym(
          handle, "cuDevicePrimaryCtxGetState");
      if (!_c10_hasPrimaryContext) {
        TORCH_WARN_ONCE("Error in dlopen: ", dlerror());
      }
    }
  }

  ~CUDADriverAPI() {
    if (!handle) {
      return;
    }
    dlclose(handle);
  }
#else // ifdef _WIN32
  CUDADriverAPI() {
    using namespace c10::cuda::win32;
    LPCWSTR libcaffe2_nvrtc = L"libcaffe2_nvrtc.dll";
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    HMODULE theModule;
    bool reload = true;
    // Check if LOAD_LIBRARY_SEARCH_DEFAULT_DIRS is supported
    if (GetProcAddress(GetModuleHandleW(L"KERNEL32.DLL"), "AddDllDirectory") !=
        NULL) {
      theModule = LoadLibraryExW(
          libcaffe2_nvrtc, NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
      if (theModule != NULL || (GetLastError() != ERROR_MOD_NOT_FOUND)) {
        reload = false;
      }
    }

    if (reload) {
      theModule = LoadLibraryW(libcaffe2_nvrtc);
    }

    if (theModule) {
      handle = theModule;
    } else {
      char buf[256];
      DWORD dw = GetLastError();
      FormatMessageA(
          FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
          NULL,
          dw,
          MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
          buf,
          (sizeof(buf) / sizeof(char)),
          NULL);
      TORCH_WARN_ONCE(
          false,
          " LoadLibrary for libcaffe2_nvrtc.dll. WinError ",
          dw,
          ": ",
          buf);
    }

    FARPROC procAddress =
        GetProcAddress((HMODULE)handle, "cuDevicePrimaryCtxGetState");
    if (!procAddress) {
      TORCH_WARN_ONCE(false, " error in GetProcAddress");
    }

    _c10_hasPrimaryContext = (_cuDevicePrimaryCtxGetState)procAddress;
  }

  ~CUDADriverAPI() {
    using namespace c10::cuda::win32;
    if (!handle) {
      return;
    }
    FreeLibrary((HMODULE)handle);
  }
#endif // _WIN32
  bool hasPrimaryContext(int device) {
    if (!_c10_hasPrimaryContext) {
      return true;
    }

    int active = 0;
    unsigned int flags = 0;
    _c10_hasPrimaryContext(device, &flags, &active);

    return active == 1;
  }

 private:
  void* handle = nullptr;
  typedef CUresult (*_cuDevicePrimaryCtxGetState)(
      CUdevice dev,
      unsigned int* flags,
      int* active);
  _cuDevicePrimaryCtxGetState _c10_hasPrimaryContext = nullptr;
};
#endif // C10_MOBILE
} // namespace cuda
} // namespace c10
