#ifndef C10_MOBILE

#include <c10/cuda/CUDADriverAPI.h>
#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#else
#include <c10/util/win32-headers.h>
#endif

namespace c10 {
namespace cuda {
#ifndef _WIN32
CUDADriverAPI::CUDADriverAPI() {
#if defined(__APPLE__)
  std::string libcaffe2_nvrtc = "libcaffe2_nvrtc.dylib";
#else // if Linux
  std::string libcaffe2_nvrtc = "libcaffe2_nvrtc.so";
#endif
  handle = dlopen(libcaffe2_nvrtc.c_str(), RTLD_LOCAL | RTLD_NOW);
  if (!handle) {
    TORCH_WARN_ONCE("Error in dlopen: ", dlerror());
  } else {
    _hasPrimaryContext_funcptr = (_cuDevicePrimaryCtxGetState)dlsym(
        handle, "cuDevicePrimaryCtxGetState");
    if (!_hasPrimaryContext_funcptr) {
      TORCH_WARN_ONCE("Error in dlopen: ", dlerror());
    }
  }
}

CUDADriverAPI::~CUDADriverAPI() {
  if (!handle) {
    return;
  }
  dlclose(handle);
}
#else // if _WIN32
CUDADriverAPI::CUDADriverAPI() {
  LPCWSTR libcaffe2_nvrtc = L"libcaffe2_nvrtc.dll";
  HMODULE theModule;
  bool reload = true;
  // Check if LOAD_LIBRARY_SEARCH_DEFAULT_DIRS is supported
  if (GetProcAddress(GetModuleHandleW(L"KERNEL32.DLL"), "AddDllDirectory") !=
      NULL) {
    theModule =
        LoadLibraryExW(libcaffe2_nvrtc, NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
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
        " WinError in LoadLibrary for libcaffe2_nvrtc.dll. ",
        dw,
        ": ",
        buf);
  }

  FARPROC procAddress =
      GetProcAddress((HMODULE)handle, "cuDevicePrimaryCtxGetState");
  if (!procAddress) {
    TORCH_WARN_ONCE(
        false, " error in GetProcAddress. CUDA Driver API is not available.");
  }

  _hasPrimaryContext_funcptr = (_cuDevicePrimaryCtxGetState)procAddress;
}

CUDADriverAPI::~CUDADriverAPI() {
  if (!handle) {
    return;
  }
  FreeLibrary((HMODULE)handle);
}
#endif // _WIN32
bool CUDADriverAPI::hasPrimaryContext(int device) {
  if (!_hasPrimaryContext_funcptr) {
    return true;
  }

  int active = 0;
  unsigned int flags = 0;
  _hasPrimaryContext_funcptr(device, &flags, &active);

  return active == 1;
}
} // namespace cuda
} // namespace c10
#endif // C10_MOBILE