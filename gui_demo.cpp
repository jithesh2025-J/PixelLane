#define IMAGE_PROCESSOR_NO_MAIN
#include "image_processor.cpp"

#include <cpuid.h>
#include <windows.h>
#include <dwmapi.h>
#include <commdlg.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cwctype>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{

// -----------------------------------------------------------------------
// Design constants
// -----------------------------------------------------------------------

constexpr int DesignWidth  = 1420;
constexpr int DesignHeight = 930;

// Control IDs
constexpr int NavHome       = 2001;
constexpr int NavLoad       = 2002;
constexpr int NavProcessing = 2003;
constexpr int NavBenchmark  = 2004;
constexpr int NavAbout      = 2007;
constexpr int ModeGray      = 2101;
constexpr int ModeSobel     = 2102;
constexpr int ModeGaussian  = 2103;
constexpr int StrPass1      = 2107;   // Gaussian blur-pass selector
constexpr int StrPass2      = 2108;
constexpr int StrPass3      = 2109;
constexpr int ActionProcess   = 2201;
constexpr int ActionBenchmark = 2202;
constexpr int ActionSave      = 2203;
constexpr int ActionClear     = 2204;
constexpr int PanelOpen = 2205;
constexpr int PanelView = 2206;
constexpr int StatusId  = 2301;
constexpr UINT WM_APP_PROCESS_COMPLETE = WM_APP + 1;
constexpr UINT WM_APP_BENCHMARK_COMPLETE = WM_APP + 2;

// Theme colours
constexpr COLORREF Graphite     = RGB(9,   7,  15);
constexpr COLORREF Graphite2    = RGB(15, 11,  24);
constexpr COLORREF Panel        = RGB(23, 17,  34);
constexpr COLORREF Panel2       = RGB(31, 22,  46);
constexpr COLORREF Edge         = RGB(67, 45,  91);
constexpr COLORREF EdgeBright   = RGB(121, 82, 166);
constexpr COLORREF White        = RGB(244, 239, 255);
constexpr COLORREF Silver       = RGB(207, 196, 225);
constexpr COLORREF Dim          = RGB(164, 150, 184);
constexpr COLORREF Accent       = RGB(194, 107, 255);
constexpr COLORREF Green        = RGB(92,  222, 159);
constexpr COLORREF Red          = RGB(245, 112, 150);
constexpr COLORREF PurpleDeep   = RGB(83,  27,  126);
constexpr COLORREF PurpleBright = RGB(169, 62,  255);

enum class ProcessingMode { Grayscale, Sobel, Gaussian };
enum class Operation      { None, Load, Process, Benchmark, Save };

// Global state
HWND windowHandle    = nullptr;
HWND statusLabel     = nullptr;
HWND controls[32]    = {};
Operation operation   = Operation::None;
ProcessingMode mode   = ProcessingMode::Grayscale;
int activeNavigation  = NavHome;
COLORREF statusColor  = Dim;
HFONT titleFont    = nullptr;
HFONT headingFont  = nullptr;
HFONT bodyFont     = nullptr;
HFONT labelFont    = nullptr;
HFONT smallFont    = nullptr;
HBRUSH bgBrush     = nullptr;   // cached; freed in WM_DESTROY

std::wstring inputPath;
std::wstring outputPath;
int imageWidth  = 0;
int imageHeight = 0;
std::vector<std::uint8_t> originalRgb;
std::vector<std::uint8_t> processedPixels;
int processedChannels = 1;
bool processed    = false;
bool outputsEqual = false;
std::size_t validationMaxDiff = 0;
bool benchmarkRan = false;
double scalarMedian = 0.0;
double avxMedian    = 0.0;
std::string lastImageName = "No image loaded";
int gaussianPasses = 2;   // 1-3; default two-pass matches original behaviour

// Runtime CPU capability detection for Intel/AMD AVX2 and OS XSAVE state
bool __attribute__((target("xsave"))) detectCpuAvx2()
{
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    const bool osxsave = (ecx & (1 << 27)) != 0;
    const bool avx     = (ecx & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;

    // Check that OS actually enabled XMM and YMM state saving in XCR0
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 6) != 6) return false;

    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    return (ebx & (1 << 5)) != 0;
}

bool avx2Supported = false;
std::thread workerThread;
bool workerRunning = false;

// -----------------------------------------------------------------------
// Coordinate scaling  (design space -> window pixels)
// -----------------------------------------------------------------------

int sx(int v, int w) { return MulDiv(v, w, DesignWidth);  }
int sy(int v, int h) { return MulDiv(v, h, DesignHeight); }

void moveControl(HWND c, int x, int y, int w, int h, int cw, int ch)
{
    if (c) MoveWindow(c, sx(x,cw), sy(y,ch), sx(w,cw), sy(h,ch), TRUE);
}

void layoutControls(HWND window)
{
    RECT client = {};
    GetClientRect(window, &client);
    const int cw = std::max(1, (int)client.right);
    const int ch = std::max(1, (int)client.bottom);

    // Navigation sidebar (slots 0..3: Home, Load, Processing, Benchmark; slot 6: About)
    for (int i = 0; i < 4; ++i)
        moveControl(controls[i], 28, 136 + i * 48, 160, 42, cw, ch);
    moveControl(controls[6], 28, 328, 160, 42, cw, ch);  // ABOUT

    // Processing-mode buttons: keep a 14px inset below the panel heading.
    moveControl(controls[7], 248, 132, 132, 42, cw, ch);   // GRAYSCALE
    moveControl(controls[8], 388, 132, 132, 42, cw, ch);   // SOBEL EDGE
    moveControl(controls[9], 528, 132, 140, 42, cw, ch);   // GAUSSIAN BLUR

    // Gaussian strength buttons: inset inside the filter-settings box.
    moveControl(controls[16], 686, 130, 92, 28, cw, ch);
    moveControl(controls[17], 788, 130, 96, 28, cw, ch);
    moveControl(controls[18], 894, 130, 96, 28, cw, ch);

    // Small helper buttons inside image panels
    moveControl(controls[14], 348, 624, 160, 30, cw, ch);  // OPEN IMAGE
    moveControl(controls[15], 750, 624, 160, 30, cw, ch);  // VIEW RESULT

    // Main action buttons (below image panels)
    moveControl(controls[10], 248, 682, 190, 44, cw, ch);  // PROCESS IMAGE
    moveControl(controls[11], 448, 682, 190, 44, cw, ch);  // RUN BENCHMARK
    moveControl(controls[12], 648, 682, 160, 44, cw, ch);  // SAVE AS
    moveControl(controls[13], 818, 682, 140, 44, cw, ch);  // CLEAR

    moveControl(statusLabel, 228, 872, 1160, 28, cw, ch);
}

// -----------------------------------------------------------------------
// Gaussian RGB convolution helpers
// -----------------------------------------------------------------------

// 3x3 Gaussian kernel value for one channel at interior pixel (x,y).
// Reads from `image` (not output) so passes can ping-pong safely.
std::uint8_t gaussianRgbValue(
    const std::uint8_t* image,
    int width, int height,
    int x, int y, int channel)
{
    if (x == 0 || y == 0 || x + 1 >= width || y + 1 >= height)
        return image[(std::size_t(y) * width + x) * 3 + channel];

    const std::size_t row    = std::size_t(y) * width * 3;
    const std::size_t stride = std::size_t(width) * 3;
    const int sum =
        image[row - stride + (x-1)*3 + channel] +
      2*image[row - stride +  x   *3 + channel] +
        image[row - stride + (x+1)*3 + channel] +
      2*image[row          + (x-1)*3 + channel] +
      4*image[row          +  x   *3 + channel] +
      2*image[row          + (x+1)*3 + channel] +
        image[row + stride + (x-1)*3 + channel] +
      2*image[row + stride +  x   *3 + channel] +
        image[row + stride + (x+1)*3 + channel];
    return static_cast<std::uint8_t>((sum + 8) >> 4);
}

// Single pass — scalar
void __attribute__((optimize("no-tree-vectorize"))) gaussianRgbScalarPass(
    const std::uint8_t* image, std::uint8_t* output,
    int width, int height)
{
    std::memcpy(output, image, std::size_t(width) * height * 3);
    for (int y = 1; y + 1 < height; ++y)
        for (int x = 1; x + 1 < width; ++x)
            for (int ch = 0; ch < 3; ++ch)
                output[(std::size_t(y)*width + x)*3 + ch] =
                    gaussianRgbValue(image, width, height, x, y, ch);
}

// Single pass — AVX2
// FIX: scalar tail is now outside the SIMD x-loop so every interior pixel
// is processed exactly once; previously the tail was nested inside, causing
// SIMD results to be overwritten and tail pixels to be skipped.
void __attribute__((target("avx2"))) gaussianRgbAvx2Pass(
    const std::uint8_t* image, std::uint8_t* output,
    int width, int height)
{
    std::memcpy(output, image, std::size_t(width) * height * 3);
    if (width < 3 || height < 3) return;

    for (int y = 1; y + 1 < height; ++y)
    {
        int x = 1;

        // SIMD: process 8 pixels at a time.
        // Condition: x+8 <= width-1 ensures the rightmost neighbour (x+8)
        // is within bounds for every pixel in the block.
        for (; x + 8 <= width - 1; x += 8)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                // Lambda reads one channel byte from image
                auto v = [&](int row, int col) -> int {
                    return image[(std::size_t(row)*width + col)*3 + channel];
                };

                // 9 neighbour vectors for pixels [x .. x+7]
                const __m256i topLeft = _mm256_setr_epi32(
                    v(y-1,x-1), v(y-1,x),   v(y-1,x+1), v(y-1,x+2),
                    v(y-1,x+3), v(y-1,x+4), v(y-1,x+5), v(y-1,x+6));
                const __m256i top = _mm256_setr_epi32(
                    v(y-1,x),   v(y-1,x+1), v(y-1,x+2), v(y-1,x+3),
                    v(y-1,x+4), v(y-1,x+5), v(y-1,x+6), v(y-1,x+7));
                const __m256i topRight = _mm256_setr_epi32(
                    v(y-1,x+1), v(y-1,x+2), v(y-1,x+3), v(y-1,x+4),
                    v(y-1,x+5), v(y-1,x+6), v(y-1,x+7), v(y-1,x+8));

                const __m256i midLeft = _mm256_setr_epi32(
                    v(y,x-1), v(y,x),   v(y,x+1), v(y,x+2),
                    v(y,x+3), v(y,x+4), v(y,x+5), v(y,x+6));
                const __m256i mid = _mm256_setr_epi32(
                    v(y,x),   v(y,x+1), v(y,x+2), v(y,x+3),
                    v(y,x+4), v(y,x+5), v(y,x+6), v(y,x+7));
                const __m256i midRight = _mm256_setr_epi32(
                    v(y,x+1), v(y,x+2), v(y,x+3), v(y,x+4),
                    v(y,x+5), v(y,x+6), v(y,x+7), v(y,x+8));

                const __m256i botLeft = _mm256_setr_epi32(
                    v(y+1,x-1), v(y+1,x),   v(y+1,x+1), v(y+1,x+2),
                    v(y+1,x+3), v(y+1,x+4), v(y+1,x+5), v(y+1,x+6));
                const __m256i bot = _mm256_setr_epi32(
                    v(y+1,x),   v(y+1,x+1), v(y+1,x+2), v(y+1,x+3),
                    v(y+1,x+4), v(y+1,x+5), v(y+1,x+6), v(y+1,x+7));
                const __m256i botRight = _mm256_setr_epi32(
                    v(y+1,x+1), v(y+1,x+2), v(y+1,x+3), v(y+1,x+4),
                    v(y+1,x+5), v(y+1,x+6), v(y+1,x+7), v(y+1,x+8));

                // Kernel [1 2 1 / 2 4 2 / 1 2 1] / 16
                __m256i sum = _mm256_add_epi32(
                    _mm256_add_epi32(topLeft, topRight),
                    _mm256_slli_epi32(top, 1));
                sum = _mm256_add_epi32(sum, _mm256_add_epi32(
                    _mm256_slli_epi32(midLeft, 1),
                    _mm256_slli_epi32(midRight, 1)));
                sum = _mm256_add_epi32(sum, _mm256_slli_epi32(mid, 2));
                sum = _mm256_add_epi32(sum,
                    _mm256_add_epi32(botLeft, botRight));
                sum = _mm256_add_epi32(sum, _mm256_slli_epi32(bot, 1));
                sum = _mm256_srli_epi32(
                    _mm256_add_epi32(sum, _mm256_set1_epi32(8)), 4);

                alignas(32) std::uint32_t blurred[8];
                _mm256_store_si256((__m256i*)blurred, sum);
                for (int i = 0; i < 8; ++i)
                    output[(std::size_t(y)*width + x + i)*3 + channel] =
                        static_cast<std::uint8_t>(blurred[i]);
            }
        } // ← SIMD x-loop closes here

        // Scalar tail: up to 7 remaining interior pixels
        for (; x + 1 < width; ++x)
            for (int ch = 0; ch < 3; ++ch)
                output[(std::size_t(y)*width + x)*3 + ch] =
                    gaussianRgbValue(image, width, height, x, y, ch);
    }
}

// N-pass Gaussian blur — scalar with preallocated ping-pong buffers.
void __attribute__((optimize("no-tree-vectorize"))) gaussianRgbScalarPreallocated(
    const std::uint8_t* image, std::uint8_t* output,
    int width, int height, int passes,
    std::uint8_t* buf0, std::uint8_t* buf1)
{
    const std::size_t bytes = std::size_t(width) * height * 3;
    std::memcpy(buf0, image, bytes);
    for (int p = 0; p < passes; ++p)
    {
        const std::uint8_t* src = (p % 2 == 0) ? buf0 : buf1;
        std::uint8_t*       dst = (p % 2 == 0) ? buf1 : buf0;
        gaussianRgbScalarPass(src, dst, width, height);
    }
    const std::uint8_t* result = (passes % 2 == 1) ? buf1 : buf0;
    std::memcpy(output, result, bytes);
}

// N-pass Gaussian blur — AVX2 with preallocated ping-pong buffers.
void __attribute__((target("avx2"))) gaussianRgbAvx2Preallocated(
    const std::uint8_t* image, std::uint8_t* output,
    int width, int height, int passes,
    std::uint8_t* buf0, std::uint8_t* buf1)
{
    const std::size_t bytes = std::size_t(width) * height * 3;
    std::memcpy(buf0, image, bytes);
    for (int p = 0; p < passes; ++p)
    {
        const std::uint8_t* src = (p % 2 == 0) ? buf0 : buf1;
        std::uint8_t*       dst = (p % 2 == 0) ? buf1 : buf0;
        gaussianRgbAvx2Pass(src, dst, width, height);
    }
    const std::uint8_t* result = (passes % 2 == 1) ? buf1 : buf0;
    std::memcpy(output, result, bytes);
}

// N-pass Gaussian blur — scalar. Ping-pong between two buffers.
void __attribute__((optimize("no-tree-vectorize"))) gaussianRgbScalar(
    const std::uint8_t* image, std::uint8_t* output,
    int width, int height, int passes)
{
    const std::size_t bytes = std::size_t(width) * height * 3;
    std::vector<std::uint8_t> buf0(bytes);
    std::vector<std::uint8_t> buf1(bytes);
    gaussianRgbScalarPreallocated(image, output, width, height, passes, buf0.data(), buf1.data());
}

// N-pass Gaussian blur — AVX2. Same ping-pong structure.
void __attribute__((target("avx2"))) gaussianRgbAvx2(
    const std::uint8_t* image, std::uint8_t* output,
    int width, int height, int passes)
{
    const std::size_t bytes = std::size_t(width) * height * 3;
    std::vector<std::uint8_t> buf0(bytes);
    std::vector<std::uint8_t> buf1(bytes);
    gaussianRgbAvx2Preallocated(image, output, width, height, passes, buf0.data(), buf1.data());
}

// -----------------------------------------------------------------------
// Correctness tests: Gaussian RGB scalar vs AVX2 across width/pass combos
// -----------------------------------------------------------------------

bool runGaussianRgbEdgeCaseTests(std::string& error)
{
    const int widths[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,
                           24,31,32,33,63,64,65};
    const int heights[] = {3, 5, 9};
    const int passCounts[] = {1, 2, 3};

    for (int width : widths)
    {
        for (int height : heights)
        {
            for (int passes : passCounts)
            {
                const std::size_t bytes = std::size_t(width) * height * 3;
                std::vector<std::uint8_t> img(bytes);
                for (std::size_t i = 0; i < bytes; ++i)
                    img[i] = static_cast<std::uint8_t>((i * 37 + 19) & 0xFF);

                std::vector<std::uint8_t> scalarOut(bytes);
                std::vector<std::uint8_t> avxOut(bytes);
                gaussianRgbScalar(img.data(), scalarOut.data(), width, height, passes);
                gaussianRgbAvx2  (img.data(), avxOut.data(),    width, height, passes);

                for (std::size_t i = 0; i < bytes; ++i)
                {
                    if (scalarOut[i] != avxOut[i])
                    {
                        char buf[320];
                        std::snprintf(buf, sizeof(buf),
                            "Gaussian RGB mismatch:\n"
                            "  width=%d  height=%d  passes=%d\n"
                            "  byte=%zu  scalar=%d  avx2=%d",
                            width, height, passes,
                            i, (int)scalarOut[i], (int)avxOut[i]);
                        error = buf;
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

// -----------------------------------------------------------------------
// String / path utilities
// -----------------------------------------------------------------------

const char* modeName(ProcessingMode m)
{
    switch (m)
    {
    case ProcessingMode::Sobel:    return "Sobel Edge Detection";
    case ProcessingMode::Gaussian: return "Gaussian Blur";
    default:                       return "Grayscale";
    }
}

std::string narrowPath(const std::wstring& path)
{
    const int sz = WideCharToMultiByte(
        CP_UTF8, 0, path.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (sz <= 1) return {};
    std::string r(std::size_t(sz), '\0');
    WideCharToMultiByte(CP_UTF8, 0, path.c_str(), -1, r.data(), sz, nullptr, nullptr);
    r.pop_back();
    return r;
}

std::wstring chooseOpenPath(HWND owner)
{
    std::vector<wchar_t> buf(32768, L'\0');
    OPENFILENAMEW d = {};
    d.lStructSize = sizeof(d);
    d.hwndOwner   = owner;
    d.lpstrFilter =
        L"Images\0*.jpg;*.jpeg;*.png;*.bmp;*.tga;*.hdr\0All files\0*.*\0";
    d.lpstrFile = buf.data();
    d.nMaxFile  = (DWORD)buf.size();
    d.Flags     = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    return GetOpenFileNameW(&d) ? std::wstring(buf.data()) : std::wstring();
}

std::wstring defaultSaveFileName()
{
    if (mode == ProcessingMode::Sobel)
        return L"image_sobel.png";
    if (mode == ProcessingMode::Gaussian)
    {
        switch (gaussianPasses)
        {
        case 1:  return L"image_gaussian_1pass.png";
        case 2:  return L"image_gaussian_2pass.png";
        case 3:  return L"image_gaussian_3pass.png";
        default: return L"image_gaussian_" + std::to_wstring(gaussianPasses) + L"pass.png";
        }
    }
    return L"image_grayscale.png";
}

std::wstring chooseSavePath(HWND owner)
{
    std::vector<wchar_t> buf(32768, L'\0');
    const std::wstring name = defaultSaveFileName();
    std::copy(name.begin(), name.end(), buf.begin());
    OPENFILENAMEW d = {};
    d.lStructSize = sizeof(d);
    d.hwndOwner   = owner;
    d.lpstrFilter = L"PNG or JPEG image\0*.png;*.jpg;*.jpeg\0All files\0*.*\0";
    d.lpstrFile   = buf.data();
    d.nMaxFile    = (DWORD)buf.size();
    d.lpstrDefExt = L"png";
    d.Flags       = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
    return GetSaveFileNameW(&d) ? std::wstring(buf.data()) : std::wstring();
}

std::wstring executableDirectory()
{
    std::vector<wchar_t> buf(MAX_PATH);
    DWORD len = GetModuleFileNameW(nullptr, buf.data(), (DWORD)buf.size());
    while (len == buf.size())
    {
        buf.resize(buf.size() * 2);
        len = GetModuleFileNameW(nullptr, buf.data(), (DWORD)buf.size());
    }
    if (len == 0) return {};
    const std::wstring p(buf.data(), len);
    const std::size_t sl = p.find_last_of(L"\\/");
    return sl == std::wstring::npos ? std::wstring() : p.substr(0, sl);
}

std::wstring outputFilePath()
{
    const std::wstring dir = executableDirectory();
    return dir.empty() ? L"output_grayscale.png" : dir + L"\\output_grayscale.png";
}

std::wstring withImageExtension(std::wstring path)
{
    std::wstring lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(),
        [](wchar_t c){ return (wchar_t)std::towlower(c); });
    const bool hasPng = lower.size() >= 4 &&
                        lower.substr(lower.size()-4) == L".png";
    const bool hasJpg = lower.size() >= 4 &&
                        lower.substr(lower.size()-4) == L".jpg";
    const bool hasJpeg = lower.size() >= 5 &&
                         lower.substr(lower.size()-5) == L".jpeg";
    if (!hasPng && !hasJpg && !hasJpeg)
        path += L".png";
    return path;
}

std::string errorText(const char* op)
{
    const DWORD code = GetLastError();
    char buf[256] = {};
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, code, 0, buf, sizeof(buf), nullptr);
    std::ostringstream oss;
    oss << op << " failed (Windows error " << code << "): " << buf;
    return oss.str();
}

// -----------------------------------------------------------------------
// Image I/O
// -----------------------------------------------------------------------

bool loadImage(const std::wstring& path, std::string& error)
{
    FILE* file = _wfopen(path.c_str(), L"rb");
    if (!file)
    {
        error = "Could not open image: " + narrowPath(path) + "\r\n";
        error += errorText("Opening image");
        return false;
    }
    int ch = 0, w = 0, h = 0;
    stbi_uc* decoded = stbi_load_from_file(file, &w, &h, &ch, 3);
    std::fclose(file);
    if (!decoded)
    {
        error = "Could not decode image: " + narrowPath(path) + "\r\nstb_image: ";
        error += stbi_failure_reason();
        return false;
    }
    const std::size_t bytes = std::size_t(w) * h * 3;
    originalRgb.assign(decoded, decoded + bytes);
    stbi_image_free(decoded);
    if (w <= 0 || h <= 0) { error = "Invalid image dimensions."; return false; }
    inputPath   = path;
    imageWidth  = w;
    imageHeight = h;
    lastImageName = narrowPath(path);
    const std::size_t sl = lastImageName.find_last_of("\\/");
    if (sl != std::string::npos) lastImageName.erase(0, sl + 1);
    processedPixels.clear();
    outputPath.clear();
    processed = outputsEqual = benchmarkRan = false;
    validationMaxDiff = 0;
    scalarMedian = 0.0;
    avxMedian    = 0.0;
    return true;
}

void writePng(void* ctx, void* data, int size)
{
    std::fwrite(data, 1, std::size_t(size), (FILE*)ctx);
}

std::size_t maxOutputDiff(
    const std::uint8_t* first,
    const std::uint8_t* second,
    std::size_t count)
{
    std::size_t maximum = 0;
    for (std::size_t i = 0; i < count; ++i)
    {
        const std::size_t diff = first[i] > second[i]
            ? std::size_t(first[i] - second[i])
            : std::size_t(second[i] - first[i]);
        maximum = std::max(maximum, diff);
    }
    return maximum;
}

std::size_t outputTolerance()
{
    return 0;
}

bool saveProcessedImage(const std::wstring& requestedPath, std::string& error)
{
    const std::wstring destination = withImageExtension(requestedPath);
    std::wstring lower = destination;
    std::transform(lower.begin(), lower.end(), lower.begin(),
        [](wchar_t c){ return (wchar_t)std::towlower(c); });
    const bool jpeg = lower.size() >= 4 &&
        (lower.substr(lower.size()-4) == L".jpg" ||
         (lower.size() >= 5 && lower.substr(lower.size()-5) == L".jpeg"));

    FILE* file = _wfopen(destination.c_str(), L"wb");
    if (!file)
    {
        error = errorText("Creating output image");
        return false;
    }

    int written = 0;
    if (jpeg)
    {
        written = stbi_write_jpg_to_func(
            writePng, file, imageWidth, imageHeight,
            processedChannels, processedPixels.data(), 95);
    }
    else
    {
        written = stbi_write_png_to_func(
            writePng, file, imageWidth, imageHeight,
            processedChannels, processedPixels.data(),
            imageWidth * processedChannels);
    }
    const int closed = std::fclose(file);
    if (written == 0 || closed != 0)
    {
        error = errorText("Writing output image");
        return false;
    }
    return true;
}

double median(std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    if (n % 2 == 0)
        return (v[n / 2 - 1] + v[n / 2]) / 2.0;
    return v[n / 2];
}

// -----------------------------------------------------------------------
// UI state helpers
// -----------------------------------------------------------------------

void setStatus(const std::string& msg, COLORREF color)
{
    statusColor = color;
    SetWindowTextA(statusLabel, msg.c_str());
    InvalidateRect(windowHandle, nullptr, FALSE);
}

void setEnabledState()
{
    const bool isAbout = (activeNavigation == NavAbout);

    // Controls 7..18 are workspace buttons (mode, strength, actions, panel helpers)
    for (int i = 7; i <= 18; ++i)
    {
        if (controls[i]) ShowWindow(controls[i], isAbout ? SW_HIDE : SW_SHOW);
    }

    if (isAbout) return;

    for (HWND c : controls) if (c) EnableWindow(c, TRUE);
    EnableWindow(controls[10], !originalRgb.empty());   // PROCESS IMAGE
    EnableWindow(controls[11], processed);               // RUN BENCHMARK
    EnableWindow(controls[12], processed);               // SAVE AS
    EnableWindow(controls[15], processed);               // VIEW RESULT
    // Strength buttons: only relevant when Gaussian mode is active
    const bool gaussianActive =
        !originalRgb.empty() && mode == ProcessingMode::Gaussian;
    EnableWindow(controls[16], gaussianActive);
    EnableWindow(controls[17], gaussianActive);
    EnableWindow(controls[18], gaussianActive);
}

void begin(Operation op, const char* msg)
{
    operation = op;
    setStatus(msg, Accent);
    for (HWND c : controls) if (c) EnableWindow(c, FALSE);
    UpdateWindow(windowHandle);
}

void endOperation()
{
    operation = Operation::None;
    setEnabledState();
    InvalidateRect(windowHandle, nullptr, FALSE);
}

// -----------------------------------------------------------------------
// GDI drawing helpers
// -----------------------------------------------------------------------

void text(
    HDC dc, const char* value,
    int x, int y, int w, int h,
    COLORREF color, HFONT font,
    UINT fmt = DT_LEFT | DT_TOP)
{
    SetTextColor(dc, color);
    SetBkMode(dc, TRANSPARENT);
    SelectObject(dc, font);
    RECT r = {x, y, x+w, y+h};
    DrawTextA(dc, value, -1, &r, fmt | DT_NOPREFIX);
}

void panel(HDC dc, RECT rect, COLORREF fill = Panel)
{
    const int h = std::max(1, (int)(rect.bottom - rect.top));
    for (int y = rect.top; y < rect.bottom; ++y)
    {
        const double t = (double)(y - rect.top) / h;
        HBRUSH row = CreateSolidBrush(RGB(
            std::min(255, GetRValue(fill) + (int)(t*5.0)),
            std::min(255, GetGValue(fill) + (int)(t*6.0)),
            std::min(255, GetBValue(fill) + (int)(t*7.0))));
        RECT s = {rect.left, y, rect.right, y+1};
        FillRect(dc, &s, row);
        DeleteObject(row);
    }
    HPEN pen = CreatePen(PS_SOLID, 1, Edge);
    HGDIOBJ old = SelectObject(dc, pen);
    SelectObject(dc, GetStockObject(NULL_BRUSH));
    RoundRect(dc, rect.left, rect.top, rect.right, rect.bottom, 8, 8);
    SelectObject(dc, old);
    DeleteObject(pen);
}

void glossyPanel(HDC dc, RECT rect)
{
    panel(dc, rect, Panel2);
    HPEN hl = CreatePen(PS_SOLID, 1, EdgeBright);
    HGDIOBJ old = SelectObject(dc, hl);
    MoveToEx(dc, rect.left+10, rect.top+1, nullptr);
    LineTo(dc, rect.right-10, rect.top+1);
    SelectObject(dc, old);
    DeleteObject(hl);
}

void line(HDC dc, int x1, int y1, int x2, int y2, COLORREF color);
void Circle(HDC dc, int x, int y, int r);

void apertureIcon(HDC dc, int cx, int cy, int radius)
{
    HBRUSH ob = CreateSolidBrush(RGB(44,51,55));
    HGDIOBJ oldB = SelectObject(dc, ob);
    HPEN ep = CreatePen(PS_SOLID, 1, EdgeBright);
    HGDIOBJ oldP = SelectObject(dc, ep);
    Ellipse(dc, cx-radius, cy-radius, cx+radius, cy+radius);
    SelectObject(dc, oldB); SelectObject(dc, oldP);
    DeleteObject(ob); DeleteObject(ep);
    HBRUSH blade = CreateSolidBrush(Silver);
    SelectObject(dc, blade);
    for (int i = 0; i < 6; ++i)
    {
        const double a = i * 1.047197551;
        const int bx = cx + (int)(std::cos(a)*radius*0.42);
        const int by = cy + (int)(std::sin(a)*radius*0.42);
        Ellipse(dc, bx-radius/3, by-radius/3, bx+radius/3, by+radius/3);
    }
    SelectObject(dc, oldB);
    DeleteObject(blade);
    HBRUSH cen = CreateSolidBrush(Graphite);
    SelectObject(dc, cen);
    Ellipse(dc, cx-radius/4, cy-radius/4, cx+radius/4, cy+radius/4);
    SelectObject(dc, oldB);
    DeleteObject(cen);
}

void navGlyph(HDC dc, int x, int y, int index, bool active)
{
    const COLORREF color = active ? White : Silver;
    HPEN pen = CreatePen(PS_SOLID, 2, color);
    HGDIOBJ old = SelectObject(dc, pen);
    switch (index)
    {
    case 0:
        MoveToEx(dc, x, y+8, nullptr); LineTo(dc, x+8, y); LineTo(dc, x+16, y+8);
        Rectangle(dc, x+3, y+7, x+13, y+17); break;
    case 1:
        Rectangle(dc, x+1, y+4, x+17, y+15);
        MoveToEx(dc, x+4, y+4, nullptr); LineTo(dc, x+7, y+1); LineTo(dc, x+13, y+1); break;
    case 2:
        Circle(dc, x+4, y+4, 3); Circle(dc, x+13, y+14, 3);
        line(dc, x+7, y+4, x+16, y+4, color);
        line(dc, x+1, y+14, x+10, y+14, color); break;
    case 3:
        line(dc, x+2, y+17, x+2, y+10, color);
        line(dc, x+9, y+17, x+9, y+5, color);
        line(dc, x+16, y+17, x+16, y+1, color); break;
    case 4:
        Circle(dc, x+9, y+9, 7); Circle(dc, x+9, y+9, 2); break;
    case 5:
        Circle(dc, x+9, y+9, 7);
        line(dc, x+9, y+2, x+9, y+16, color);
        line(dc, x+2, y+9, x+16, y+9, color); break;
    default:
        Circle(dc, x+9, y+9, 7);
        text(dc, "i", x+5, y+1, 9, 16, color, bodyFont, DT_CENTER); break;
    }
    SelectObject(dc, old);
    DeleteObject(pen);
}

void line(HDC dc, int x1, int y1, int x2, int y2, COLORREF color)
{
    HPEN pen = CreatePen(PS_SOLID, 1, color);
    HGDIOBJ old = SelectObject(dc, pen);
    MoveToEx(dc, x1, y1, nullptr);
    LineTo(dc, x2, y2);
    SelectObject(dc, old);
    DeleteObject(pen);
}

void Circle(HDC dc, int x, int y, int r)
{
    Ellipse(dc, x-r, y-r, x+r, y+r);
}

void preview(
    HDC dc,
    const std::vector<std::uint8_t>& pixels,
    int channels,
    RECT area,
    bool isProcessedPanel = false)
{
    if (pixels.empty())
    {
        // Render a centered, polished empty-state viewport placeholder
        const int cx = (area.left + area.right) / 2;
        const int cy = (area.top + area.bottom) / 2 - 25;
        const int boxW = 260;
        const int boxH = 150;
        const RECT box = {cx - boxW/2, cy - boxH/2, cx + boxW/2, cy + boxH/2};

        // Dashed border placeholder
        HPEN dashPen = CreatePen(PS_DOT, 1, EdgeBright);
        HGDIOBJ oldP = SelectObject(dc, dashPen);
        SelectObject(dc, GetStockObject(NULL_BRUSH));
        RoundRect(dc, box.left, box.top, box.right, box.bottom, 12, 12);
        SelectObject(dc, oldP);
        DeleteObject(dashPen);

        // Center aperture / glyph indicator
        Circle(dc, cx, cy - 28, 18);
        Circle(dc, cx, cy - 28, 5);

        if (!isProcessedPanel)
        {
            text(dc, "NO IMAGE LOADED",
                 box.left + 10, cy + 2, boxW - 20, 22,
                 Silver, bodyFont, DT_CENTER);
            text(dc, "Click 'LOAD IMAGE' or 'OPEN IMAGE' to select a PNG / JPG / BMP file.",
                 box.left + 15, cy + 26, boxW - 30, 36,
                 Dim, smallFont, DT_CENTER);
        }
        else
        {
            text(dc, "AWAITING PROCESSING",
                 box.left + 10, cy + 2, boxW - 20, 22,
                 Silver, bodyFont, DT_CENTER);
            text(dc, "Load an image and press 'PROCESS IMAGE' to view filtered result.",
                 box.left + 15, cy + 26, boxW - 30, 36,
                 Dim, smallFont, DT_CENTER);
        }
        return;
    }
    const double scale = std::min(
        (double)(area.right - area.left - 10) / imageWidth,
        (double)(area.bottom - area.top  - 10) / imageHeight);
    const int dw = std::max(1, (int)(imageWidth  * scale));
    const int dh = std::max(1, (int)(imageHeight * scale));
    const int left = area.left + ((area.right  - area.left) - dw) / 2;
    const int top  = area.top  + ((area.bottom - area.top)  - dh) / 2;

    std::vector<std::uint32_t> bgra(std::size_t(imageWidth) * imageHeight);
    for (std::size_t i = 0; i < bgra.size(); ++i)
    {
        const std::uint8_t r = channels == 1 ? pixels[i] : pixels[i*3];
        const std::uint8_t g = channels == 1 ? pixels[i] : pixels[i*3+1];
        const std::uint8_t b = channels == 1 ? pixels[i] : pixels[i*3+2];
        bgra[i] = b | ((std::uint32_t)g<<8) | ((std::uint32_t)r<<16);
    }
    BITMAPINFO bmi = {};
    bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth       = imageWidth;
    bmi.bmiHeader.biHeight      = -imageHeight;
    bmi.bmiHeader.biPlanes      = 1;
    bmi.bmiHeader.biBitCount    = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    SetStretchBltMode(dc, HALFTONE);
    StretchDIBits(dc, left, top, dw, dh, 0, 0, imageWidth, imageHeight,
                  bgra.data(), &bmi, DIB_RGB_COLORS, SRCCOPY);
}

void card(HDC dc, RECT rect, const char* label, const char* value, COLORREF vc)
{
    panel(dc, rect, Panel2);
    text(dc, label, rect.left+16, rect.top+14,
         rect.right-rect.left-32, 20, Silver, smallFont);
    text(dc, value, rect.left+16, rect.top+38,
         rect.right-rect.left-32, 36, vc, headingFont);
    line(dc, rect.left+16, rect.bottom-14, rect.right-16, rect.bottom-14, EdgeBright);
}

void drawButton(HDC /*dc*/, const DRAWITEMSTRUCT* item, bool selected = false)
{
    const int id       = (int)item->CtlID;
    const bool pressed  = (item->itemState & ODS_SELECTED) != 0;
    const bool disabled = (item->itemState & ODS_DISABLED) != 0;
    const bool primary  = id == ActionProcess;
    COLORREF fill = disabled
        ? RGB(25, 20, 31)
        : (primary ? (pressed ? PurpleDeep : PurpleBright)
                   : (selected ? RGB(71,43,91)
                               : (pressed ? RGB(53,32,69) : Panel2)));
    HBRUSH br = CreateSolidBrush(fill);
    FillRect(item->hDC, &item->rcItem, br);
    DeleteObject(br);
    HPEN pen = CreatePen(PS_SOLID, disabled ? 1 : ((selected||primary)?2:1),
                         disabled ? Edge : (primary ? Accent
                                                     : (selected ? Accent : Edge)));
    HGDIOBJ old = SelectObject(item->hDC, pen);
    SelectObject(item->hDC, GetStockObject(NULL_BRUSH));
    Rectangle(item->hDC, item->rcItem.left, item->rcItem.top,
              item->rcItem.right, item->rcItem.bottom);
    SelectObject(item->hDC, old);
    DeleteObject(pen);
    char label[96] = {};
    GetWindowTextA(item->hwndItem, label, sizeof(label));
    const bool nav = id >= NavHome && id <= NavAbout;
    if (nav)
    {
        const bool active = id == activeNavigation;
        navGlyph(item->hDC, item->rcItem.left+14, item->rcItem.top+12,
                 id - NavHome, active);
    }
    text(item->hDC, label,
         item->rcItem.left  + (nav ? 40 : 0),
         item->rcItem.top,
         item->rcItem.right  - item->rcItem.left - (nav ? 40 : 0),
         item->rcItem.bottom - item->rcItem.top,
         disabled ? Dim : White, bodyFont,
         nav ? DT_LEFT|DT_VCENTER|DT_SINGLELINE
             : DT_CENTER|DT_VCENTER|DT_SINGLELINE);
}

void createControl(
    const char* label, int id,
    int x, int y, int w, int h,
    bool button = true)
{
    int slot = 15;
    switch (id)
    {
    case NavHome:         slot = 0;  break;
    case NavLoad:         slot = 1;  break;
    case NavProcessing:   slot = 2;  break;
    case NavBenchmark:    slot = 3;  break;
    case NavAbout:        slot = 6;  break;
    case ModeGray:        slot = 7;  break;
    case ModeSobel:       slot = 8;  break;
    case ModeGaussian:    slot = 9;  break;
    case StrPass1:        slot = 16; break;
    case StrPass2:        slot = 17; break;
    case StrPass3:        slot = 18; break;
    case ActionProcess:   slot = 10; break;
    case ActionBenchmark: slot = 11; break;
    case ActionSave:      slot = 12; break;
    case ActionClear:     slot = 13; break;
    case PanelOpen:       slot = 14; break;
    case PanelView:       slot = 15; break;
    default: break;
    }
    controls[slot] = CreateWindowA(
        button ? "BUTTON" : "STATIC", label,
        button ? WS_VISIBLE|WS_CHILD|BS_OWNERDRAW
               : WS_VISIBLE|WS_CHILD,
        x, y, w, h,
        windowHandle, (HMENU)(INT_PTR)id, nullptr, nullptr);
}

void clearImage()
{
    originalRgb.clear();
    processedPixels.clear();
    processedChannels = 1;
    inputPath.clear();
    outputPath.clear();
    imageWidth = imageHeight = 0;
    processed = outputsEqual = benchmarkRan = false;
    validationMaxDiff = 0;
    scalarMedian = 0.0;
    avxMedian    = 0.0;
    lastImageName = "No image loaded";
}

struct WorkerResult
{
    bool success = false;
    bool processCompleted = false;
    std::string error;
    std::vector<std::uint8_t> processedPixels;
    std::wstring outputPath;
    int processedChannels = 1;
    bool outputsEqual = false;
    std::size_t validationMaxDiff = 0;
    double scalarMedian = 0.0;
    double avxMedian = 0.0;
    bool benchmarkRan = false;
};

struct ProcessingSnapshot
{
    ProcessingMode mode = ProcessingMode::Grayscale;
    int gaussianPasses = 2;
    int imageWidth = 0;
    int imageHeight = 0;
    bool avx2Supported = false;
    std::vector<std::uint8_t> originalRgb;
};

ProcessingSnapshot captureProcessingSnapshot()
{
    ProcessingSnapshot snapshot;
    snapshot.mode = mode;
    snapshot.gaussianPasses = gaussianPasses;
    snapshot.imageWidth = imageWidth;
    snapshot.imageHeight = imageHeight;
    snapshot.avx2Supported = avx2Supported;
    snapshot.originalRgb = originalRgb;
    return snapshot;
}

void calculateOutputLocal(
    const ProcessingSnapshot& snapshot,
    bool avx,
    std::vector<std::uint8_t>& output,
    int& channels)
{
    if (avx && !snapshot.avx2Supported)
        avx = false;

    const std::size_t count =
        std::size_t(snapshot.imageWidth) * snapshot.imageHeight;
    if (snapshot.mode == ProcessingMode::Gaussian)
    {
        channels = 3;
        output.resize(count * 3);
        if (avx)
            gaussianRgbAvx2(snapshot.originalRgb.data(), output.data(),
                            snapshot.imageWidth, snapshot.imageHeight,
                            snapshot.gaussianPasses);
        else
            gaussianRgbScalar(snapshot.originalRgb.data(), output.data(),
                              snapshot.imageWidth, snapshot.imageHeight,
                              snapshot.gaussianPasses);
        return;
    }

    channels = 1;
    std::vector<std::uint8_t> gray(count);
    if (avx)
        grayscaleAvx2(snapshot.originalRgb.data(), gray.data(), count);
    else
        grayscaleScalar(snapshot.originalRgb.data(), gray.data(), count);
    output.resize(count);
    if (snapshot.mode == ProcessingMode::Grayscale)
        output = gray;
    else
        avx ? sobelAvx2(gray.data(), output.data(),
                        snapshot.imageWidth, snapshot.imageHeight)
            : sobelScalar(gray.data(), output.data(),
                          snapshot.imageWidth, snapshot.imageHeight);
}

bool runBenchmarkLocal(
    const ProcessingSnapshot& snapshot,
    WorkerResult& result,
    int timedTrials)
{
    if (!snapshot.avx2Supported)
    {
        result.error = "AVX2 is unavailable on this CPU.";
        return false;
    }

    const std::size_t pixelCount =
        std::size_t(snapshot.imageWidth) * snapshot.imageHeight;
    const std::size_t outBytes = pixelCount *
        (snapshot.mode == ProcessingMode::Gaussian ? 3 : 1);
    std::vector<std::uint8_t> scalarOut(outBytes);
    std::vector<std::uint8_t> avxOut(outBytes);
    std::vector<std::uint8_t> scalarGray(pixelCount);
    std::vector<std::uint8_t> avxGray(pixelCount);
    std::vector<std::uint8_t> scalarBuf0, scalarBuf1, avxBuf0, avxBuf1;
    if (snapshot.mode == ProcessingMode::Gaussian)
    {
        scalarBuf0.resize(outBytes);
        scalarBuf1.resize(outBytes);
        avxBuf0.resize(outBytes);
        avxBuf1.resize(outBytes);
    }

    auto runKernel = [&](bool avx, std::vector<std::uint8_t>& output) {
        if (snapshot.mode == ProcessingMode::Gaussian)
        {
            if (avx)
                gaussianRgbAvx2Preallocated(
                    snapshot.originalRgb.data(), output.data(),
                    snapshot.imageWidth, snapshot.imageHeight,
                    snapshot.gaussianPasses, avxBuf0.data(), avxBuf1.data());
            else
                gaussianRgbScalarPreallocated(
                    snapshot.originalRgb.data(), output.data(),
                    snapshot.imageWidth, snapshot.imageHeight,
                    snapshot.gaussianPasses,
                    scalarBuf0.data(), scalarBuf1.data());
        }
        else if (snapshot.mode == ProcessingMode::Grayscale)
        {
            avx ? grayscaleAvx2(snapshot.originalRgb.data(), output.data(),
                                 pixelCount)
                : grayscaleScalar(snapshot.originalRgb.data(), output.data(),
                                  pixelCount);
        }
        else
        {
            if (avx)
            {
                grayscaleAvx2(snapshot.originalRgb.data(), avxGray.data(),
                              pixelCount);
                sobelAvx2(avxGray.data(), output.data(),
                          snapshot.imageWidth, snapshot.imageHeight);
            }
            else
            {
                grayscaleScalar(snapshot.originalRgb.data(), scalarGray.data(),
                                pixelCount);
                sobelScalar(scalarGray.data(), output.data(),
                            snapshot.imageWidth, snapshot.imageHeight);
            }
        }
    };

    runKernel(false, scalarOut);
    runKernel(true, avxOut);
    if (maxOutputDiff(
            scalarOut.data(), avxOut.data(), outBytes) > outputTolerance())
    {
        result.error = "Benchmark warm-up validation failed.";
        return false;
    }

    std::vector<double> scalarTimes;
    std::vector<double> avxTimes;
    scalarTimes.reserve(timedTrials);
    avxTimes.reserve(timedTrials);
    for (int trial = 0; trial < timedTrials; ++trial)
    {
        const auto scalarRun = [&]() {
            const auto start = std::chrono::steady_clock::now();
            runKernel(false, scalarOut);
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        };
        const auto avxRun = [&]() {
            const auto start = std::chrono::steady_clock::now();
            runKernel(true, avxOut);
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        };
        if (trial % 2 == 0)
        {
            scalarTimes.push_back(std::max(0.0001, scalarRun()));
            avxTimes.push_back(std::max(0.0001, avxRun()));
        }
        else
        {
            avxTimes.push_back(std::max(0.0001, avxRun()));
            scalarTimes.push_back(std::max(0.0001, scalarRun()));
        }
    }

    if (maxOutputDiff(
            scalarOut.data(), avxOut.data(), outBytes) > outputTolerance())
    {
        result.error = "Benchmark post-trial validation failed.";
        return false;
    }
    result.scalarMedian = median(scalarTimes);
    result.avxMedian = median(avxTimes);
    result.benchmarkRan = true;
    return true;
}

void startProcessWorker(HWND window, ProcessingSnapshot snapshot)
{
    workerRunning = true;
    workerThread = std::thread([window, snapshot = std::move(snapshot)]() {
        WorkerResult* result = new WorkerResult;
        int channels = 1;
        std::vector<std::uint8_t> scalar;
        calculateOutputLocal(snapshot, false, scalar, channels);
        calculateOutputLocal(snapshot, true, result->processedPixels, channels);
        const std::size_t outputSize =
            std::size_t(snapshot.imageWidth) * snapshot.imageHeight * channels;
        result->processedChannels = channels;
        if (snapshot.avx2Supported)
        {
            result->validationMaxDiff = maxOutputDiff(
                scalar.data(), result->processedPixels.data(), outputSize);
            result->outputsEqual =
                result->validationMaxDiff <= outputTolerance();
        }
        else
        {
            result->processedPixels = std::move(scalar);
            result->validationMaxDiff = 0;
            result->outputsEqual = false;
        }
        if (!result->outputsEqual)
            result->error = "Scalar and AVX2 outputs differ beyond the allowed tolerance.";
        else
        {
            result->outputPath = outputFilePath();
            FILE* file = _wfopen(result->outputPath.c_str(), L"wb");
            if (!file)
                result->error = errorText("Creating generated PNG");
            else
            {
                const int written = stbi_write_png_to_func(
                    writePng, file, snapshot.imageWidth, snapshot.imageHeight,
                    channels, result->processedPixels.data(),
                    snapshot.imageWidth * channels);
                const int closed = std::fclose(file);
                if (written == 0 || closed != 0)
                    result->error = errorText("Writing generated PNG");
                else
                {
                    result->success = true;
                    result->processCompleted = true;
                }
            }
        }
        if (result->success && snapshot.avx2Supported)
            result->success = runBenchmarkLocal(snapshot, *result, 3);
        if (!PostMessageA(window, WM_APP_PROCESS_COMPLETE, 0,
                          reinterpret_cast<LPARAM>(result)))
            delete result;
    });
}

void startBenchmarkWorker(HWND window, ProcessingSnapshot snapshot)
{
    workerRunning = true;
    workerThread = std::thread([window, snapshot = std::move(snapshot)]() {
        WorkerResult* result = new WorkerResult;
        result->success = runBenchmarkLocal(snapshot, *result, 10);
        if (!PostMessageA(window, WM_APP_BENCHMARK_COMPLETE, 0,
                          reinterpret_cast<LPARAM>(result)))
            delete result;
    });
}

// -----------------------------------------------------------------------
// Window procedure
// -----------------------------------------------------------------------

LRESULT CALLBACK windowProcedure(
    HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_APP_PROCESS_COMPLETE:
    {
        WorkerResult* result = reinterpret_cast<WorkerResult*>(lParam);
        const bool success = result->success;
        const bool processCompleted = result->processCompleted;
        const std::string error = result->error;
        if (processCompleted)
        {
            processedPixels = std::move(result->processedPixels);
            outputPath = std::move(result->outputPath);
            processedChannels = result->processedChannels;
            processed = true;
            outputsEqual = result->outputsEqual;
            validationMaxDiff = result->validationMaxDiff;
            scalarMedian = result->scalarMedian;
            avxMedian = result->avxMedian;
            benchmarkRan = result->benchmarkRan;
        }
        delete result;
        if (workerThread.joinable()) workerThread.join();
        workerRunning = false;
        if (success)
        {
            if (avx2Supported)
                setStatus(std::string("PROCESSED  |  ") + modeName(mode) +
                          "  |  PASS - max diff " +
                          std::to_string(validationMaxDiff), Green);
            else
                setStatus(std::string("PROCESSED  |  ") + modeName(mode) +
                          "  |  Scalar fallback (AVX2 unsupported)", Green);
        }
        else if (processCompleted)
            setStatus("ERROR  |  " + error, Red);
        else
            setStatus("ERROR  |  " + error, Red);
        endOperation();
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    }

    case WM_APP_BENCHMARK_COMPLETE:
    {
        WorkerResult* result = reinterpret_cast<WorkerResult*>(lParam);
        const bool success = result->success;
        const std::string error = result->error;
        delete result;
        if (workerThread.joinable()) workerThread.join();
        workerRunning = false;
        if (success)
            setStatus("BENCHMARK COMPLETE  |  Actual median results shown", Green);
        else
            setStatus("ERROR  |  " + error, Red);
        endOperation();
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    }

    case WM_COMMAND:
    {
        const int id = LOWORD(wParam);

        // Mode selection
        if (id == ModeGray || id == ModeSobel || id == ModeGaussian)
        {
            if (workerRunning) return 0;
            activeNavigation = NavProcessing;
            mode = id == ModeGray    ? ProcessingMode::Grayscale
                 : id == ModeSobel   ? ProcessingMode::Sobel
                 : ProcessingMode::Gaussian;
            processed = benchmarkRan = false;
            scalarMedian = avxMedian = 0.0;
            processedPixels.clear();
            setEnabledState();
            setStatus(std::string("MODE SELECTED  |  ") + modeName(mode), Accent);
            InvalidateRect(window, nullptr, FALSE);
            return 0;
        }

        // Gaussian blur-pass strength
        if (id == StrPass1 || id == StrPass2 || id == StrPass3)
        {
            if (workerRunning) return 0;
            gaussianPasses = id == StrPass1 ? 1 : id == StrPass2 ? 2 : 3;
            processed = benchmarkRan = false;
            scalarMedian = avxMedian = 0.0;
            processedPixels.clear();
            setEnabledState();
            char msg[96];
            std::snprintf(msg, sizeof(msg),
                "BLUR STRENGTH  |  %d pass%s  |  Press Process Image to apply",
                gaussianPasses, gaussianPasses > 1 ? "es" : "");
            setStatus(msg, Accent);
            InvalidateRect(window, nullptr, FALSE);
            return 0;
        }

        if (id == NavLoad || id == PanelOpen)
        {
            if (workerRunning) return 0;
            activeNavigation = NavLoad;
            begin(Operation::Load, "LOAD IMAGE  |  Choose an image file...");
            const std::wstring path = chooseOpenPath(window);
            if (!path.empty())
            {
                std::string err;
                if (loadImage(path, err))
                    setStatus("IMAGE LOADED  |  " + lastImageName, Green);
                else
                {
                    setStatus("ERROR  |  " + err, Red);
                    MessageBoxA(window, err.c_str(), "PixelLane", MB_OK|MB_ICONERROR);
                }
            }
            endOperation();
            InvalidateRect(window, nullptr, FALSE);
            return 0;
        }

        if (id == ActionProcess)
        {
            if (workerRunning) return 0;
            activeNavigation = NavProcessing;
            begin(Operation::Process, "PROCESS IMAGE  |  Running selected filter...");
            startProcessWorker(window, captureProcessingSnapshot());
            return 0;
        }

        if (id == ActionBenchmark || id == NavBenchmark)
        {
            if (workerRunning) return 0;
            if (!avx2Supported)
            {
                setStatus("ERROR  |  AVX2 is not supported by this CPU.", Red);
                return 0;
            }
            activeNavigation = NavBenchmark;
            begin(Operation::Benchmark, "BENCHMARK  |  Warm-up + 10 timed trials...");
            startBenchmarkWorker(window, captureProcessingSnapshot());
            return 0;
        }

        if (id == ActionSave)
        {
            if (workerRunning) return 0;
            begin(Operation::Save, "SAVE AS  |  Choose a destination...");
            const std::wstring sel = chooseSavePath(window);
            if (!sel.empty() && processed)
            {
                std::string err;
                const std::wstring dest = withImageExtension(sel);
                if (!saveProcessedImage(dest, err))
                    setStatus("ERROR  |  " + err, Red);
                else
                    setStatus("SAVED  |  " + narrowPath(dest), Green);
            }
            endOperation();
            return 0;
        }

        if (id == ActionClear)
        {
            if (workerRunning) return 0;
            clearImage();
            setStatus("CLEARED  |  Workspace reset", Accent);
            setEnabledState();
            InvalidateRect(window, nullptr, FALSE);
            return 0;
        }

        if (id == PanelView)
        {
            if (!outputPath.empty())
            {
                ShellExecuteW(window, L"open", outputPath.c_str(),
                              nullptr, nullptr, SW_SHOWNORMAL);
                setStatus("VIEW RESULT  |  Processed image opened", Green);
            }
            return 0;
        }

        if (id == NavHome || id == NavProcessing || id == NavAbout)
        {
            if (workerRunning) return 0;
            activeNavigation = id;
            if (id == NavAbout)
                setStatus("ABOUT  |  Architecture, SIMD AVX2 acceleration & filter design", Accent);
            else if (id == NavProcessing)
                setStatus("PROCESSING  |  Select filter mode and pass options to process", Accent);
            else
                setStatus("HOME  |  PixelLane workspace", Accent);
            setEnabledState();
            InvalidateRect(window, nullptr, FALSE);
            return 0;
        }
        return 0;
    }

    case WM_CTLCOLORSTATIC:
    {
        // FIX: cache the brush rather than allocating a new one every message.
        if (!bgBrush) bgBrush = CreateSolidBrush(Graphite);
        SetTextColor((HDC)wParam, statusColor);
        SetBkColor((HDC)wParam, Graphite);
        return (LRESULT)bgBrush;
    }

    case WM_SIZE:
        layoutControls(window);
        InvalidateRect(window, nullptr, FALSE);
        return 0;

    case WM_ERASEBKGND:
        return 1;

    case WM_CLOSE:
        if (workerThread.joinable())
        {
            workerThread.join();
            workerRunning = false;
        }
        {
            MSG pending = {};
            while (PeekMessageA(&pending, window,
                                WM_APP_PROCESS_COMPLETE,
                                WM_APP_BENCHMARK_COMPLETE,
                                PM_REMOVE))
            {
                delete reinterpret_cast<WorkerResult*>(pending.lParam);
            }
        }
        DestroyWindow(window);
        return 0;

    case WM_DRAWITEM:
    {
        const DRAWITEMSTRUCT* item = (const DRAWITEMSTRUCT*)lParam;
        const int id = (int)item->CtlID;

        const bool modeSelected =
            (id == ModeGray     && mode == ProcessingMode::Grayscale) ||
            (id == ModeSobel    && mode == ProcessingMode::Sobel)     ||
            (id == ModeGaussian && mode == ProcessingMode::Gaussian);

        const bool strengthSelected =
            mode == ProcessingMode::Gaussian &&
            ((id == StrPass1 && gaussianPasses == 1) ||
             (id == StrPass2 && gaussianPasses == 2) ||
             (id == StrPass3 && gaussianPasses == 3));

        const bool selected = modeSelected || strengthSelected || (id == activeNavigation);
        drawButton(item->hDC, item, selected);
        return TRUE;
    }

    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(window, &ps);
        RECT client;
        GetClientRect(window, &client);
        const int cw = std::max(1, (int)client.right);
        const int ch = std::max(1, (int)client.bottom);

        HBRUSH bg = CreateSolidBrush(Graphite);
        FillRect(dc, &client, bg);
        DeleteObject(bg);

        SetMapMode(dc, MM_ANISOTROPIC);
        SetWindowExtEx  (dc, DesignWidth,  DesignHeight,  nullptr);
        SetViewportExtEx(dc, cw,           ch,            nullptr);

        // ── Header ────────────────────────────────────────────────────────
        glossyPanel(dc, {8, 8, 1392, 76});
        apertureIcon(dc, 42, 42, 22);
        text(dc, "PIXEL",     78,  20, 82,  32, White,  titleFont);
        text(dc, "LANE",      160, 20, 220, 32, Accent, titleFont);
        text(dc, "SIMD IMAGE PROCESSING ACCELERATOR - AVX2",
             162, 52, 380, 18, Silver, labelFont);
        text(dc, "NEON PURPLE", 1160, 22, 190, 18, White, labelFont, DT_RIGHT);
        text(dc, "PRECISION  |  PERFORMANCE  |  SIMPLICITY",
             1010, 44, 340, 18, Silver, labelFont, DT_RIGHT);
        line(dc, 1040, 76, 1040, 860, Edge);

        // ── Left navigation sidebar ───────────────────────────────────────
        glossyPanel(dc, {8, 84, 210, 860});
        text(dc, "NAVIGATION", 34, 104, 150, 18, Dim, smallFont);
        text(dc, "AVX2",       32, 790, 100, 34, Silver, headingFont);
        text(dc, "FAST\nSHARP\nCLEAN", 34, 828, 100, 50, Dim, smallFont);

        if (activeNavigation == NavAbout)
        {
            // ── About view panel ──────────────────────────────────────────────
            glossyPanel(dc, {228, 84, 1374, 858});
            text(dc, "PROJECT ARCHITECTURE & SIMD VECTORIZATION", 258, 106, 600, 26, White, headingFont);
            text(dc, "PixelLane  |  NATIVE WIN32 C++17 WITH INTEL AVX2 INTRINSICS", 258, 136, 600, 18, Accent, smallFont);

            // Card 1: SIMD & AVX2 Fundamentals
            panel(dc, {258, 172, 790, 480}, Panel2);
            text(dc, "AVX2 SIMD ACCELERATION", 278, 188, 300, 20, White, bodyFont);
            text(dc, "Single Instruction, Multiple Data (SIMD) enables parallel processing on 256-bit vector registers (__m256i). Each vector instruction computes multiple data elements simultaneously across independent lanes:",
                 278, 214, 490, 38, Silver, smallFont);
            text(dc, "- Grayscale: 8 RGB pixels (24 bytes) processed simultaneously using vector arithmetic, shifting, and saturation packing.",
                 278, 256, 490, 36, Silver, smallFont);
            text(dc, "- Sobel Edge Detection: 8 contiguous horizontal (Gx) and vertical (Gy) gradients computed in parallel across 256-bit SIMD lanes.",
                 278, 298, 490, 36, Silver, smallFont);
            text(dc, "- Gaussian Blur: 8 contiguous RGB pixels (all 3 channels) computed simultaneously via 2D 3x3 convolution with bitwise scalar equivalence.",
                 278, 340, 490, 36, Silver, smallFont);
            text(dc, "- Safe boundary handling: Unaligned tail pixels (1..7) execute via scalar boundary loops to ensure memory safety without out-of-bounds reads.",
                 278, 382, 490, 40, Dim, smallFont);
            line(dc, 278, 432, 770, 432, Edge);
            text(dc, "ISA Target: AVX2 + FMA  |  Vector Width: 256 bits (32 bytes)  |  Toolchain: GCC x64",
                 278, 444, 490, 18, Green, smallFont);

            // Card 2: Algorithms & Mathematics
            panel(dc, {810, 172, 1344, 480}, Panel2);
            text(dc, "FILTERS & MATHEMATICAL FORMULATIONS", 830, 188, 350, 20, White, bodyFont);
            text(dc, "1. Grayscale Conversion:\n   Formula: Gray = (77*R + 150*G + 29*B) >> 8\n   Fixed-point integer arithmetic equivalent to ITU-R BT.601 standard weights.",
                 830, 214, 490, 50, Silver, smallFont);
            text(dc, "2. Sobel Edge Detection:\n   Horizontal Kernel: [-1 0 1; -2 0 2; -1 0 1]\n   Vertical Kernel:   [ 1 2 1;  0 0 0; -1 -2 -1]\n   Magnitude: min(255, |Gx| + |Gy|) with zero-padded boundaries.",
                 830, 274, 490, 56, Silver, smallFont);
            text(dc, "3. Gaussian Blur (2D 3x3 Convolution & Multi-Pass):\n   Normalized Kernel: [1 2 1; 2 4 2; 1 2 1] / 16 (direct 2D convolution)\n   Multi-pass ping-pong buffers support 1 to 3 repeated passes; each additional pass compounds smoothing and broadens the effective blur radius.",
                 830, 340, 490, 68, Silver, smallFont);
            line(dc, 830, 432, 1324, 432, Edge);
            text(dc, "Repeated passes ping-pong between preallocated buffers with zero heap reallocation.",
                 830, 444, 490, 18, Silver, smallFont);

            // Card 3: Benchmarking Methodology
            panel(dc, {258, 498, 1344, 820}, Panel2);
            text(dc, "RIGOROUS BENCHMARKING METHODOLOGY", 278, 514, 400, 20, White, bodyFont);
            text(dc, "To guarantee scientific accuracy and eliminate synthetic measurement biases:",
                 278, 540, 900, 20, Silver, smallFont);
            text(dc, "- Memory Preallocation: All destination and ping-pong buffers are preallocated prior to starting timed trials. Heap allocation and OS memory management overhead are completely excluded from measured kernel times.",
                 278, 566, 1040, 36, Silver, smallFont);
            text(dc, "- Alternating Trial Execution: Order alternates between Scalar-first and AVX2-first to eliminate CPU cache warmth asymmetries and branch predictor bias.",
                 278, 608, 1040, 36, Silver, smallFont);
            text(dc, "- Median Filtering: Median runtime across 10 trials is computed by averaging the two center values, discarding OS scheduling interrupts and CPU frequency scaling outliers.",
                 278, 650, 1040, 36, Silver, smallFont);
            text(dc, "- Validation Guarantee: Startup edge-case testing checks widths 1..17 and boundary multiples; every benchmark run verifies that scalar and AVX2 outputs match byte-for-byte.",
                 278, 692, 1040, 36, Silver, smallFont);
            line(dc, 278, 742, 1324, 742, Edge);
            text(dc, "Click 'PROCESSING' or 'HOME' in the navigation sidebar to return to the active workspace.",
                 278, 764, 800, 20, Accent, bodyFont);

            line(dc, 228, 874, 1374, 874, Edge);
            text(dc, "PixelLane  v1.0.0", 228, 882, 220, 16, Dim, smallFont);
            text(dc, "SIMD IMAGE PROCESSING ACCELERATOR - AVX2", 500, 882, 300, 16, Dim, smallFont);
            text(dc, "NEON PURPLE  |  MINIMAL. SHARP. POWERFUL.", 940, 882, 430, 16, Dim, smallFont, DT_RIGHT);

            EndPaint(window, &ps);
            return 0;
        }

        // ── Mode / filter control bar ─────────────────────────────────────
        glossyPanel(dc, {228, 84, 1030, 196});
        text(dc, "PROCESSING MODE", 248, 98, 180, 18, Silver, smallFont);
        text(dc, modeName(mode), 748, 98, 250, 18, Dim, smallFont, DT_RIGHT);

        // Filter settings sub-panel
        text(dc, "FILTER SETTINGS", 674, 98, 160, 18, Dim, smallFont);
        panel(dc, {674, 120, 1018, 184}, Panel2);

        if (mode == ProcessingMode::Gaussian)
        {
            // Three ownerdraw strength buttons are positioned by layoutControls
            // at design coords {686,130}, {788,130}, {894,130}, size ~94x28.
            // Their selected state is rendered by WM_DRAWITEM.
            char passInfo[140];
            const char* blurDesc = (gaussianPasses == 1) ? "1 pass (light blur)"
                                 : (gaussianPasses == 2) ? "2 passes (medium blur)"
                                                         : "3 passes (strong blur)";
            std::snprintf(passInfo, sizeof(passInfo),
                          "Kernel 3x3  |  %s  |  Additional passes increase smoothing",
                          blurDesc);
            text(dc, passInfo, 686, 164, 324, 18, Silver, smallFont);
        }
        else
        {
            text(dc,
                 mode == ProcessingMode::Sobel
                     ? "Edge strength  |  Absolute gradient magnitude | min(255, |Gx| + |Gy|)"
                     : "Weighted RGB conversion  |  Exact scalar/AVX2 match (BT.601 weights)",
                 686, 138, 324, 30, Silver, labelFont);
        }

        // ── Image panels (equal size, 2-px gap) ───────────────────────────
        // Left panel: original
        glossyPanel(dc, {228, 198, 628, 660});
        text(dc, "ORIGINAL IMAGE", 248, 213, 250, 20, White, bodyFont);
        if (imageWidth > 0)
            text(dc, (std::to_string(imageWidth) + " x " +
                      std::to_string(imageHeight) + "  |  RGB").c_str(),
                 248, 213, 370, 16, Dim, smallFont, DT_RIGHT);
        else
            text(dc, "--", 248, 213, 370, 16, Dim, smallFont, DT_RIGHT);

        // Right panel: processed
        glossyPanel(dc, {630, 198, 1030, 660});
        text(dc, "PROCESSED IMAGE", 650, 213, 260, 20, White, bodyFont);
        if (processed && imageWidth > 0)
        {
            const std::string procMeta = std::to_string(imageWidth) + " x " +
                std::to_string(imageHeight) + "  |  " +
                (processedChannels == 3 ? "RGB (3-ch)" : "Grayscale (1-ch)");
            text(dc, procMeta.c_str(), 650, 213, 370, 16, Dim, smallFont, DT_RIGHT);
        }
        else
        {
            text(dc, "--", 650, 213, 370, 16, Dim, smallFont, DT_RIGHT);
        }
        std::string procModeStr = modeName(mode);
        if (mode == ProcessingMode::Gaussian)
        {
            procModeStr += " (" + std::to_string(gaussianPasses) + (gaussianPasses > 1 ? " passes)" : " pass)");
        }
        text(dc, procModeStr.c_str(), 650, 233, 260, 16, Dim, smallFont);

        // Image previews — both rects are identical height/width so images
        // are scaled the same way; centered with aspect ratio preserved.
        preview(dc, originalRgb,      3,                 {236, 242, 620, 616}, false);
        preview(dc, processedPixels,  processedChannels, {638, 242, 1022, 616}, true);

        // Status line below panels
        text(dc, lastImageName.c_str(), 248, 668, 360, 16, Dim, smallFont);
        text(dc, processed ? "OUTPUT READY" : "WAITING FOR PROCESSING",
             650, 668, 260, 16, processed ? Green : Dim, smallFont);

        // ── Performance cards (right sidebar) ────────────────────────────
        text(dc, "PERFORMANCE", 1060, 104, 200, 20, White, bodyFont);
        auto fmtMs = [](double v) -> std::string {
            std::ostringstream o;
            o << std::fixed << std::setprecision(2) << v << " ms";
            return o.str();
        };
        auto fmtX = [](double v) -> std::string {
            std::ostringstream o;
            o << std::fixed << std::setprecision(2) << v << " x";
            return o.str();
        };
        const std::string scStr = benchmarkRan ? fmtMs(scalarMedian) : "--";
        const std::string avStr = benchmarkRan ? fmtMs(avxMedian)    : "--";
        const std::string spStr = (benchmarkRan && avxMedian > 0.0)
                                  ? fmtX(scalarMedian / avxMedian) : "--";
        card(dc, {1060, 132, 1374, 214}, "SCALAR TIME",       scStr.c_str(), Silver);
        card(dc, {1060, 224, 1374, 306}, "AVX2 TIME",         avStr.c_str(), Accent);
        card(dc, {1060, 316, 1374, 398}, "SPEEDUP",           spStr.c_str(), Green);

        std::string valLabel = "NOT RUN";
        COLORREF valColor = Dim;
        if (!processed)
        {
            valLabel = "NOT RUN";
            valColor = Dim;
        }
        else if (avx2Supported)
        {
            if (outputsEqual)
            {
                valLabel = "PASS - max diff " + std::to_string(validationMaxDiff);
                valColor = Green;
            }
            else
            {
                valLabel = "FAIL";
                valColor = Red;
            }
        }
        else
        {
            valLabel = "SCALAR ONLY";
            valColor = Dim;
        }
        card(dc, {1060, 408, 1374, 496}, "OUTPUT VALIDATION",
             valLabel.c_str(), valColor);

        text(dc, "SYSTEM INFO", 1060, 528, 200, 20, White, bodyFont);
        const std::string imgDimStr = (imageWidth > 0)
            ? ("Image size  " + std::to_string(imageWidth) + " x " + std::to_string(imageHeight))
            : "Image size  --";
        text(dc, imgDimStr.c_str(), 1060, 558, 300, 18, Silver, labelFont);

        std::string chanStr = "Channels    --";
        if (!originalRgb.empty())
        {
            if (processed)
            {
                chanStr = "Channels    " + std::string(processedChannels == 3 ? "3 (RGB)" : "1 (Mono)") +
                          " [In: 3]";
            }
            else
            {
                chanStr = "Channels    3 (RGB in)";
            }
        }
        text(dc, chanStr.c_str(), 1060, 582, 300, 18, Silver, labelFont);

        const std::string avxStatus = avx2Supported ? "AVX2        SUPPORTED" : "AVX2        UNSUPPORTED";
        text(dc, avxStatus.c_str(), 1060, 606, 300, 18, avx2Supported ? Green : Red, labelFont);
        text(dc, "Build       GCC / Release x64", 1060, 630, 300, 18, Silver, labelFont);

        // ── Workspace / recent images ─────────────────────────────────────
        glossyPanel(dc, {228, 738, 1030, 858});
        text(dc, "RECENT IMAGES",    248, 754, 200, 20, White,  bodyFont);
        text(dc, lastImageName.c_str(), 248, 782, 350, 22, Silver, bodyFont);
        text(dc, "Current workspace image", 248, 808, 300, 18, Dim, smallFont);
        text(dc, "No history stored in this session",
             650, 826, 330, 18, Dim, smallFont, DT_RIGHT);

        line(dc, 228, 874, 1374, 874, Edge);
        text(dc, "PixelLane  v1.0.0",
             228, 882, 220, 16, Dim, smallFont);
        text(dc, "SIMD IMAGE PROCESSING ACCELERATOR - AVX2",
             500, 882, 300, 16, Dim, smallFont);
        text(dc, "NEON PURPLE  |  MINIMAL. SHARP. POWERFUL.",
             940, 882, 430, 16, Dim, smallFont, DT_RIGHT);

        EndPaint(window, &ps);
        return 0;
    }

    case WM_DESTROY:
        if (workerThread.joinable()) workerThread.join();
        workerRunning = false;
        if (bgBrush) { DeleteObject(bgBrush); bgBrush = nullptr; }
        PostQuitMessage(0);
        return 0;

    case WM_GETMINMAXINFO:
    {
        MINMAXINFO* info = (MINMAXINFO*)lParam;
        info->ptMinTrackSize.x = 1180;
        info->ptMinTrackSize.y = 820;
        return 0;
    }

    default:
        return DefWindowProcA(window, message, wParam, lParam);
    }
}

} // anonymous namespace

// -----------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------

int WINAPI WinMain(
    HINSTANCE instance, HINSTANCE, LPSTR, int showCommand)
{
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

    titleFont = CreateFontA(
        28, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_SWISS, "Segoe UI");
    headingFont = CreateFontA(
        22, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_SWISS, "Segoe UI");
    bodyFont = CreateFontA(
        15, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_SWISS, "Segoe UI");
    labelFont = CreateFontA(
        13, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_SWISS, "Segoe UI");
    smallFont = CreateFontA(
        12, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_SWISS, "Segoe UI");

    // Detect AVX2 capability on host CPU
    avx2Supported = detectCpuAvx2();
    if (!avx2Supported)
    {
        MessageBoxA(nullptr,
            "This CPU does not support AVX2. PixelLane will use the "
            "scalar processing path; AVX2 benchmarking is unavailable.",
            "PixelLane - AVX2 unavailable",
            MB_OK | MB_ICONWARNING);
    }

    // ---- Gaussian RGB correctness tests (run before window appears) ----
    if (avx2Supported)
    {
        std::string testErr;
        if (!runGaussianRgbEdgeCaseTests(testErr))
        {
            MessageBoxA(nullptr,
                (testErr + "\n\nThe application cannot start.\n"
                 "Scalar and AVX2 Gaussian RGB paths must produce identical output.").c_str(),
                "PixelLane - Correctness Test FAILED",
                MB_OK | MB_ICONERROR);
            return 1;
        }
    }
    // If we reach here all tests passed (or running in safe scalar fallback mode).
    // -------------------------------------------------------------------

    const char className[] = "PixelLaneStealthGraphite";
    WNDCLASSEXA wc = {};
    wc.cbSize        = sizeof(wc);
    wc.hInstance     = instance;
    wc.lpfnWndProc   = windowProcedure;
    wc.lpszClassName = className;
    wc.hIcon        = LoadIcon(instance, MAKEINTRESOURCE(1));
    wc.hIconSm      = (HICON)LoadImage(
        instance, MAKEINTRESOURCE(1), IMAGE_ICON,
        GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), 0);
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    if (!RegisterClassExA(&wc))
    {
        MessageBoxA(nullptr, "Could not register PixelLane.",
                    "Startup error", MB_OK | MB_ICONERROR);
        return 1;
    }

    windowHandle = CreateWindowA(
        className, "PixelLane  |  Stealth Graphite",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1280, 820,
        nullptr, nullptr, instance, nullptr);
    if (!windowHandle) return 1;

    const BOOL dark = TRUE;
    DwmSetWindowAttribute(windowHandle, DWMWA_USE_IMMERSIVE_DARK_MODE,
                          &dark, sizeof(dark));
    const COLORREF frame = RGB(18, 10, 28);
    DwmSetWindowAttribute(windowHandle, DWMWA_BORDER_COLOR,  &frame, sizeof(frame));
    DwmSetWindowAttribute(windowHandle, DWMWA_CAPTION_COLOR, &frame, sizeof(frame));

    // Navigation
    createControl("HOME",       NavHome,       28, 136, 160, 42);
    createControl("LOAD IMAGE", NavLoad,       28, 184, 160, 42);
    createControl("PROCESSING", NavProcessing, 28, 232, 160, 42);
    createControl("BENCHMARK",  NavBenchmark,  28, 280, 160, 42);
    createControl("ABOUT",      NavAbout,      28, 328, 160, 42);

    // Mode buttons
    createControl("GRAYSCALE",     ModeGray,     248, 132, 132, 42);
    createControl("SOBEL EDGE",    ModeSobel,    388, 132, 132, 42);
    createControl("GAUSSIAN BLUR", ModeGaussian, 528, 132, 140, 42);

    // Gaussian strength buttons (slots 16, 17, 18)
    createControl("1 PASS",   StrPass1, 686, 130, 92, 28);
    createControl("2 PASSES", StrPass2, 788, 130, 96, 28);
    createControl("3 PASSES", StrPass3, 894, 130, 96, 28);

    // Main action buttons
    createControl("PROCESS IMAGE",  ActionProcess,   248, 682, 190, 44);
    createControl("RUN BENCHMARK",  ActionBenchmark, 448, 682, 190, 44);
    createControl("SAVE AS",        ActionSave,      648, 682, 160, 44);
    createControl("CLEAR",          ActionClear,     818, 682, 140, 44);

    // Small helper buttons inside image panels
    createControl("OPEN IMAGE",  PanelOpen, 348, 624, 160, 30);
    createControl("VIEW RESULT", PanelView, 750, 624, 160, 30);

    statusLabel = CreateWindowA(
        "STATIC", "READY  |  Load an image to begin",
        WS_VISIBLE | WS_CHILD,
        228, 872, 1160, 28,
        windowHandle, (HMENU)(INT_PTR)StatusId, nullptr, nullptr);
    SendMessageA(statusLabel, WM_SETFONT, (WPARAM)bodyFont, TRUE);

    layoutControls(windowHandle);
    int actualShow = (showCommand == SW_HIDE || showCommand == 0) ? SW_SHOWNORMAL : showCommand;
    ShowWindow(windowHandle, actualShow);
    UpdateWindow(windowHandle);
    setEnabledState();
    InvalidateRect(windowHandle, nullptr, FALSE);

    MSG msg = {};
    while (GetMessageA(&msg, nullptr, 0, 0) > 0)
    {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }

    DeleteObject(titleFont);
    DeleteObject(headingFont);
    DeleteObject(bodyFont);
    DeleteObject(labelFont);
    DeleteObject(smallFont);
    return (int)msg.wParam;
}
