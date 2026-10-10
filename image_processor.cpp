#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static std::uint8_t grayscalePixel(
    std::uint8_t red,
    std::uint8_t green,
    std::uint8_t blue)
{
    return static_cast<std::uint8_t>(
        (77 * red + 150 * green + 29 * blue) >> 8);
}

static void __attribute__((optimize("no-tree-vectorize"))) grayscaleScalar(
    const std::uint8_t* rgb,
    std::uint8_t* grayscale,
    std::size_t pixelCount)
{
    for (std::size_t pixel = 0; pixel < pixelCount; ++pixel)
    {
        const std::size_t sourceIndex = pixel * 3;

        grayscale[pixel] = grayscalePixel(
            rgb[sourceIndex],
            rgb[sourceIndex + 1],
            rgb[sourceIndex + 2]);
    }
}

static __m256i __attribute__((target("avx2"))) combineFourLanes(
    __m128i lower,
    __m128i upper)
{
    __m256i result = _mm256_castsi128_si256(lower);
    return _mm256_inserti128_si256(result, upper, 1);
}

static __m256i __attribute__((target("avx2"))) extractChannel(
    __m128i firstBytes,
    __m128i secondBytes,
    int firstOffset,
    int secondOffset)
{
    const __m128i firstMask = _mm_setr_epi8(
        static_cast<char>(firstOffset),
        static_cast<char>(firstOffset + 3),
        static_cast<char>(firstOffset + 6),
        static_cast<char>(firstOffset + 9),
        -1, -1, -1, -1,
        -1, -1, -1, -1,
        -1, -1, -1, -1);

    const __m128i secondMask = _mm_setr_epi8(
        static_cast<char>(secondOffset),
        static_cast<char>(secondOffset + 3),
        static_cast<char>(secondOffset + 6),
        static_cast<char>(secondOffset + 9),
        -1, -1, -1, -1,
        -1, -1, -1, -1,
        -1, -1, -1, -1);

    const __m256i firstValues =
        _mm256_cvtepu8_epi32(
            _mm_shuffle_epi8(firstBytes, firstMask));
    const __m256i secondValues =
        _mm256_cvtepu8_epi32(
            _mm_shuffle_epi8(secondBytes, secondMask));

    return combineFourLanes(
        _mm256_castsi256_si128(firstValues),
        _mm256_castsi256_si128(secondValues));
}

static void __attribute__((target("avx2"))) grayscaleEightAvx2(
    const std::uint8_t* rgb,
    std::uint8_t* grayscale)
{
    // Eight RGB pixels occupy 24 bytes. These overlapping 16-byte loads
    // cover bytes 0..15 and 8..23, so neither load exceeds this block.
    const __m128i firstBytes =
        _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(rgb));
    const __m128i secondBytes =
        _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(rgb + 8));

    const __m256i red =
        extractChannel(firstBytes, secondBytes, 0, 4);
    const __m256i green =
        extractChannel(firstBytes, secondBytes, 1, 5);
    const __m256i blue =
        extractChannel(firstBytes, secondBytes, 2, 6);

    // Keep the weighted sum in 32-bit lanes.
    __m256i total =
        _mm256_mullo_epi32(
            red,
            _mm256_set1_epi32(77));

    total =
        _mm256_add_epi32(
            total,
            _mm256_mullo_epi32(
                green,
                _mm256_set1_epi32(150)));

    total =
        _mm256_add_epi32(
            total,
            _mm256_mullo_epi32(
                blue,
                _mm256_set1_epi32(29)));

    const __m256i shifted =
        _mm256_srli_epi32(total, 8);

    const __m128i lower =
        _mm256_castsi256_si128(shifted);
    const __m128i upper =
        _mm256_extracti128_si256(shifted, 1);

    // Every result is 0..255, so unsigned saturation preserves it exactly.
    const __m128i packed16 =
        _mm_packus_epi32(lower, upper);
    const __m128i packed8 =
        _mm_packus_epi16(
            packed16,
            _mm_setzero_si128());

    std::memcpy(grayscale, &packed8, 8);
}

static void __attribute__((target("avx2"))) grayscaleAvx2(
    const std::uint8_t* rgb,
    std::uint8_t* grayscale,
    std::size_t pixelCount)
{
    std::size_t pixel = 0;

    for (; pixel + 8 <= pixelCount; pixel += 8)
    {
        grayscaleEightAvx2(
            rgb + pixel * 3,
            grayscale + pixel);
    }

    // Safely process one to seven remaining pixels.
    for (; pixel < pixelCount; ++pixel)
    {
        const std::size_t sourceIndex = pixel * 3;

        grayscale[pixel] = grayscalePixel(
            rgb[sourceIndex],
            rgb[sourceIndex + 1],
            rgb[sourceIndex + 2]);
    }
}

static std::uint8_t sobelPixel(
    const std::uint8_t* image,
    int width,
    int x,
    int y)
{
    if (x == 0 || y == 0 || x + 1 >= width)
    {
        return 0;
    }

    const std::size_t row = static_cast<std::size_t>(y) * width;
    const int gx =
        image[row - width + x + 1] +
        2 * image[row + x + 1] +
        image[row + width + x + 1] -
        image[row - width + x - 1] -
        2 * image[row + x - 1] -
        image[row + width + x - 1];
    const int gy =
        image[row + width + x - 1] +
        2 * image[row + width + x] +
        image[row + width + x + 1] -
        image[row - width + x - 1] -
        2 * image[row - width + x] -
        image[row - width + x + 1];
    return static_cast<std::uint8_t>(
        std::min(255, std::abs(gx) + std::abs(gy)));
}

static void __attribute__((optimize("no-tree-vectorize"))) sobelScalar(
    const std::uint8_t* image,
    std::uint8_t* output,
    int width,
    int height)
{
    const std::size_t pixelCount =
        static_cast<std::size_t>(width) * height;
    std::fill(output, output + pixelCount, 0);

    for (int y = 1; y + 1 < height; ++y)
    {
        for (int x = 1; x + 1 < width; ++x)
        {
            output[static_cast<std::size_t>(y) * width + x] =
                sobelPixel(image, width, x, y);
        }
    }
}

static __m256i __attribute__((target("avx2"))) loadEightU8(const std::uint8_t* values)
{
    std::uint64_t packed = 0;
    std::memcpy(&packed, values, sizeof(packed));
    return _mm256_cvtepu8_epi32(
        _mm_cvtsi64_si128(static_cast<long long>(packed)));
}

static void __attribute__((target("avx2"))) storeEightU8(
    std::uint8_t* output,
    __m256i values)
{
    const __m256i clamped =
        _mm256_min_epu32(values, _mm256_set1_epi32(255));
    const __m128i low = _mm256_castsi256_si128(clamped);
    const __m128i high = _mm256_extracti128_si256(clamped, 1);
    const __m128i packed16 = _mm_packus_epi32(low, high);
    const __m128i packed8 = _mm_packus_epi16(
        packed16,
        _mm_setzero_si128());
    std::memcpy(output, &packed8, 8);
}

static void __attribute__((target("avx2"))) sobelAvx2(
    const std::uint8_t* image,
    std::uint8_t* output,
    int width,
    int height)
{
    const std::size_t pixelCount =
        static_cast<std::size_t>(width) * height;
    std::fill(output, output + pixelCount, 0);

    for (int y = 1; y + 1 < height; ++y)
    {
        int x = 1;
        for (; x + 8 <= width - 1; x += 8)
        {
            const std::uint8_t* top = image +
                static_cast<std::size_t>(y - 1) * width;
            const std::uint8_t* middle = image +
                static_cast<std::size_t>(y) * width;
            const std::uint8_t* bottom = image +
                static_cast<std::size_t>(y + 1) * width;

            const __m256i tl = loadEightU8(top + x - 1);
            const __m256i tc = loadEightU8(top + x);
            const __m256i tr = loadEightU8(top + x + 1);
            const __m256i ml = loadEightU8(middle + x - 1);
            const __m256i mr = loadEightU8(middle + x + 1);
            const __m256i bl = loadEightU8(bottom + x - 1);
            const __m256i bc = loadEightU8(bottom + x);
            const __m256i br = loadEightU8(bottom + x + 1);

            __m256i gx = _mm256_add_epi32(
                _mm256_add_epi32(tr, br),
                _mm256_slli_epi32(mr, 1));
            gx = _mm256_sub_epi32(
                gx,
                _mm256_add_epi32(
                    _mm256_add_epi32(tl, bl),
                    _mm256_slli_epi32(ml, 1)));

            __m256i gy = _mm256_add_epi32(
                _mm256_add_epi32(bl, br),
                _mm256_slli_epi32(bc, 1));
            gy = _mm256_sub_epi32(
                gy,
                _mm256_add_epi32(
                    _mm256_add_epi32(tl, tr),
                    _mm256_slli_epi32(tc, 1)));

            gx = _mm256_abs_epi32(gx);
            gy = _mm256_abs_epi32(gy);
            storeEightU8(
                output + static_cast<std::size_t>(y) * width + x,
                _mm256_add_epi32(gx, gy));
        }

        for (; x + 1 < width; ++x)
        {
            output[static_cast<std::size_t>(y) * width + x] =
                sobelPixel(image, width, x, y);
        }
    }
}

static std::uint8_t gaussianPixel(
    const std::uint8_t* image,
    int width,
    int height,
    int x,
    int y)
{
    if (x == 0 || y == 0 || x + 1 >= width || y + 1 >= height)
    {
        return image[static_cast<std::size_t>(y) * width + x];
    }

    const std::size_t row = static_cast<std::size_t>(y) * width;
    const int sum =
        image[row - width + x - 1] +
        2 * image[row - width + x] +
        image[row - width + x + 1] +
        2 * image[row + x - 1] +
        4 * image[row + x] +
        2 * image[row + x + 1] +
        image[row + width + x - 1] +
        2 * image[row + width + x] +
        image[row + width + x + 1];
    return static_cast<std::uint8_t>((sum + 8) >> 4);
}

static void __attribute__((optimize("no-tree-vectorize"))) gaussianScalar(
    const std::uint8_t* image,
    std::uint8_t* output,
    int width,
    int height)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            output[static_cast<std::size_t>(y) * width + x] =
                gaussianPixel(image, width, height, x, y);
        }
    }
}

static void __attribute__((target("avx2"))) gaussianAvx2(
    const std::uint8_t* image,
    std::uint8_t* output,
    int width,
    int height)
{
    const std::size_t pixelCount =
        static_cast<std::size_t>(width) * height;
    std::memcpy(output, image, pixelCount);

    if (width < 3 || height < 3)
    {
        return;
    }

    for (int y = 1; y + 1 < height; ++y)
    {
        const std::size_t row = static_cast<std::size_t>(y) * width;
        int x = 1;

        for (x = 1; x + 8 <= width - 1; x += 8)
        {
            const std::uint8_t* top = image +
                static_cast<std::size_t>(y - 1) * width;
            const std::uint8_t* middle = image + row;
            const std::uint8_t* bottom = image +
                static_cast<std::size_t>(y + 1) * width;

            __m256i sum = _mm256_add_epi32(
                loadEightU8(top + x - 1),
                loadEightU8(top + x + 1));
            sum = _mm256_add_epi32(
                sum,
                _mm256_slli_epi32(loadEightU8(top + x), 1));
            sum = _mm256_add_epi32(
                sum,
                _mm256_add_epi32(
                    _mm256_slli_epi32(loadEightU8(middle + x - 1), 1),
                    _mm256_slli_epi32(loadEightU8(middle + x + 1), 1)));
            sum = _mm256_add_epi32(
                sum,
                _mm256_slli_epi32(loadEightU8(middle + x), 2));
            sum = _mm256_add_epi32(
                sum,
                _mm256_add_epi32(
                    loadEightU8(bottom + x - 1),
                    loadEightU8(bottom + x + 1)));
            sum = _mm256_add_epi32(
                sum,
                _mm256_slli_epi32(loadEightU8(bottom + x), 1));
            storeEightU8(
                output + row + x,
                _mm256_srli_epi32(
                    _mm256_add_epi32(sum, _mm256_set1_epi32(8)),
                    4));
        }

        for (; x + 1 < width; ++x)
        {
            output[row + x] = gaussianPixel(image, width, height, x, y);
        }
    }
}

static bool outputsMatch(
    const std::uint8_t* scalar,
    const std::uint8_t* simd,
    std::size_t pixelCount)
{
    for (std::size_t pixel = 0; pixel < pixelCount; ++pixel)
    {
        if (scalar[pixel] != simd[pixel])
        {
            std::cerr << "Error: scalar and AVX2 outputs differ at pixel "
                      << pixel << ".\n"
                      << "Scalar value: "
                      << static_cast<int>(scalar[pixel]) << '\n'
                      << "SIMD value: "
                      << static_cast<int>(simd[pixel]) << '\n';
            return false;
        }
    }

    return true;
}

static bool runEdgeCaseTests()
{
    // Exercise zero-to-seven tail pixels and complete eight-pixel blocks.
    for (std::size_t pixelCount = 1; pixelCount <= 17; ++pixelCount)
    {
        std::vector<std::uint8_t> rgb(pixelCount * 3);
        std::vector<std::uint8_t> scalar(pixelCount);
        std::vector<std::uint8_t> simd(pixelCount);

        for (std::size_t index = 0; index < rgb.size(); ++index)
        {
            rgb[index] = static_cast<std::uint8_t>(
                (index * 37 + 19) & 0xFF);
        }

        grayscaleScalar(
            rgb.data(),
            scalar.data(),
            pixelCount);

        grayscaleAvx2(
            rgb.data(),
            simd.data(),
            pixelCount);

        if (!outputsMatch(
                scalar.data(),
                simd.data(),
                pixelCount))
        {
            std::cerr << "Error: edge-case test failed for "
                      << pixelCount << " pixels.\n";
            return false;
        }
    }

    return true;
}

static double medianTime(std::vector<double> times)
{
    std::sort(times.begin(), times.end());

    const std::size_t middle = times.size() / 2;

    if (times.size() % 2 == 0)
    {
        return (times[middle - 1] + times[middle]) / 2.0;
    }

    return times[middle];
}

#ifndef IMAGE_PROCESSOR_NO_MAIN
int main()
{
    constexpr const char* inputFilename = "input.jpg";
    constexpr const char* outputFilename = "output_grayscale.png";
    constexpr int timedTrials = 10;

    if (!runEdgeCaseTests())
    {
        return 1;
    }

    int width = 0;
    int height = 0;
    int originalChannels = 0;

    // Request exactly three channels: red, green, and blue.
    std::uint8_t* loadedImage = stbi_load(
        inputFilename,
        &width,
        &height,
        &originalChannels,
        3);

    if (loadedImage == nullptr)
    {
        std::cerr << "Error: could not load '"
                  << inputFilename << "'.\n"
                  << "stb_image reason: "
                  << stbi_failure_reason() << '\n';
        return 1;
    }

    if (width <= 0 || height <= 0)
    {
        std::cerr << "Error: image has invalid dimensions.\n";
        stbi_image_free(loadedImage);
        return 1;
    }

    const std::size_t pixelCount =
        static_cast<std::size_t>(width) *
        static_cast<std::size_t>(height);

    std::vector<std::uint8_t> scalarGrayscale(pixelCount);
    std::vector<std::uint8_t> simdGrayscale(pixelCount);
    std::vector<double> scalarTimes;
    std::vector<double> simdTimes;

    scalarTimes.reserve(timedTrials);
    simdTimes.reserve(timedTrials);

    // Warm-up runs are not included in the reported timings.
    grayscaleScalar(
        loadedImage,
        scalarGrayscale.data(),
        pixelCount);

    grayscaleAvx2(
        loadedImage,
        simdGrayscale.data(),
        pixelCount);

    if (!outputsMatch(
            scalarGrayscale.data(),
            simdGrayscale.data(),
            pixelCount))
    {
        stbi_image_free(loadedImage);
        return 1;
    }

    for (int trial = 0; trial < timedTrials; ++trial)
    {
        const auto scalarStart = Clock::now();

        grayscaleScalar(
            loadedImage,
            scalarGrayscale.data(),
            pixelCount);

        const auto scalarEnd = Clock::now();

        const auto simdStart = Clock::now();

        grayscaleAvx2(
            loadedImage,
            simdGrayscale.data(),
            pixelCount);

        const auto simdEnd = Clock::now();

        scalarTimes.push_back(
            std::chrono::duration<double, std::milli>(
                scalarEnd - scalarStart).count());

        simdTimes.push_back(
            std::chrono::duration<double, std::milli>(
                simdEnd - simdStart).count());
    }

    // Verify the results produced by the final timed trial.
    if (!outputsMatch(
            scalarGrayscale.data(),
            simdGrayscale.data(),
            pixelCount))
    {
        stbi_image_free(loadedImage);
        return 1;
    }

    // Save only after the SIMD result has passed verification.
    if (stbi_write_png(
            outputFilename,
            width,
            height,
            1,
            simdGrayscale.data(),
            width) == 0)
    {
        std::cerr << "Error: could not save '"
                  << outputFilename << "'.\n";
        stbi_image_free(loadedImage);
        return 1;
    }

    stbi_image_free(loadedImage);

    const double scalarMedianMs =
        medianTime(scalarTimes);
    const double simdMedianMs =
        medianTime(simdTimes);

    std::cout << "Image loaded: "
              << inputFilename << '\n';
    std::cout << "Image size: "
              << width << " x " << height << '\n';
    std::cout << "Edge-case tests: PASS (1 through 17 pixels)\n";
    std::cout << "Scalar and AVX2 outputs match exactly.\n";
    std::cout << "Saved grayscale image: "
              << outputFilename << '\n';
    std::cout << "Warm-up runs: 1 per implementation\n";
    std::cout << "Timed trials: "
              << timedTrials << " per implementation\n";
    std::cout << "Scalar median processing time: "
              << scalarMedianMs << " ms\n";
    std::cout << "AVX2 median processing time: "
              << simdMedianMs << " ms\n";

    if (simdMedianMs > 0.0)
    {
        std::cout << "Median speedup: "
                  << scalarMedianMs / simdMedianMs
                  << "x\n";
    }

    return 0;
}
#endif
