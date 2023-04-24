#include "Utilities.h"
#include <cuda.h>

__host__ __device__ int32_t cc__Max(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

__host__ __device__  float cc__Minf(float x, float y)
{
    return x < y ? x : y;
}

__host__ __device__   float cc__Maxf(float x, float y)
{
    return x > y ? x : y;
}

__host__ __device__   float cc__Satf(float x)
{
    return cc__Maxf(0.0f, cc__Minf(x, 1.0f));
}

__host__ __device__   float cc__Signf(float x)
{
    if (x < 0.0f) {
        return -1.0f;
    } else if (x > 0.0f) {
        return +1.0f;
    }

    return 0.0f;
}

__host__ __device__   void
cc__Lerpfv(int32_t n, float *out, const float *x, const float *y, float u)
{
    for (int32_t i = 0; i < n; ++i) {
        out[i] = x[i] + u * (y[i] - x[i]);
    }
}

__host__ __device__   void cc__Lerp2f(float *out, const float *x, const float *y, float u)
{
    cc__Lerpfv(2, out, x, y, u);
}

__host__ __device__   void cc__Lerp3f(float *out, const float *x, const float *y, float u)
{
    cc__Lerpfv(3, out, x, y, u);
}

__host__ __device__   void cc__Mulfv(int32_t n, float *out, const float *x, float y)
{
    for (int32_t i = 0; i < n; ++i) {
        out[i] = x[i] * y;
    }
}

__host__ __device__   void cc__Mul3f(float *out, const float *x, float y)
{
    cc__Mulfv(3, out, x, y);
}

__host__ __device__   void cc__Addfv(int32_t n, float *out, const float *x, const float *y)
{
    for (int32_t i = 0; i < n; ++i) {
        out[i] = x[i] + y[i];
    }
}

__host__ __device__   void cc__Add3f(float *out, const float *x, const float *y)
{
    cc__Addfv(3, out, x, y);
}


/*******************************************************************************
 * UV Encoding / Decoding routines
 *
 */
__host__ __device__  cc_VertexUv cc__DecodeUv(int32_t uvEncoded)
{
    const uint32_t tmp = (uint32_t)uvEncoded;
    const cc_VertexUv uv = {
        ((tmp >>  0) & 0xFFFF) / 65535.0f,
        ((tmp >> 16) & 0xFFFF) / 65535.0f
    };

    return uv;
}

__host__ __device__   int32_t cc__EncodeUv(const cc_VertexUv uv)
{
    const uint32_t u = uv.array[0] * 65535.0f;
    const uint32_t v = uv.array[1] * 65535.0f;
    const uint32_t tmp = ((u & 0xFFFFu) | ((v & 0xFFFFu) << 16));

    return (int32_t)tmp;
}
