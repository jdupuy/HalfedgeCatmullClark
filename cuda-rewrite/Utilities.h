#pragma once
#include <cuda.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef CC_LOG
#    include <stdio.h>
#    define CC_LOG(format, ...) do { fprintf(stdout, format "\n", ##__VA_ARGS__); fflush(stdout); } while(0)
#endif

// uv data
typedef union {
    struct {float u, v;};
    float array[2];
} cc_VertexUv;

__host__ __device__ int32_t cc__Max(int32_t a, int32_t b);
__host__ __device__  float cc__Minf(float x, float y);
__host__ __device__   float cc__Maxf(float x, float y);
__host__ __device__   float cc__Satf(float x);
__host__ __device__   float cc__Signf(float x);

__host__ __device__   void cc__Lerpfv(int32_t n, float *out, const float *x, const float *y, float u);

__host__ __device__   void cc__Lerp2f(float *out, const float *x, const float *y, float u);
__host__ __device__   void cc__Lerp3f(float *out, const float *x, const float *y, float u);

__host__ __device__   void cc__Mulfv(int32_t n, float *out, const float *x, float y);

__host__ __device__   void cc__Mul3f(float *out, const float *x, float y);

__host__ __device__  void cc__Addfv(int32_t n, float *out, const float *x, const float *y);
__host__ __device__  void cc__Add3f(float *out, const float *x, const float *y);
__host__ __device__  cc_VertexUv cc__DecodeUv(int32_t uvEncoded);
__host__ __device__  int32_t cc__EncodeUv(const cc_VertexUv uv);