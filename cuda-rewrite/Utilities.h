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
    struct {double u, v;};
    double array[2];
} cc_VertexUv;

__host__ __device__ int32_t cc__Max(int32_t a, int32_t b);
__host__ __device__  double cc__Minf(double x, double y);
__host__ __device__   double cc__Maxf(double x, double y);
__host__ __device__   double cc__Satf(double x);
__host__ __device__   double cc__Signf(double x);

__host__ __device__   void cc__Lerpfv(int32_t n, double *out, const double *x, const double *y, double u);

__host__ __device__   void cc__Lerp2f(double *out, const double *x, const double *y, double u);
__host__ __device__   void cc__Lerp3f(double *out, const double *x, const double *y, double u);

__host__ __device__   void cc__Mulfv(int32_t n, double *out, const double *x, double y);

__host__ __device__   void cc__Mul3f(double *out, const double *x, double y);

__host__ __device__  void cc__Addfv(int32_t n, double *out, const double *x, const double *y);
__host__ __device__  void cc__Add3f(double *out, const double *x, const double *y);
__host__ __device__  cc_VertexUv cc__DecodeUv(int32_t uvEncoded);
__host__ __device__  int32_t cc__EncodeUv(const cc_VertexUv uv);