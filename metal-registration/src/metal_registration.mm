// metal_registration.mm — Metal compute backend for BROCCOLI image registration
// Objective-C++ required for Metal API

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
// Accelerate framework no longer needed — using custom Gaussian elimination
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>

#include "metal_registration.h"

// ============================================================
//  Internal helpers
// ============================================================

namespace metal_reg {
namespace {

struct Dims {
    int W, H, D;
};

struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;

    // Pipeline states (lazily created)
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines;

    MetalContext() : pipelines(nil) {}

    void init() {
        device = MTLCreateSystemDefaultDevice();
        assert(device && "No Metal device found");
        queue = [device newCommandQueue];

        // Load shader library from metallib file next to executable,
        // or compile from source
        NSError* err = nil;
        NSString* libPath = [[NSBundle mainBundle] pathForResource:@"registration" ofType:@"metallib"];
        if (libPath) {
            NSURL* libURL = [NSURL fileURLWithPath:libPath];
            library = [device newLibraryWithURL:libURL error:&err];
        }

        if (!library) {
            // Try loading from the same directory as this code
            // Look for the .metal source and compile it
            NSString* srcPath = nil;

            // Search in several candidate locations
            NSArray* candidates = @[
                @"src/shaders/registration.metal",
                @"shaders/registration.metal",
                @"registration.metal",
            ];

            NSFileManager* fm = [NSFileManager defaultManager];

            // Try relative to the process working directory
            for (NSString* cand in candidates) {
                if ([fm fileExistsAtPath:cand]) { srcPath = cand; break; }
            }

            // Try relative to the executable
            if (!srcPath) {
                NSString* execDir = [[[NSProcessInfo processInfo] arguments][0] stringByDeletingLastPathComponent];
                for (NSString* cand in candidates) {
                    NSString* full = [execDir stringByAppendingPathComponent:cand];
                    if ([fm fileExistsAtPath:full]) { srcPath = full; break; }
                }
            }

            // Try from environment variable
            if (!srcPath) {
                const char* envPath = getenv("METAL_SHADER_PATH");
                if (envPath) {
                    srcPath = [NSString stringWithUTF8String:envPath];
                }
            }

            if (srcPath) {
                NSString* src = [NSString stringWithContentsOfFile:srcPath
                                                         encoding:NSUTF8StringEncoding error:&err];
                if (src) {
                    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
                    opts.mathMode = MTLMathModeSafe;
                    library = [device newLibraryWithSource:src options:opts error:&err];
                }
            }
        }

        if (!library) {
            NSLog(@"Failed to create Metal library: %@", err);
            assert(false && "Could not load Metal shaders");
        }

        pipelines = [NSMutableDictionary dictionary];
    }

    id<MTLComputePipelineState> getPipeline(const char* name) {
        NSString* key = [NSString stringWithUTF8String:name];
        id<MTLComputePipelineState> ps = pipelines[key];
        if (!ps) {
            id<MTLFunction> fn = [library newFunctionWithName:key];
            assert(fn && "Shader function not found");
            NSError* err = nil;
            ps = [device newComputePipelineStateWithFunction:fn error:&err];
            assert(ps && "Failed to create pipeline state");
            pipelines[key] = ps;
        }
        return ps;
    }

    id<MTLBuffer> newBuffer(size_t bytes) {
        return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }

    id<MTLBuffer> newBuffer(const void* data, size_t bytes) {
        return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    }

    id<MTLTexture> newTexture3D(int W, int H, int D) {
        MTLTextureDescriptor* desc = [MTLTextureDescriptor new];
        desc.textureType = MTLTextureType3D;
        desc.pixelFormat = MTLPixelFormatR32Float;
        desc.width = W;
        desc.height = H;
        desc.depth = D;
        desc.storageMode = MTLStorageModeShared;
        desc.usage = MTLTextureUsageShaderRead;
        return [device newTextureWithDescriptor:desc];
    }

    void copyBufferToTexture(id<MTLBuffer> buf, id<MTLTexture> tex, int W, int H, int D) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:buf
               sourceOffset:0
          sourceBytesPerRow:W * sizeof(float)
        sourceBytesPerImage:W * H * sizeof(float)
                 sourceSize:MTLSizeMake(W, H, D)
                  toTexture:tex
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
};

// Singleton context
MetalContext& ctx() {
    static MetalContext c;
    static bool inited = false;
    if (!inited) { c.init(); inited = true; }
    return c;
}

// ============================================================
//  Dispatch helpers
// ============================================================

void dispatch3D(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps,
                int W, int H, int D) {
    NSUInteger tw = ps.threadExecutionWidth;
    NSUInteger th = ps.maxTotalThreadsPerThreadgroup / tw;
    MTLSize threads = MTLSizeMake(W, H, D);
    MTLSize tgSize = MTLSizeMake(tw, std::min(th, (NSUInteger)H), 1);
    [enc setComputePipelineState:ps];
    [enc dispatchThreads:threads threadsPerThreadgroup:tgSize];
}

void dispatch1D(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps,
                int count) {
    NSUInteger tw = std::min((NSUInteger)count, ps.maxTotalThreadsPerThreadgroup);
    [enc setComputePipelineState:ps];
    [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
}

void dispatch2D(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps,
                int W, int H) {
    NSUInteger tw = ps.threadExecutionWidth;
    NSUInteger th = std::min(ps.maxTotalThreadsPerThreadgroup / tw, (NSUInteger)H);
    [enc setComputePipelineState:ps];
    [enc dispatchThreads:MTLSizeMake(W, H, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

// ============================================================
//  GPU operations
// ============================================================

void fillBuffer(id<MTLBuffer> buf, float value, int count) {
    auto& c = ctx();
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    auto ps = c.getPipeline("fillFloat");
    [enc setComputePipelineState:ps];
    [enc setBuffer:buf offset:0 atIndex:0];
    id<MTLBuffer> valBuf = c.newBuffer(&value, sizeof(float));
    [enc setBuffer:valBuf offset:0 atIndex:1];
    dispatch1D(enc, ps, count);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void fillFloat2Buffer(id<MTLBuffer> buf, int count) {
    auto& c = ctx();
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    auto ps = c.getPipeline("fillFloat2");
    [enc setComputePipelineState:ps];
    [enc setBuffer:buf offset:0 atIndex:0];
    float zero[2] = {0, 0};
    id<MTLBuffer> valBuf = c.newBuffer(zero, sizeof(float) * 2);
    [enc setBuffer:valBuf offset:0 atIndex:1];
    dispatch1D(enc, ps, count);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void addVolumes(id<MTLBuffer> A, id<MTLBuffer> B, int count) {
    auto& c = ctx();
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    auto ps = c.getPipeline("addVolumes");
    [enc setComputePipelineState:ps];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    dispatch1D(enc, ps, count);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void multiplyVolume(id<MTLBuffer> vol, float factor, int count) {
    auto& c = ctx();
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    auto ps = c.getPipeline("multiplyVolume");
    [enc setComputePipelineState:ps];
    [enc setBuffer:vol offset:0 atIndex:0];
    id<MTLBuffer> fBuf = c.newBuffer(&factor, sizeof(float));
    [enc setBuffer:fBuf offset:0 atIndex:1];
    dispatch1D(enc, ps, count);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void multiplyVolumes(id<MTLBuffer> A, id<MTLBuffer> B, int count) {
    auto& c = ctx();
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    auto ps = c.getPipeline("multiplyVolumes");
    [enc setComputePipelineState:ps];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    dispatch1D(enc, ps, count);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

float calculateMax(id<MTLBuffer> volume, int W, int H, int D) {
    auto& c = ctx();
    Dims dims = {W, H, D};

    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
    id<MTLBuffer> colMaxs = c.newBuffer(H * D * sizeof(float));
    id<MTLBuffer> rowMaxs = c.newBuffer(D * sizeof(float));

    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    auto ps1 = c.getPipeline("calculateColumnMaxs");
    [enc setComputePipelineState:ps1];
    [enc setBuffer:colMaxs offset:0 atIndex:0];
    [enc setBuffer:volume offset:0 atIndex:1];
    [enc setBuffer:dimBuf offset:0 atIndex:2];
    dispatch2D(enc, ps1, H, D);

    auto ps2 = c.getPipeline("calculateRowMaxs");
    [enc setComputePipelineState:ps2];
    [enc setBuffer:rowMaxs offset:0 atIndex:0];
    [enc setBuffer:colMaxs offset:0 atIndex:1];
    [enc setBuffer:dimBuf offset:0 atIndex:2];
    dispatch1D(enc, ps2, D);

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    float* rowData = (float*)[rowMaxs contents];
    float mx = rowData[0];
    for (int i = 1; i < D; i++) mx = std::max(mx, rowData[i]);
    return mx;
}

// ============================================================
//  3D Nonseparable Convolution (3 quadrature filters)
// ============================================================

void nonseparableConvolution3D(
    id<MTLBuffer> resp1, id<MTLBuffer> resp2, id<MTLBuffer> resp3,
    id<MTLBuffer> volume,
    const float* filterReal1, const float* filterImag1,
    const float* filterReal2, const float* filterImag2,
    const float* filterReal3, const float* filterImag3,
    int W, int H, int D)
{
    auto& c = ctx();
    int vol = W * H * D;

    // Zero-init response buffers
    fillFloat2Buffer(resp1, vol);
    fillFloat2Buffer(resp2, vol);
    fillFloat2Buffer(resp3, vol);

    auto ps = c.getPipeline("nonseparableConv3D_ThreeFilters");
    Dims dims = {W, H, D};
    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));

    // For each z-slice of the 7x7x7 filter (reverse z iteration to match OpenCL)
    int zOff = -3;
    for (int zz = 6; zz >= 0; zz--) {
      @autoreleasepool {
        // Extract 7x7 slice from each 7x7x7 filter
        int zSlice = zz;
        float slice1R[49], slice1I[49], slice2R[49], slice2I[49], slice3R[49], slice3I[49];
        for (int fy = 0; fy < 7; fy++) {
            for (int fx = 0; fx < 7; fx++) {
                int fi2d = fx + fy * 7;
                int fi3d = fx + fy * 7 + zSlice * 49;
                slice1R[fi2d] = filterReal1[fi3d];
                slice1I[fi2d] = filterImag1[fi3d];
                slice2R[fi2d] = filterReal2[fi3d];
                slice2I[fi2d] = filterImag2[fi3d];
                slice3R[fi2d] = filterReal3[fi3d];
                slice3I[fi2d] = filterImag3[fi3d];
            }
        }

        id<MTLBuffer> f1r = c.newBuffer(slice1R, 49 * sizeof(float));
        id<MTLBuffer> f1i = c.newBuffer(slice1I, 49 * sizeof(float));
        id<MTLBuffer> f2r = c.newBuffer(slice2R, 49 * sizeof(float));
        id<MTLBuffer> f2i = c.newBuffer(slice2I, 49 * sizeof(float));
        id<MTLBuffer> f3r = c.newBuffer(slice3R, 49 * sizeof(float));
        id<MTLBuffer> f3i = c.newBuffer(slice3I, 49 * sizeof(float));
        id<MTLBuffer> zBuf = c.newBuffer(&zOff, sizeof(int));

        id<MTLCommandBuffer> cb = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:resp1 offset:0 atIndex:0];
        [enc setBuffer:resp2 offset:0 atIndex:1];
        [enc setBuffer:resp3 offset:0 atIndex:2];
        [enc setBuffer:volume offset:0 atIndex:3];
        [enc setBuffer:f1r offset:0 atIndex:4];
        [enc setBuffer:f1i offset:0 atIndex:5];
        [enc setBuffer:f2r offset:0 atIndex:6];
        [enc setBuffer:f2i offset:0 atIndex:7];
        [enc setBuffer:f3r offset:0 atIndex:8];
        [enc setBuffer:f3i offset:0 atIndex:9];
        [enc setBuffer:zBuf offset:0 atIndex:10];
        [enc setBuffer:dimBuf offset:0 atIndex:11];

        // Dispatch with threadgroup size including halos
        int validW = 32 - 2 * 3; // 26
        int validH = 32 - 2 * 3; // 26
        int groupsX = (W + validW - 1) / validW;
        int groupsY = (H + validH - 1) / validH;
        [enc dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, D)
            threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        zOff++;
      }
    }
}

// ============================================================
//  Separable smoothing (3-pass: rows, columns, rods)
// ============================================================

void performSmoothing(id<MTLBuffer> output, id<MTLBuffer> input, int W, int H, int D,
                      const float* smoothingFilter) {
    auto& c = ctx();
    Dims dims = {W, H, D};
    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
    id<MTLBuffer> filterBuf = c.newBuffer(smoothingFilter, 9 * sizeof(float));
    id<MTLBuffer> temp1 = c.newBuffer(W * H * D * sizeof(float));
    id<MTLBuffer> temp2 = c.newBuffer(W * H * D * sizeof(float));

    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    // Rows (along y)
    auto ps1 = c.getPipeline("separableConvRows");
    [enc setComputePipelineState:ps1];
    [enc setBuffer:temp1 offset:0 atIndex:0];
    [enc setBuffer:input offset:0 atIndex:1];
    [enc setBuffer:filterBuf offset:0 atIndex:2];
    [enc setBuffer:dimBuf offset:0 atIndex:3];
    dispatch3D(enc, ps1, W, H, D);

    // Barrier: pass 2 reads temp1 written by pass 1
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Columns (along x)
    auto ps2 = c.getPipeline("separableConvColumns");
    [enc setComputePipelineState:ps2];
    [enc setBuffer:temp2 offset:0 atIndex:0];
    [enc setBuffer:temp1 offset:0 atIndex:1];
    [enc setBuffer:filterBuf offset:0 atIndex:2];
    [enc setBuffer:dimBuf offset:0 atIndex:3];
    dispatch3D(enc, ps2, W, H, D);

    // Barrier: pass 3 reads temp2 written by pass 2
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Rods (along z)
    auto ps3 = c.getPipeline("separableConvRods");
    [enc setComputePipelineState:ps3];
    [enc setBuffer:output offset:0 atIndex:0];
    [enc setBuffer:temp2 offset:0 atIndex:1];
    [enc setBuffer:filterBuf offset:0 atIndex:2];
    [enc setBuffer:dimBuf offset:0 atIndex:3];
    dispatch3D(enc, ps3, W, H, D);

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// In-place smoothing
void performSmoothingInPlace(id<MTLBuffer> volume, int W, int H, int D,
                             const float* smoothingFilter) {
    auto& c = ctx();
    id<MTLBuffer> output = c.newBuffer(W * H * D * sizeof(float));
    performSmoothing(output, volume, W, H, D, smoothingFilter);
    // GPU-to-GPU copy (avoids CPU roundtrip)
    NSUInteger bytes = W * H * D * sizeof(float);
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:output sourceOffset:0 toBuffer:volume destinationOffset:0 size:bytes];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// ============================================================
//  Create smoothing filter (Gaussian, same as BROCCOLI)
// ============================================================

void createSmoothingFilter(float* filter, float sigma) {
    float sum = 0;
    for (int i = 0; i < 9; i++) {
        float x = float(i) - 4.0f;
        filter[i] = expf(-0.5f * x * x / (sigma * sigma));
        sum += filter[i];
    }
    for (int i = 0; i < 9; i++) filter[i] /= sum;
}

// ============================================================
//  Rescale volume (change voxel size)
// ============================================================

id<MTLBuffer> rescaleVolume(id<MTLBuffer> input, int srcW, int srcH, int srcD,
                            int dstW, int dstH, int dstD,
                            float scaleX, float scaleY, float scaleZ) {
    auto& c = ctx();
    id<MTLTexture> tex = c.newTexture3D(srcW, srcH, srcD);
    c.copyBufferToTexture(input, tex, srcW, srcH, srcD);

    id<MTLBuffer> output = c.newBuffer(dstW * dstH * dstD * sizeof(float));
    fillBuffer(output, 0.0f, dstW * dstH * dstD);

    Dims dims = {dstW, dstH, dstD};
    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
    id<MTLBuffer> sxBuf = c.newBuffer(&scaleX, sizeof(float));
    id<MTLBuffer> syBuf = c.newBuffer(&scaleY, sizeof(float));
    id<MTLBuffer> szBuf = c.newBuffer(&scaleZ, sizeof(float));

    auto ps = c.getPipeline("rescaleVolumeLinear");
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:output offset:0 atIndex:0];
    [enc setTexture:tex atIndex:0];
    [enc setBuffer:sxBuf offset:0 atIndex:1];
    [enc setBuffer:syBuf offset:0 atIndex:2];
    [enc setBuffer:szBuf offset:0 atIndex:3];
    [enc setBuffer:dimBuf offset:0 atIndex:4];
    dispatch3D(enc, ps, dstW, dstH, dstD);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    return output;
}

// ============================================================
//  Copy volume to new dimensions (crop/pad)
// ============================================================

id<MTLBuffer> copyVolumeToNew(id<MTLBuffer> src,
                               int srcW, int srcH, int srcD,
                               int dstW, int dstH, int dstD,
                               int mmZCut, float voxelSizeZ) {
    auto& c = ctx();
    id<MTLBuffer> dst = c.newBuffer(dstW * dstH * dstD * sizeof(float));
    fillBuffer(dst, 0.0f, dstW * dstH * dstD);

    struct CopyParams {
        int newW, newH, newD;
        int srcW, srcH, srcD;
        int xDiff, yDiff, zDiff;
        int mmZCut;
        float voxelSizeZ;
    };

    CopyParams cp = {
        dstW, dstH, dstD,
        srcW, srcH, srcD,
        srcW - dstW, srcH - dstH, srcD - dstD,
        mmZCut, voxelSizeZ
    };

    id<MTLBuffer> paramBuf = c.newBuffer(&cp, sizeof(CopyParams));

    // Dispatch over the smaller of src/dst dimensions
    int dispW = std::min(srcW, dstW);
    int dispH = std::min(srcH, dstH);
    int dispD = std::min(srcD, dstD);

    auto ps = c.getPipeline("copyVolumeToNew");
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:dst offset:0 atIndex:0];
    [enc setBuffer:src offset:0 atIndex:1];
    [enc setBuffer:paramBuf offset:0 atIndex:2];
    dispatch3D(enc, ps, dispW, dispH, dispD);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    return dst;
}

// ============================================================
//  Change volume resolution and size
// ============================================================

id<MTLBuffer> changeVolumesResolutionAndSize(
    id<MTLBuffer> input, int srcW, int srcH, int srcD,
    VoxelSize srcVox, int dstW, int dstH, int dstD,
    VoxelSize dstVox, int mmZCut)
{
    // Step 1: Rescale to match voxel sizes
    float scaleX = srcVox.x / dstVox.x;
    float scaleY = srcVox.y / dstVox.y;
    float scaleZ = srcVox.z / dstVox.z;

    int interpW = (int)roundf(srcW * scaleX);
    int interpH = (int)roundf(srcH * scaleY);
    int interpD = (int)roundf(srcD * scaleZ);

    id<MTLBuffer> interpolated = rescaleVolume(input, srcW, srcH, srcD,
                                                interpW, interpH, interpD,
                                                1.0f / scaleX, 1.0f / scaleY, 1.0f / scaleZ);

    // Step 2: Copy to target dimensions (crop/pad)
    return copyVolumeToNew(interpolated, interpW, interpH, interpD,
                           dstW, dstH, dstD, mmZCut, dstVox.z);
}

// ============================================================
//  Center-of-mass calculation (CPU-side)
// ============================================================

void centerOfMass(const float* vol, int W, int H, int D,
                  float& cx, float& cy, float& cz) {
    double sum = 0, sx = 0, sy = 0, sz = 0;
    for (int z = 0; z < D; z++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                float v = vol[x + y * W + z * W * H];
                if (v > 0) {
                    sum += v;
                    sx += v * x;
                    sy += v * y;
                    sz += v * z;
                }
            }
    if (sum > 0) {
        cx = sx / sum;
        cy = sy / sum;
        cz = sz / sum;
    } else {
        cx = W * 0.5f;
        cy = H * 0.5f;
        cz = D * 0.5f;
    }
}

// ============================================================
//  Affine interpolation (GPU)
// ============================================================

void interpolateLinear(id<MTLBuffer> output, id<MTLBuffer> volume,
                       const float* params, int W, int H, int D) {
    auto& c = ctx();
    id<MTLTexture> tex = c.newTexture3D(W, H, D);
    c.copyBufferToTexture(volume, tex, W, H, D);

    Dims dims = {W, H, D};
    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
    id<MTLBuffer> paramBuf = c.newBuffer(params, 12 * sizeof(float));

    auto ps = c.getPipeline("interpolateLinearLinear");
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:output offset:0 atIndex:0];
    [enc setTexture:tex atIndex:0];
    [enc setBuffer:paramBuf offset:0 atIndex:1];
    [enc setBuffer:dimBuf offset:0 atIndex:2];
    dispatch3D(enc, ps, W, H, D);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// ============================================================
//  Nonlinear interpolation (GPU)
// ============================================================

void interpolateNonLinear(id<MTLBuffer> output, id<MTLBuffer> volume,
                          id<MTLBuffer> dispX, id<MTLBuffer> dispY, id<MTLBuffer> dispZ,
                          int W, int H, int D) {
    auto& c = ctx();
    id<MTLTexture> tex = c.newTexture3D(W, H, D);
    c.copyBufferToTexture(volume, tex, W, H, D);

    Dims dims = {W, H, D};
    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));

    auto ps = c.getPipeline("interpolateLinearNonLinear");
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:output offset:0 atIndex:0];
    [enc setTexture:tex atIndex:0];
    [enc setBuffer:dispX offset:0 atIndex:1];
    [enc setBuffer:dispY offset:0 atIndex:2];
    [enc setBuffer:dispZ offset:0 atIndex:3];
    [enc setBuffer:dimBuf offset:0 atIndex:4];
    dispatch3D(enc, ps, W, H, D);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// ============================================================
//  Add linear + nonlinear displacement (GPU)
// ============================================================

void addLinearNonLinearDisplacement(id<MTLBuffer> dispX, id<MTLBuffer> dispY, id<MTLBuffer> dispZ,
                                    const float* params, int W, int H, int D) {
    auto& c = ctx();
    Dims dims = {W, H, D};
    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
    id<MTLBuffer> paramBuf = c.newBuffer(params, 12 * sizeof(float));

    auto ps = c.getPipeline("addLinearAndNonLinearDisplacement");
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:dispX offset:0 atIndex:0];
    [enc setBuffer:dispY offset:0 atIndex:1];
    [enc setBuffer:dispZ offset:0 atIndex:2];
    [enc setBuffer:paramBuf offset:0 atIndex:3];
    [enc setBuffer:dimBuf offset:0 atIndex:4];
    dispatch3D(enc, ps, W, H, D);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// ============================================================
//  Solve equation system (CPU, via Accelerate LAPACK)
// ============================================================

void solveEquationSystem(float* A, float* h, double* params, int n) {
    // Convert to double for precision
    double Ad[144], hd[12];
    for (int i = 0; i < n * n; i++) Ad[i] = A[i];
    for (int i = 0; i < n; i++) hd[i] = h[i];

    // Mirror symmetric matrix
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            Ad[i * n + j] = Ad[j * n + i];

    // Solve using simple Gaussian elimination (avoids LAPACK deprecation issues)
    // Make augmented matrix: [A | h]
    double aug[12][13];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            aug[i][j] = Ad[j * n + i]; // column-major to row-major
        aug[i][n] = hd[i];
    }

    // Forward elimination with partial pivoting
    for (int col = 0; col < n; col++) {
        // Find pivot
        int pivotRow = col;
        double pivotVal = fabs(aug[col][col]);
        for (int row = col + 1; row < n; row++) {
            if (fabs(aug[row][col]) > pivotVal) {
                pivotVal = fabs(aug[row][col]);
                pivotRow = row;
            }
        }
        if (pivotVal < 1e-30) {
            for (int i = 0; i < n; i++) params[i] = 0;
            return;
        }
        if (pivotRow != col) {
            for (int j = 0; j <= n; j++)
                std::swap(aug[col][j], aug[pivotRow][j]);
        }
        // Eliminate
        for (int row = col + 1; row < n; row++) {
            double factor = aug[row][col] / aug[col][col];
            for (int j = col; j <= n; j++)
                aug[row][j] -= factor * aug[col][j];
        }
    }

    // Back substitution
    for (int row = n - 1; row >= 0; row--) {
        double sum = aug[row][n];
        for (int j = row + 1; j < n; j++)
            sum -= aug[row][j] * params[j];
        params[row] = sum / aug[row][row];
    }
}

// ============================================================
//  Affine parameter composition via 4x4 matrix multiplication
//  Matches BROCCOLI's AddAffineRegistrationParameters
// ============================================================

// Build a 4x4 affine matrix from 12 parameters:
//   (p3+1  p4   p5   tx)
//   (p6    p7+1 p8   ty)
//   (p9    p10  p11+1 tz)
//   (0     0    0     1 )
static void paramsToMatrix(const float* p, double M[4][4], float translationScale = 1.0f) {
    M[0][0] = p[3] + 1.0; M[0][1] = p[4];       M[0][2] = p[5];       M[0][3] = p[0] * translationScale;
    M[1][0] = p[6];       M[1][1] = p[7] + 1.0;  M[1][2] = p[8];       M[1][3] = p[1] * translationScale;
    M[2][0] = p[9];       M[2][1] = p[10];       M[2][2] = p[11] + 1.0; M[2][3] = p[2] * translationScale;
    M[3][0] = 0;          M[3][1] = 0;           M[3][2] = 0;           M[3][3] = 1.0;
}

static void matrixToParams(const double M[4][4], float* p) {
    p[0] = (float)M[0][3];
    p[1] = (float)M[1][3];
    p[2] = (float)M[2][3];
    p[3] = (float)(M[0][0] - 1.0);
    p[4] = (float)M[0][1];
    p[5] = (float)M[0][2];
    p[6] = (float)M[1][0];
    p[7] = (float)(M[1][1] - 1.0);
    p[8] = (float)M[1][2];
    p[9] = (float)M[2][0];
    p[10] = (float)M[2][1];
    p[11] = (float)(M[2][2] - 1.0);
}

static void matMul4x4(const double A[4][4], const double B[4][4], double C[4][4]) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            C[i][j] = 0;
            for (int k = 0; k < 4; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

// Compose: result = New * Old (result stored in oldParams)
void composeAffineParams(float* oldParams, const float* newParams) {
    double O[4][4], N[4][4], T[4][4];
    paramsToMatrix(oldParams, O);
    paramsToMatrix(newParams, N);
    matMul4x4(N, O, T);
    matrixToParams(T, oldParams);
}

// Compose for next scale: translations scaled by 2x before matrix multiply
void composeAffineParamsNextScale(float* oldParams, const float* newParams) {
    double O[4][4], N[4][4], T[4][4];
    paramsToMatrix(oldParams, O, 2.0f);  // old translations * 2
    paramsToMatrix(newParams, N, 2.0f);  // new translations * 2
    matMul4x4(N, O, T);
    matrixToParams(T, oldParams);
}

// 3-arg version: result = New * Old, stored in resultParams
void composeAffineParams3(float* resultParams, const float* oldParams, const float* newParams) {
    double O[4][4], N[4][4], T[4][4];
    paramsToMatrix(oldParams, O);
    paramsToMatrix(newParams, N);
    matMul4x4(N, O, T);
    matrixToParams(T, resultParams);
}

// ============================================================
//  Change volume size (multi-scale rescaling)
// ============================================================

id<MTLBuffer> changeVolumeSize(id<MTLBuffer> input, int srcW, int srcH, int srcD,
                                int dstW, int dstH, int dstD) {
    float scaleX = float(srcW) / float(dstW);
    float scaleY = float(srcH) / float(dstH);
    float scaleZ = float(srcD) / float(dstD);
    return rescaleVolume(input, srcW, srcH, srcD, dstW, dstH, dstD,
                         scaleX, scaleY, scaleZ);
}

// ============================================================
//  LINEAR REGISTRATION
// ============================================================

// Single scale, single iteration batch
void alignTwoVolumesLinear(
    id<MTLBuffer> alignedVolume,
    id<MTLBuffer> referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D,
    int filterSize,
    int numIterations,
    float* registrationParams,  // 12 params, accumulated
    bool verbose)
{
    auto& c = ctx();
    int vol = W * H * D;

    // Save the original aligned volume — re-interpolate from this each iteration
    id<MTLBuffer> originalAligned = c.newBuffer(vol * sizeof(float));
    memcpy([originalAligned contents], [alignedVolume contents], vol * sizeof(float));

    // Reset params to zero — each scale solves independently
    // (BROCCOLI resets h_Registration_Parameters_Align_Two_Volumes to 0 at start)
    for (int i = 0; i < 12; i++) registrationParams[i] = 0.0f;

    // Allocate filter response buffers (complex, float2)
    id<MTLBuffer> q11 = c.newBuffer(vol * sizeof(float) * 2);
    id<MTLBuffer> q12 = c.newBuffer(vol * sizeof(float) * 2);
    id<MTLBuffer> q13 = c.newBuffer(vol * sizeof(float) * 2);
    id<MTLBuffer> q21 = c.newBuffer(vol * sizeof(float) * 2);
    id<MTLBuffer> q22 = c.newBuffer(vol * sizeof(float) * 2);
    id<MTLBuffer> q23 = c.newBuffer(vol * sizeof(float) * 2);

    // Phase/certainty buffers
    id<MTLBuffer> phaseDiff = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> certainties = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> phaseGrad = c.newBuffer(vol * sizeof(float));

    // A-matrix / h-vector buffers — must be zero-initialized
    int HD = H * D;
    id<MTLBuffer> A2D = c.newBuffer(30 * HD * sizeof(float));
    id<MTLBuffer> A1D = c.newBuffer(30 * D * sizeof(float));
    id<MTLBuffer> Amat = c.newBuffer(144 * sizeof(float));
    id<MTLBuffer> h2D = c.newBuffer(12 * HD * sizeof(float));
    id<MTLBuffer> h1D = c.newBuffer(12 * D * sizeof(float));
    id<MTLBuffer> hvec = c.newBuffer(12 * sizeof(float));
    memset([A2D contents], 0, 30 * HD * sizeof(float));
    memset([A1D contents], 0, 30 * D * sizeof(float));
    memset([h2D contents], 0, 12 * HD * sizeof(float));
    memset([h1D contents], 0, 12 * D * sizeof(float));

    // Filter reference volume once
    nonseparableConvolution3D(q11, q12, q13, referenceVolume,
        filters.linearReal[0].data(), filters.linearImag[0].data(),
        filters.linearReal[1].data(), filters.linearImag[1].data(),
        filters.linearReal[2].data(), filters.linearImag[2].data(),
        W, H, D);

    Dims dims = {W, H, D};
    id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));

    for (int iter = 0; iter < numIterations; iter++) {
      @autoreleasepool {
        // Filter aligned volume
        nonseparableConvolution3D(q21, q22, q23, alignedVolume,
            filters.linearReal[0].data(), filters.linearImag[0].data(),
            filters.linearReal[1].data(), filters.linearImag[1].data(),
            filters.linearReal[2].data(), filters.linearImag[2].data(),
            W, H, D);

        // Process each direction (X, Y, Z)
        struct { int dirOff; int hOff; id<MTLBuffer> q1; id<MTLBuffer> q2; const char* gradKernel; }
        dirs[3] = {
            {0, 0, q11, q21, "calculatePhaseGradientsX"},
            {10, 1, q12, q22, "calculatePhaseGradientsY"},
            {20, 2, q13, q23, "calculatePhaseGradientsZ"},
        };

        // Zero intermediate buffers before each iteration
        fillBuffer(phaseDiff, 0.0f, vol);
        fillBuffer(certainties, 0.0f, vol);
        fillBuffer(phaseGrad, 0.0f, vol);
        memset([h2D contents], 0, 12 * HD * sizeof(float));
        memset([A2D contents], 0, 30 * HD * sizeof(float));

        for (int d = 0; d < 3; d++) {
            // Phase differences and certainties
            {
                auto ps = c.getPipeline("calculatePhaseDifferencesAndCertainties");
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:ps];
                [enc setBuffer:phaseDiff offset:0 atIndex:0];
                [enc setBuffer:certainties offset:0 atIndex:1];
                [enc setBuffer:dirs[d].q1 offset:0 atIndex:2];
                [enc setBuffer:dirs[d].q2 offset:0 atIndex:3];
                [enc setBuffer:dimBuf offset:0 atIndex:4];
                dispatch3D(enc, ps, W, H, D);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            // Phase gradients
            {
                auto ps = c.getPipeline(dirs[d].gradKernel);
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:ps];
                [enc setBuffer:phaseGrad offset:0 atIndex:0];
                [enc setBuffer:dirs[d].q1 offset:0 atIndex:1];
                [enc setBuffer:dirs[d].q2 offset:0 atIndex:2];
                [enc setBuffer:dimBuf offset:0 atIndex:3];
                dispatch3D(enc, ps, W, H, D);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            // A-matrix and h-vector 2D values
            {
                struct AParams {
                    int W, H, D, filterSize, dirOff, hOff;
                };
                AParams ap = {W, H, D, filterSize, dirs[d].dirOff, dirs[d].hOff};
                id<MTLBuffer> apBuf = c.newBuffer(&ap, sizeof(AParams));

                auto ps = c.getPipeline("calculateAMatrixAndHVector2D");
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:ps];
                [enc setBuffer:A2D offset:0 atIndex:0];
                [enc setBuffer:h2D offset:0 atIndex:1];
                [enc setBuffer:phaseDiff offset:0 atIndex:2];
                [enc setBuffer:phaseGrad offset:0 atIndex:3];
                [enc setBuffer:certainties offset:0 atIndex:4];
                [enc setBuffer:apBuf offset:0 atIndex:5];
                dispatch2D(enc, ps, H, D);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }

        // Reduce A-matrix: 2D -> 1D -> final
        // Each step in its own command buffer to avoid ordering issues
        {
            id<MTLBuffer> hBuf = c.newBuffer(&H, sizeof(int));
            id<MTLBuffer> dBuf = c.newBuffer(&D, sizeof(int));
            id<MTLBuffer> fsBuf = c.newBuffer(&filterSize, sizeof(int));

            // A-matrix 1D
            {
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                auto ps = c.getPipeline("calculateAMatrix1D");
                [enc setComputePipelineState:ps];
                [enc setBuffer:A1D offset:0 atIndex:0];
                [enc setBuffer:A2D offset:0 atIndex:1];
                [enc setBuffer:hBuf offset:0 atIndex:2];
                [enc setBuffer:dBuf offset:0 atIndex:3];
                [enc setBuffer:fsBuf offset:0 atIndex:4];
                dispatch2D(enc, ps, D, 30);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            // Reset and compute A-matrix final
            fillBuffer(Amat, 0.0f, 144);
            {
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                auto ps = c.getPipeline("calculateAMatrixFinal");
                [enc setComputePipelineState:ps];
                [enc setBuffer:Amat offset:0 atIndex:0];
                [enc setBuffer:A1D offset:0 atIndex:1];
                [enc setBuffer:dBuf offset:0 atIndex:2];
                [enc setBuffer:fsBuf offset:0 atIndex:3];
                dispatch1D(enc, ps, 30);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            // h-vector 1D
            {
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                auto ps = c.getPipeline("calculateHVector1D");
                [enc setComputePipelineState:ps];
                [enc setBuffer:h1D offset:0 atIndex:0];
                [enc setBuffer:h2D offset:0 atIndex:1];
                [enc setBuffer:hBuf offset:0 atIndex:2];
                [enc setBuffer:dBuf offset:0 atIndex:3];
                [enc setBuffer:fsBuf offset:0 atIndex:4];
                dispatch2D(enc, ps, D, 12);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            // h-vector final
            {
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                auto ps = c.getPipeline("calculateHVectorFinal");
                [enc setComputePipelineState:ps];
                [enc setBuffer:hvec offset:0 atIndex:0];
                [enc setBuffer:h1D offset:0 atIndex:1];
                [enc setBuffer:dBuf offset:0 atIndex:2];
                [enc setBuffer:fsBuf offset:0 atIndex:3];
                dispatch1D(enc, ps, 12);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }

        // Read back A and h, solve on CPU
        float hA[144], hh[12];
        memcpy(hA, [Amat contents], 144 * sizeof(float));

        // Do h-vector final reduction on CPU to bypass GPU kernel issue
        {
            int fhalf = (filterSize - 1) / 2;
            float* h1Dp = (float*)[h1D contents];
            for (int elem = 0; elem < 12; elem++) {
                float sum = 0;
                for (int z = fhalf; z < D - fhalf; z++) {
                    sum += h1Dp[elem * D + z];
                }
                hh[elem] = sum;
            }
        }

        double paramsDbl[12];
        solveEquationSystem(hA, hh, paramsDbl, 12);

        // Compose parameters via matrix multiplication (BROCCOLI pattern)
        float deltaParams[12];
        for (int i = 0; i < 12; i++) deltaParams[i] = (float)paramsDbl[i];
        composeAffineParams(registrationParams, deltaParams);

        // Apply affine transform from original volume (not already-transformed)
        interpolateLinear(alignedVolume, originalAligned, registrationParams, W, H, D);
      }
    }
}

// Multi-scale linear registration
// Matches BROCCOLI's AlignTwoVolumesLinearSeveralScales pattern:
// - Reset total params to zero
// - At each scale: downscale from originals, pre-transform with accumulated params,
//   solve independently from zero, compose result via matrix multiplication
void alignTwoVolumesLinearSeveralScales(
    id<MTLBuffer>& alignedVolume,
    id<MTLBuffer> referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D,
    int filterSize,
    int numIterations,
    int coarsestScale,
    float* registrationParams,
    bool verbose)
{
    auto& c = ctx();
    int vol = W * H * D;

    // Reset total parameters (BROCCOLI pattern)
    for (int i = 0; i < 12; i++) registrationParams[i] = 0.0f;

    // Keep a copy of the original full-resolution aligned volume
    id<MTLBuffer> originalAligned = c.newBuffer(vol * sizeof(float));
    memcpy([originalAligned contents], [alignedVolume contents], vol * sizeof(float));

    // Start from coarsest scale, work to finest
    for (int scale = coarsestScale; scale >= 1; scale /= 2) {
      @autoreleasepool {
        int sW = (int)roundf((float)W / (float)scale);
        int sH = (int)roundf((float)H / (float)scale);
        int sD = (int)roundf((float)D / (float)scale);

        if (sW < 8 || sH < 8 || sD < 8) continue;

        // Downscale both volumes from originals at each scale
        id<MTLBuffer> scaledRef = (scale == 1) ? referenceVolume :
            changeVolumeSize(referenceVolume, W, H, D, sW, sH, sD);
        id<MTLBuffer> scaledAligned = (scale == 1) ? alignedVolume :
            changeVolumeSize(originalAligned, W, H, D, sW, sH, sD);

        // For non-coarsest scales: pre-transform with accumulated params
        if (scale < coarsestScale) {
            interpolateLinear(scaledAligned, scaledAligned, registrationParams, sW, sH, sD);
        }

        // Fewer iterations at finest scale (BROCCOLI: ceil(N/5))
        int iters = (scale == 1) ? (int)ceilf((float)numIterations / 5.0f) : numIterations;

        if (verbose) {
            printf("  Linear scale %d: %dx%dx%d, %d iterations\n", scale, sW, sH, sD, iters);
        }

        // Temp params for this scale (starts at zero inside alignTwoVolumesLinear)
        float tempParams[12] = {0};

        alignTwoVolumesLinear(scaledAligned, scaledRef, filters,
                              sW, sH, sD, filterSize, iters,
                              tempParams, verbose);

        // Compose this scale's params with accumulated total
        if (scale != 1) {
            // NextScale variant: translations * 2 before matrix multiply
            // After this, total params are at the next finer scale's resolution
            composeAffineParamsNextScale(registrationParams, tempParams);
        } else {
            // Final scale: standard composition (no 2x scaling)
            // Result is in full-resolution coordinates
            composeAffineParams(registrationParams, tempParams);
        }
      }
    }

    // Final transform of original volume with complete parameters (full resolution)
    interpolateLinear(alignedVolume, originalAligned, registrationParams, W, H, D);
}

// ============================================================
//  NONLINEAR REGISTRATION
// ============================================================

void alignTwoVolumesNonLinear(
    id<MTLBuffer> alignedVolume,
    id<MTLBuffer> referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D,
    int numIterations,
    id<MTLBuffer> updateDispX, id<MTLBuffer> updateDispY, id<MTLBuffer> updateDispZ,
    bool verbose)
{
    auto& c = ctx();
    int vol = W * H * D;
    int filterSize = 7;

    // Allocate filter response buffers (6 filters, 2 volumes = 12 complex buffers)
    id<MTLBuffer> q1[6], q2[6];
    for (int i = 0; i < 6; i++) {
        q1[i] = c.newBuffer(vol * sizeof(float) * 2);
        q2[i] = c.newBuffer(vol * sizeof(float) * 2);
    }

    // Tensor components
    id<MTLBuffer> t11 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> t12 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> t13 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> t22 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> t23 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> t33 = c.newBuffer(vol * sizeof(float));

    // A-matrix (6 unique) and h-vector (3)
    id<MTLBuffer> a11 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> a12 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> a13 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> a22 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> a23 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> a33 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> h1 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> h2 = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> h3 = c.newBuffer(vol * sizeof(float));

    // Tensor norms
    id<MTLBuffer> tensorNorms = c.newBuffer(vol * sizeof(float));

    // Displacement update
    id<MTLBuffer> dux = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> duy = c.newBuffer(vol * sizeof(float));
    id<MTLBuffer> duz = c.newBuffer(vol * sizeof(float));

    // Save the original aligned volume — re-interpolate from this each iteration
    id<MTLBuffer> originalAligned = c.newBuffer(vol * sizeof(float));
    memcpy([originalAligned contents], [alignedVolume contents], vol * sizeof(float));

    // Filter directions
    id<MTLBuffer> fdxBuf = c.newBuffer(filters.filterDirectionsX, 6 * sizeof(float));
    id<MTLBuffer> fdyBuf = c.newBuffer(filters.filterDirectionsY, 6 * sizeof(float));
    id<MTLBuffer> fdzBuf = c.newBuffer(filters.filterDirectionsZ, 6 * sizeof(float));

    // Smoothing filters
    float smoothTensor[9], smoothEq[9], smoothDisp[9];
    createSmoothingFilter(smoothTensor, 1.0f);
    createSmoothingFilter(smoothEq, 2.0f);
    createSmoothingFilter(smoothDisp, 2.0f);

    Dims dims = {W, H, D};

    // Filter reference volume (once per scale)
    nonseparableConvolution3D(q1[0], q1[1], q1[2], referenceVolume,
        filters.nonlinearReal[0].data(), filters.nonlinearImag[0].data(),
        filters.nonlinearReal[1].data(), filters.nonlinearImag[1].data(),
        filters.nonlinearReal[2].data(), filters.nonlinearImag[2].data(),
        W, H, D);
    nonseparableConvolution3D(q1[3], q1[4], q1[5], referenceVolume,
        filters.nonlinearReal[3].data(), filters.nonlinearImag[3].data(),
        filters.nonlinearReal[4].data(), filters.nonlinearImag[4].data(),
        filters.nonlinearReal[5].data(), filters.nonlinearImag[5].data(),
        W, H, D);

    for (int iter = 0; iter < numIterations; iter++) {
      @autoreleasepool {
        if (verbose) printf("    Nonlinear iter %d/%d\n", iter + 1, numIterations);

        // Filter aligned volume
        nonseparableConvolution3D(q2[0], q2[1], q2[2], alignedVolume,
            filters.nonlinearReal[0].data(), filters.nonlinearImag[0].data(),
            filters.nonlinearReal[1].data(), filters.nonlinearImag[1].data(),
            filters.nonlinearReal[2].data(), filters.nonlinearImag[2].data(),
            W, H, D);
        nonseparableConvolution3D(q2[3], q2[4], q2[5], alignedVolume,
            filters.nonlinearReal[3].data(), filters.nonlinearImag[3].data(),
            filters.nonlinearReal[4].data(), filters.nonlinearImag[4].data(),
            filters.nonlinearReal[5].data(), filters.nonlinearImag[5].data(),
            W, H, D);

        // Reset tensor components and displacement
        for (auto buf : {t11, t12, t13, t22, t23, t33, dux, duy, duz}) {
            fillBuffer(buf, 0.0f, vol);
        }

        // Calculate tensor components (6 filters)
        for (int f = 0; f < 6; f++) {
          @autoreleasepool {
            const float* pt = filters.projectionTensors[f];
            id<MTLBuffer> mBufs[6];
            for (int k = 0; k < 6; k++)
                mBufs[k] = c.newBuffer(&pt[k], sizeof(float));

            id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));

            auto ps = c.getPipeline("calculateTensorComponents");
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBuffer:t11 offset:0 atIndex:0];
            [enc setBuffer:t12 offset:0 atIndex:1];
            [enc setBuffer:t13 offset:0 atIndex:2];
            [enc setBuffer:t22 offset:0 atIndex:3];
            [enc setBuffer:t23 offset:0 atIndex:4];
            [enc setBuffer:t33 offset:0 atIndex:5];
            [enc setBuffer:q1[f] offset:0 atIndex:6];
            [enc setBuffer:q2[f] offset:0 atIndex:7];
            for (int k = 0; k < 6; k++)
                [enc setBuffer:mBufs[k] offset:0 atIndex:8 + k];
            [enc setBuffer:dimBuf offset:0 atIndex:14];
            dispatch3D(enc, ps, W, H, D);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
          }
        }

        // Compute tensor norms (before smoothing, for normalization)
        {
            id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
            auto ps = c.getPipeline("calculateTensorNorms");
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBuffer:tensorNorms offset:0 atIndex:0];
            [enc setBuffer:t11 offset:0 atIndex:1];
            [enc setBuffer:t12 offset:0 atIndex:2];
            [enc setBuffer:t13 offset:0 atIndex:3];
            [enc setBuffer:t22 offset:0 atIndex:4];
            [enc setBuffer:t23 offset:0 atIndex:5];
            [enc setBuffer:t33 offset:0 atIndex:6];
            [enc setBuffer:dimBuf offset:0 atIndex:7];
            dispatch3D(enc, ps, W, H, D);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        // Smooth tensor components
        for (auto buf : {t11, t12, t13, t22, t23, t33}) {
            performSmoothingInPlace(buf, W, H, D, smoothTensor);
        }

        // Recompute tensor norms after smoothing
        {
            id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
            auto ps = c.getPipeline("calculateTensorNorms");
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBuffer:tensorNorms offset:0 atIndex:0];
            [enc setBuffer:t11 offset:0 atIndex:1];
            [enc setBuffer:t12 offset:0 atIndex:2];
            [enc setBuffer:t13 offset:0 atIndex:3];
            [enc setBuffer:t22 offset:0 atIndex:4];
            [enc setBuffer:t23 offset:0 atIndex:5];
            [enc setBuffer:t33 offset:0 atIndex:6];
            [enc setBuffer:dimBuf offset:0 atIndex:7];
            dispatch3D(enc, ps, W, H, D);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        // Find max tensor norm and normalize
        float maxNorm = calculateMax(tensorNorms, W, H, D);
        if (maxNorm > 0) {
            float invMax = 1.0f / maxNorm;
            for (auto buf : {t11, t12, t13, t22, t23, t33}) {
                multiplyVolume(buf, invMax, vol);
            }
        }

        // Calculate A-matrices and h-vectors (6 filters)
        for (int f = 0; f < 6; f++) {
          @autoreleasepool {
            struct MorphonParams { int W, H, D, FILTER; };
            MorphonParams mp = {W, H, D, f};
            id<MTLBuffer> mpBuf = c.newBuffer(&mp, sizeof(MorphonParams));

            auto ps = c.getPipeline("calculateAMatricesAndHVectors");
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBuffer:a11 offset:0 atIndex:0];
            [enc setBuffer:a12 offset:0 atIndex:1];
            [enc setBuffer:a13 offset:0 atIndex:2];
            [enc setBuffer:a22 offset:0 atIndex:3];
            [enc setBuffer:a23 offset:0 atIndex:4];
            [enc setBuffer:a33 offset:0 atIndex:5];
            [enc setBuffer:h1 offset:0 atIndex:6];
            [enc setBuffer:h2 offset:0 atIndex:7];
            [enc setBuffer:h3 offset:0 atIndex:8];
            [enc setBuffer:q1[f] offset:0 atIndex:9];
            [enc setBuffer:q2[f] offset:0 atIndex:10];
            [enc setBuffer:t11 offset:0 atIndex:11];
            [enc setBuffer:t12 offset:0 atIndex:12];
            [enc setBuffer:t13 offset:0 atIndex:13];
            [enc setBuffer:t22 offset:0 atIndex:14];
            [enc setBuffer:t23 offset:0 atIndex:15];
            [enc setBuffer:t33 offset:0 atIndex:16];
            [enc setBuffer:fdxBuf offset:0 atIndex:17];
            [enc setBuffer:fdyBuf offset:0 atIndex:18];
            [enc setBuffer:fdzBuf offset:0 atIndex:19];
            [enc setBuffer:mpBuf offset:0 atIndex:20];
            dispatch3D(enc, ps, W, H, D);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
          }
        }

        // Smooth A-matrix and h-vector components
        for (auto buf : {a11, a12, a13, a22, a23, a33, h1, h2, h3}) {
            performSmoothingInPlace(buf, W, H, D, smoothEq);
        }

        // Calculate displacement update
        {
            id<MTLBuffer> dimBuf = c.newBuffer(&dims, sizeof(Dims));
            auto ps = c.getPipeline("calculateDisplacementUpdate");
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBuffer:dux offset:0 atIndex:0];
            [enc setBuffer:duy offset:0 atIndex:1];
            [enc setBuffer:duz offset:0 atIndex:2];
            [enc setBuffer:a11 offset:0 atIndex:3];
            [enc setBuffer:a12 offset:0 atIndex:4];
            [enc setBuffer:a13 offset:0 atIndex:5];
            [enc setBuffer:a22 offset:0 atIndex:6];
            [enc setBuffer:a23 offset:0 atIndex:7];
            [enc setBuffer:a33 offset:0 atIndex:8];
            [enc setBuffer:h1 offset:0 atIndex:9];
            [enc setBuffer:h2 offset:0 atIndex:10];
            [enc setBuffer:h3 offset:0 atIndex:11];
            [enc setBuffer:dimBuf offset:0 atIndex:12];
            dispatch3D(enc, ps, W, H, D);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        // Smooth displacement update
        for (auto buf : {dux, duy, duz}) {
            performSmoothingInPlace(buf, W, H, D, smoothDisp);
        }

        // Accumulate displacement
        addVolumes(updateDispX, dux, vol);
        addVolumes(updateDispY, duy, vol);
        addVolumes(updateDispZ, duz, vol);

        // Interpolate from original volume with accumulated displacement
        interpolateNonLinear(alignedVolume, originalAligned,
                             updateDispX, updateDispY, updateDispZ, W, H, D);
      }
    }
}

// Multi-scale nonlinear registration
void alignTwoVolumesNonLinearSeveralScales(
    id<MTLBuffer>& alignedVolume,
    id<MTLBuffer> referenceVolume,
    const QuadratureFilters& filters,
    int W, int H, int D,
    int numIterations,
    int coarsestScale,
    id<MTLBuffer>& totalDispX, id<MTLBuffer>& totalDispY, id<MTLBuffer>& totalDispZ,
    bool verbose)
{
    auto& c = ctx();
    int vol = W * H * D;

    // Keep a copy of the original full-resolution aligned volume
    id<MTLBuffer> originalAligned = c.newBuffer(vol * sizeof(float));
    memcpy([originalAligned contents], [alignedVolume contents], vol * sizeof(float));

    totalDispX = c.newBuffer(vol * sizeof(float));
    totalDispY = c.newBuffer(vol * sizeof(float));
    totalDispZ = c.newBuffer(vol * sizeof(float));
    fillBuffer(totalDispX, 0.0f, vol);
    fillBuffer(totalDispY, 0.0f, vol);
    fillBuffer(totalDispZ, 0.0f, vol);

    for (int scale = coarsestScale; scale >= 1; scale /= 2) {
      @autoreleasepool {
        int sW = W / scale;
        int sH = H / scale;
        int sD = D / scale;

        if (sW < 8 || sH < 8 || sD < 8) continue;

        if (verbose) printf("  Nonlinear scale %d: %dx%dx%d\n", scale, sW, sH, sD);

        id<MTLBuffer> scaledRef = (scale == 1) ? referenceVolume :
            changeVolumeSize(referenceVolume, W, H, D, sW, sH, sD);
        // Always downscale from current aligned (which was transformed at full res)
        id<MTLBuffer> scaledAligned = (scale == 1) ? alignedVolume :
            changeVolumeSize(alignedVolume, W, H, D, sW, sH, sD);

        int sVol = sW * sH * sD;
        id<MTLBuffer> updateX = c.newBuffer(sVol * sizeof(float));
        id<MTLBuffer> updateY = c.newBuffer(sVol * sizeof(float));
        id<MTLBuffer> updateZ = c.newBuffer(sVol * sizeof(float));
        fillBuffer(updateX, 0.0f, sVol);
        fillBuffer(updateY, 0.0f, sVol);
        fillBuffer(updateZ, 0.0f, sVol);

        int iters = numIterations;

        alignTwoVolumesNonLinear(scaledAligned, scaledRef, filters,
                                 sW, sH, sD, iters,
                                 updateX, updateY, updateZ, verbose);

        // Accumulate into total displacement
        if (scale > 1) {
            // Rescale displacement to full resolution
            id<MTLBuffer> rescX = changeVolumeSize(updateX, sW, sH, sD, W, H, D);
            id<MTLBuffer> rescY = changeVolumeSize(updateY, sW, sH, sD, W, H, D);
            id<MTLBuffer> rescZ = changeVolumeSize(updateZ, sW, sH, sD, W, H, D);
            multiplyVolume(rescX, (float)scale, vol);
            multiplyVolume(rescY, (float)scale, vol);
            multiplyVolume(rescZ, (float)scale, vol);
            addVolumes(totalDispX, rescX, vol);
            addVolumes(totalDispY, rescY, vol);
            addVolumes(totalDispZ, rescZ, vol);

            // Re-interpolate from original aligned volume at full resolution
            interpolateNonLinear(alignedVolume, originalAligned,
                                 totalDispX, totalDispY, totalDispZ, W, H, D);
        } else {
            addVolumes(totalDispX, updateX, vol);
            addVolumes(totalDispY, updateY, vol);
            addVolumes(totalDispZ, updateZ, vol);
        }
      }
    }
}

} // anonymous namespace

// ============================================================
//  PUBLIC API
// ============================================================

EPIT1Result registerEPIT1(
    const float* epiData, VolumeDims epiDims, VoxelSize epiVox,
    const float* t1Data, VolumeDims t1Dims, VoxelSize t1Vox,
    const QuadratureFilters& filters,
    int numIterations, int coarsestScale, int mmZCut, bool verbose)
{
    auto& c = ctx();

    if (verbose) printf("registerEPIT1: EPI %dx%dx%d -> T1 %dx%dx%d\n",
                        epiDims.W, epiDims.H, epiDims.D,
                        t1Dims.W, t1Dims.H, t1Dims.D);

    int t1Vol = t1Dims.size();

    // Upload volumes to GPU
    id<MTLBuffer> t1Buf = c.newBuffer(t1Data, t1Vol * sizeof(float));
    id<MTLBuffer> epiBuf = c.newBuffer(epiData, epiDims.size() * sizeof(float));

    // Resample EPI to T1 resolution and size
    id<MTLBuffer> epiInT1 = changeVolumesResolutionAndSize(
        epiBuf, epiDims.W, epiDims.H, epiDims.D, epiVox,
        t1Dims.W, t1Dims.H, t1Dims.D, t1Vox, mmZCut);

    // Save interpolated volume before alignment
    std::vector<float> interpResult(t1Vol);
    memcpy(interpResult.data(), [epiInT1 contents], t1Vol * sizeof(float));

    // Center-of-mass alignment
    float cx1, cy1, cz1, cx2, cy2, cz2;
    centerOfMass((float*)[t1Buf contents], t1Dims.W, t1Dims.H, t1Dims.D, cx1, cy1, cz1);
    centerOfMass((float*)[epiInT1 contents], t1Dims.W, t1Dims.H, t1Dims.D, cx2, cy2, cz2);

    float initParams[12] = {0};
    initParams[0] = cx1 - cx2;
    initParams[1] = cy1 - cy2;
    initParams[2] = cz1 - cz2;

    // Apply initial translation
    interpolateLinear(epiInT1, epiInT1, initParams, t1Dims.W, t1Dims.H, t1Dims.D);

    // Multi-scale linear registration (rigid body = 6 DOF subset of 12-param affine)
    float regParams[12] = {0};
    memcpy(regParams, initParams, 12 * sizeof(float));

    alignTwoVolumesLinearSeveralScales(
        epiInT1, t1Buf, filters,
        t1Dims.W, t1Dims.H, t1Dims.D,
        7, numIterations, coarsestScale,
        regParams, verbose);

    // Read back result
    EPIT1Result result;
    result.aligned.resize(t1Vol);
    memcpy(result.aligned.data(), [epiInT1 contents], t1Vol * sizeof(float));
    result.interpolated = std::move(interpResult);

    // Extract 6 rigid body parameters (3 translation + 3 rotation)
    for (int i = 0; i < 6; i++) result.params[i] = regParams[i];

    return result;
}

T1MNIResult registerT1MNI(
    const float* t1Data, VolumeDims t1Dims, VoxelSize t1Vox,
    const float* mniData, VolumeDims mniDims, VoxelSize mniVox,
    const float* mniBrainData,
    const float* mniMaskData,
    const QuadratureFilters& filters,
    int linearIterations, int nonlinearIterations, int coarsestScale,
    int mmZCut, bool verbose)
{
    auto& c = ctx();

    if (verbose) printf("registerT1MNI: T1 %dx%dx%d -> MNI %dx%dx%d\n",
                        t1Dims.W, t1Dims.H, t1Dims.D,
                        mniDims.W, mniDims.H, mniDims.D);

    int mniVol = mniDims.size();

    // Upload volumes
    id<MTLBuffer> mniBuf = c.newBuffer(mniData, mniVol * sizeof(float));
    id<MTLBuffer> mniBrainBuf = c.newBuffer(mniBrainData, mniVol * sizeof(float));
    id<MTLBuffer> mniMaskBuf = c.newBuffer(mniMaskData, mniVol * sizeof(float));
    id<MTLBuffer> t1Buf = c.newBuffer(t1Data, t1Dims.size() * sizeof(float));

    // Resample T1 to MNI resolution and size
    id<MTLBuffer> t1InMNI = changeVolumesResolutionAndSize(
        t1Buf, t1Dims.W, t1Dims.H, t1Dims.D, t1Vox,
        mniDims.W, mniDims.H, mniDims.D, mniVox, mmZCut);

    // Save interpolated volume
    std::vector<float> interpResult(mniVol);
    memcpy(interpResult.data(), [t1InMNI contents], mniVol * sizeof(float));

    // Center-of-mass alignment
    float cx1, cy1, cz1, cx2, cy2, cz2;
    centerOfMass((float*)[mniBuf contents], mniDims.W, mniDims.H, mniDims.D, cx1, cy1, cz1);
    centerOfMass((float*)[t1InMNI contents], mniDims.W, mniDims.H, mniDims.D, cx2, cy2, cz2);

    float initParams[12] = {0};
    initParams[0] = cx1 - cx2;
    initParams[1] = cy1 - cy2;
    initParams[2] = cz1 - cz2;

    interpolateLinear(t1InMNI, t1InMNI, initParams, mniDims.W, mniDims.H, mniDims.D);

    // Linear registration
    float regParams[12] = {0};
    memcpy(regParams, initParams, 12 * sizeof(float));

    if (verbose) printf("Running linear registration (%d iterations)...\n", linearIterations);

    alignTwoVolumesLinearSeveralScales(
        t1InMNI, mniBuf, filters,
        mniDims.W, mniDims.H, mniDims.D,
        7, linearIterations, coarsestScale,
        regParams, verbose);

    // Save linear result
    T1MNIResult result;
    result.alignedLinear.resize(mniVol);
    memcpy(result.alignedLinear.data(), [t1InMNI contents], mniVol * sizeof(float));
    for (int i = 0; i < 12; i++) result.params[i] = regParams[i];

    // Nonlinear registration
    if (nonlinearIterations > 0) {
        if (verbose) printf("Running nonlinear registration (%d iterations)...\n", nonlinearIterations);

        id<MTLBuffer> totalDispX, totalDispY, totalDispZ;

        alignTwoVolumesNonLinearSeveralScales(
            t1InMNI, mniBuf, filters,
            mniDims.W, mniDims.H, mniDims.D,
            nonlinearIterations, coarsestScale,
            totalDispX, totalDispY, totalDispZ, verbose);

        // Save nonlinear result
        result.alignedNonLinear.resize(mniVol);
        memcpy(result.alignedNonLinear.data(), [t1InMNI contents], mniVol * sizeof(float));

        // Combine linear + nonlinear displacement
        addLinearNonLinearDisplacement(totalDispX, totalDispY, totalDispZ,
                                       regParams, mniDims.W, mniDims.H, mniDims.D);

        // Copy displacement fields
        result.dispX.resize(mniVol);
        result.dispY.resize(mniVol);
        result.dispZ.resize(mniVol);
        memcpy(result.dispX.data(), [totalDispX contents], mniVol * sizeof(float));
        memcpy(result.dispY.data(), [totalDispY contents], mniVol * sizeof(float));
        memcpy(result.dispZ.data(), [totalDispZ contents], mniVol * sizeof(float));

        // Skullstrip: multiply by MNI brain mask
        id<MTLBuffer> ssBuf = c.newBuffer([t1InMNI contents], mniVol * sizeof(float));
        multiplyVolumes(ssBuf, mniMaskBuf, mniVol);
        result.skullstripped.resize(mniVol);
        memcpy(result.skullstripped.data(), [ssBuf contents], mniVol * sizeof(float));
    } else {
        result.alignedNonLinear = result.alignedLinear;
        result.skullstripped.resize(mniVol, 0);
    }

    result.interpolated = std::move(interpResult);

    return result;
}

} // namespace metal_reg
