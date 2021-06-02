/* cbf.h - public domain library for building binary trees in parallel
by Jonathan Dupuy

   Do this:
      #define CBF_IMPLEMENTATION
   before you include this file in *one* C or C++ file to create the implementation.

   // i.e. it should look like this:
   #include ...
   #include ...
   #include ...
   #define CBF_IMPLEMENTATION
   #include "cbf.h"

   INTERFACING
   define CBF_ASSERT(x) to avoid using assert.h
   define CBF_MALLOC(x) to use your own memory allocator
   define CBF_FREE(x) to use your own memory deallocator
   define CBF_MEMCPY(dst, src, num) to use your own memcpy routine
   define CBF_MEMSET(ptr, value, num) to use your own memset routine
*/

#ifndef CBF_INCLUDE_CBF_H
#define CBF_INCLUDE_CBF_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef CBF_STATIC
#define CBFDEF static
#else
#define CBFDEF extern
#endif

#include <stdint.h>

// data-structure
typedef struct cbf_BitField cbf_BitField;

// create / destroy bitfield
CBFDEF cbf_BitField *cbf_Create(int64_t size);
CBFDEF void cbf_Release(cbf_BitField *tree);

// O(1) queries
CBFDEF int64_t cbf_Size(const cbf_BitField *bf);
CBFDEF int64_t cbf_BitCount(const cbf_BitField *bf);

// O(Lg(size)) queries
CBFDEF int64_t cbf_DecodeBit(const cbf_BitField *cbf, int64_t handle);
CBFDEF int64_t cbf_EncodeBit(const cbf_BitField *cbf, int64_t bitID);

// manipulation
CBFDEF void cbf_Clear(cbf_BitField *cbf);
CBFDEF uint64_t cbf_GetBit(const cbf_BitField *cbf, int64_t bitID);
CBFDEF void cbf_SetBit(cbf_BitField *cbf, int64_t bitID, uint64_t bitValue);
CBFDEF void cbf_Reduce(cbf_BitField *cbf);
typedef void (*cbf_UpdateCallback)(cbf_BitField *cbf,
                                   const int64_t bitID,
                                   const void *userData);
CBFDEF void cbf_Update(cbf_BitField *cbf,
                       cbf_UpdateCallback updater,
                       const void *userData);

// serialization
CBFDEF int64_t cbf_HeapMaxDepth(const cbf_BitField *cbf);
CBFDEF int64_t cbf_HeapByteSize(const cbf_BitField *cbf);
CBFDEF const char *cbf_GetHeap(const cbf_BitField *cbf);
CBFDEF void cbf_SetHeap(cbf_BitField *tree, const char *heapToCopy);

#ifdef __cplusplus
} // extern "C"
#endif

//
//
//// end header file ///////////////////////////////////////////////////////////
#endif // CBF_INCLUDE_CBF_H

#ifdef CBF_IMPLEMENTATION

#include <stdbool.h>

#ifndef CBF_ASSERT
#    include <assert.h>
#    define CBF_ASSERT(x) assert(x)
#endif

#ifndef CBF_MALLOC
#    include <stdlib.h>
#    define CBF_MALLOC(x) (malloc(x))
#    define CBF_FREE(x) (free(x))
#else
#    ifndef CBF_FREE
#        error CBF_MALLOC defined without CBF_FREE
#    endif
#endif

#ifndef CBF_MEMCPY
#    include <string.h>
#    define CBF_MEMCPY(dst, src, num) memcpy(dst, src, num)
#endif

#ifndef CBF_MEMSET
#    include <string.h>
#    define CBF_MEMSET(ptr, value, num) memset(ptr, value, num)
#endif

#ifndef _OPENMP
#   define CBF_ATOMIC
#   define CBF_PARALLEL_FOR
#   define CBF_BARRIER
#else
#   if defined(_WIN32)
#       define CBF_ATOMIC          __pragma("omp atomic" )
#       define CBF_PARALLEL_FOR    __pragma("omp parallel for")
#       define CBF_BARRIER         __pragma("omp barrier")
#   else
#       define CBF_ATOMIC          _Pragma("omp atomic" )
#       define CBF_PARALLEL_FOR    _Pragma("omp parallel for")
#       define CBF_BARRIER         _Pragma("omp barrier")
#   endif
#endif


/*******************************************************************************
 * NextPowerOfTwo -- Returns the upper power of two value
 *
 * if the input is already a power of two, its value is returned.
 *
 */
static int64_t cbf__NextPowerOfTwo(int64_t x)
{
    x--;
    x|= x >> 1;
    x|= x >> 2;
    x|= x >> 4;
    x|= x >> 8;
    x|= x >> 16;
    x|= x >> 32;
    x++;

    return x;
}


/*******************************************************************************
 * FindLSB -- Returns the position of the least significant bit
 *
 */
static inline int64_t cbf__FindLSB(uint64_t x)
{
    int64_t lsb = 0;

    while (((x >> lsb) & 1u) == 0u) {
        ++lsb;
    }

    return lsb;
}


/*******************************************************************************
 * FindMSB -- Returns the position of the most significant bit
 *
 */
static inline int64_t cbf__FindMSB(uint64_t x)
{
    int64_t msb = 0;

    while (x > 1u) {
        ++msb;
        x = x >> 1;
    }

    return msb;
}


/*******************************************************************************
 * MinValue -- Returns the minimum value between two inputs
 *
 */
static inline uint64_t cbf__MinValue(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}


/*******************************************************************************
 * SetBitValue -- Sets the value of a bit stored in a bitfield
 *
 */
static void
cbf__SetBitValue(uint64_t *bitField, int64_t bitID, uint64_t bitValue)
{
    const uint64_t bitMask = ~(1ULL << bitID);

CBF_ATOMIC
    (*bitField)&= bitMask;
CBF_ATOMIC
    (*bitField)|= (bitValue << bitID);
}


/*******************************************************************************
 * GetBitValue -- Sets the value of a bit stored in a bitfield
 *
 */
static uint64_t cbf__GetBitValue(const uint64_t *bitField, int64_t bitID)
{
    return ((*bitField) >> bitID) & 1ULL;
}


/*******************************************************************************
 * BitfieldInsert -- Inserts data in range [offset, offset + count - 1]
 *
 */
static inline void
cbf__BitFieldInsert(
    uint64_t *bitField,
    int64_t  bitOffset,
    int64_t  bitCount,
    uint64_t bitData
) {
    CBF_ASSERT(bitOffset < 64 && bitCount <= 64 && bitOffset + bitCount <= 64);
    uint64_t bitMask = ~(~(0xFFFFFFFFFFFFFFFFULL << bitCount) << bitOffset);
CBF_ATOMIC
    (*bitField)&= bitMask;
CBF_ATOMIC
    (*bitField)|= (bitData << bitOffset);
}


/*******************************************************************************
 * BitFieldExtract -- Extracts bits [bitOffset, bitOffset + bitCount - 1] from
 * a bitfield, returning them in the least significant bits of the result.
 *
 */
static inline uint64_t
cbf__BitFieldExtract(
    const uint64_t bitField,
    int64_t bitOffset,
    int64_t bitCount
) {
    CBF_ASSERT(bitOffset < 64 && bitCount < 64 && bitOffset + bitCount <= 64);
    uint64_t bitMask = ~(0xFFFFFFFFFFFFFFFFULL << bitCount);

    return (bitField >> bitOffset) & bitMask;
}


/*******************************************************************************
 * Parallel Binary Tree Data-Structure
 *
 */
struct cbf_BitField {
    uint64_t *heap;
};


/*******************************************************************************
 * Node Data-Structure
 *
 * This data-structure is used internally to lookup data in the heap.
 *
 */
typedef struct {
    uint64_t id   : 58; // heapID
    uint64_t depth:  6; // log2(heapID)
} cbf__Node;


/*******************************************************************************
 * CreateNode -- Constructor for the Node data structure
 *
 */
static cbf__Node cbf__CreateNode(uint64_t id, int64_t depth)
{
    cbf__Node node;

    node.id = id;
    node.depth = depth;

    return node;
}


/*******************************************************************************
 * ParentNode -- Computes the parent of the input node
 *
 */
static cbf__Node cbf__ParentNode(const cbf__Node node)
{
    return cbf__CreateNode(node.id >> 1, node.depth - 1);
}


/*******************************************************************************
 * LeftSiblingNode -- Computes the left sibling of the input node
 *
 */
static cbf__Node cbf__LeftSiblingNode(const cbf__Node node)
{
    return cbf__CreateNode(node.id & (~1u), node.depth);
}


/*******************************************************************************
 * HeapByteSize -- Computes the number of Bytes to allocate for the bitfield
 *
 * For a tree of max depth D, the number of Bytes is 2^(D-1).
 * Note that 2 bits are "wasted" in the sense that they only serve
 * to round the required number of bytes to a power of two.
 *
 */
static int64_t cbf__HeapByteSize(uint64_t heapMaxDepth)
{
    return 1LL << (heapMaxDepth - 1);
}


/*******************************************************************************
 * HeapUint64Size -- Computes the number of uints to allocate for the bitfield
 *
 */
static inline int64_t cbf__HeapUint64Size(int64_t treeMaxDepth)
{
    return cbf__HeapByteSize(treeMaxDepth) >> 3;
}


/*******************************************************************************
 * BitFieldUint64Index -- Computes the index for accessing the bitfield in the heap
 *
 */
static int64_t cbf__BitFieldUint64Index(const cbf_BitField *cbf)
{
    return (3LL << (cbf_HeapMaxDepth(cbf) - 6));
}


/*******************************************************************************
 * NodeBitID -- Returns the bit index that stores data associated with a given node
 *
 * For a tree of max depth D and given an index in [0, 2^(D+1) - 1], this
 * functions is used to emulate the behaviour of a lookup in an array, i.e.,
 * uint[nodeID]. It provides the first bit in memory that stores
 * information associated with the element of index nodeID.
 *
 * For data located at level d, the bit offset is 2^d x (3 - d + D)
 * We then offset this quantity by the index by (nodeID - 2^d) x (D + 1 - d)
 * Note that the null index (nodeID = 0) is also supported.
 *
 */
static inline int64_t cbf__NodeBitID(const cbf_BitField *tree, const cbf__Node node)
{
    int64_t tmp1 = 2LL << node.depth;
    int64_t tmp2 = 1LL + cbf_HeapMaxDepth(tree) - node.depth;

    return tmp1 + node.id * tmp2;
}


/*******************************************************************************
 * NodeBitSize -- Returns the number of bits storing the input node value
 *
 */
static inline int64_t
cbf__NodeBitSize(const cbf_BitField *tree, const cbf__Node node)
{
    return cbf_HeapMaxDepth(tree) - node.depth + 1;
}


/*******************************************************************************
 * HeapArgs
 *
 * The CBF heap data structure uses an array of 64-bit words to store its data.
 * Whenever we need to access a certain bit range, we need to query two such
 * words (because sometimes the requested bit range overlaps two 64-bit words).
 * The HeapArg data structure provides arguments for reading from and/or
 * writing to the two 64-bit words that bound the queries range.
 *
 */
typedef struct {
    uint64_t *bitFieldLSB, *bitFieldMSB;
    int64_t bitOffsetLSB;
    int64_t bitCountLSB, bitCountMSB;
} cbf__HeapArgs;

cbf__HeapArgs
cbf__CreateHeapArgs(const cbf_BitField *tree, const cbf__Node node, int64_t bitCount)
{
    int64_t alignedBitOffset = cbf__NodeBitID(tree, node);
    int64_t maxBufferIndex = cbf__HeapUint64Size(cbf_HeapMaxDepth(tree)) - 1;
    int64_t bufferIndexLSB = (alignedBitOffset >> 6);
    int64_t bufferIndexMSB = cbf__MinValue(bufferIndexLSB + 1, maxBufferIndex);
    cbf__HeapArgs args;

    args.bitOffsetLSB = alignedBitOffset & 63;
    args.bitCountLSB = cbf__MinValue(64 - args.bitOffsetLSB, bitCount);
    args.bitCountMSB = bitCount - args.bitCountLSB;
    args.bitFieldLSB = &tree->heap[bufferIndexLSB];
    args.bitFieldMSB = &tree->heap[bufferIndexMSB];

    return args;
}


/*******************************************************************************
 * HeapWrite -- Sets bitCount bits located at nodeID to bitData
 *
 * Note that this procedure writes to at most two uint64 elements.
 * Two elements are relevant whenever the specified interval overflows 64-bit
 * words.
 *
 */
static void
cbf__HeapWriteExplicit(
    cbf_BitField *tree,
    const cbf__Node node,
    int64_t bitCount,
    uint64_t bitData
) {
    cbf__HeapArgs args = cbf__CreateHeapArgs(tree, node, bitCount);

    cbf__BitFieldInsert(args.bitFieldLSB,
                        args.bitOffsetLSB,
                        args.bitCountLSB,
                        bitData);
    cbf__BitFieldInsert(args.bitFieldMSB,
                        0u,
                        args.bitCountMSB,
                        bitData >> args.bitCountLSB);
}

static void
cbf__HeapWrite(cbf_BitField *tree, const cbf__Node node, uint64_t bitData)
{
    cbf__HeapWriteExplicit(tree, node, cbf__NodeBitSize(tree, node), bitData);
}


/*******************************************************************************
 * HeapRead -- Returns bitCount bits located at nodeID
 *
 * Note that this procedure reads from two uint64 elements.
 * This is because the data is not necessarily aligned with 64-bit
 * words.
 *
 */
static uint64_t
cbf__HeapReadExplicit(
    const cbf_BitField *tree,
    const cbf__Node node,
    int64_t bitCount
) {
    cbf__HeapArgs args = cbf__CreateHeapArgs(tree, node, bitCount);
    uint64_t lsb = cbf__BitFieldExtract(*args.bitFieldLSB,
                                        args.bitOffsetLSB,
                                        args.bitCountLSB);
    uint64_t msb = cbf__BitFieldExtract(*args.bitFieldMSB,
                                        0u,
                                        args.bitCountMSB);

    return (lsb | (msb << args.bitCountLSB));
}

CBFDEF uint64_t cbf__HeapRead(const cbf_BitField *tree, const cbf__Node node)
{
    return cbf__HeapReadExplicit(tree, node, cbf__NodeBitSize(tree, node));
}


/*******************************************************************************
 * SetBit -- Set a specific bit to either 0 or 1 in the bitfield
 *
 */
CBFDEF void cbf_SetBit(cbf_BitField *cbf, int64_t bitID, uint64_t bitValue)
{
    uint64_t *bitField = &cbf->heap[cbf__BitFieldUint64Index(cbf)];

    cbf__SetBitValue(&bitField[bitID >> 6], bitID & 63, bitValue);
}


/*******************************************************************************
 * GetBit -- Returns a specific bit value in the bitfield
 *
 */
CBFDEF uint64_t cbf_GetBit(const cbf_BitField *cbf, int64_t bitID)
{
    const uint64_t *bitField = &cbf->heap[cbf__BitFieldUint64Index(cbf)];

    return cbf__GetBitValue(&bitField[bitID >> 6], bitID & 63);
}


/*******************************************************************************
 * Clear -- Clears the bitfield
 *
 */
CBFDEF void cbf_Clear(cbf_BitField *cbf)
{
    int64_t heapDepth = cbf_HeapMaxDepth(cbf);

    CBF_MEMSET(cbf->heap, 0, cbf_HeapByteSize(cbf));
    cbf->heap[0] = 1ULL << heapDepth;
}


/*******************************************************************************
 * GetHeap -- Returns a read-only pointer to the heap memory
 *
 */
CBFDEF const char *cbf_GetHeap(const cbf_BitField *tree)
{
    return (const char *)tree->heap;
}


/*******************************************************************************
 * SetHeap -- Sets the heap memory from a read-only buffer
 *
 */
CBFDEF void cbf_SetHeap(cbf_BitField *tree, const char *buffer)
{
    CBF_MEMCPY(tree->heap, buffer, cbf_HeapByteSize(tree));
}


/*******************************************************************************
 * HeapByteSize -- Returns the amount of bytes consumed by the CBF heap
 *
 */
CBFDEF int64_t cbf_HeapByteSize(const cbf_BitField *tree)
{
    return cbf__HeapByteSize(cbf_HeapMaxDepth(tree));
}


/*******************************************************************************
 * Reduce -- Sums the 2 elements below the current slot
 *
 */
CBFDEF void cbf_Reduce(cbf_BitField *tree)
{
    int64_t depth = cbf_HeapMaxDepth(tree);
    uint64_t minNodeID = (1ULL << depth);
    uint64_t maxNodeID = (2ULL << depth);

    // prepass: processes deepest levels in parallel
CBF_PARALLEL_FOR
    for (uint64_t nodeID = minNodeID; nodeID < maxNodeID; nodeID+= 64u) {
        cbf__Node heapNode = cbf__CreateNode(nodeID, depth);
        int64_t alignedBitOffset = cbf__NodeBitID(tree, heapNode);
        uint64_t bitField = tree->heap[alignedBitOffset >> 6];
        uint64_t bitData = 0u;

        // 2-bits
        bitField = (bitField & 0x5555555555555555ULL)
                 + ((bitField >>  1) & 0x5555555555555555ULL);
        bitData = bitField;
        tree->heap[(alignedBitOffset - minNodeID) >> 6] = bitData;

        // 3-bits
        bitField = (bitField & 0x3333333333333333ULL)
                 + ((bitField >>  2) & 0x3333333333333333ULL);
        bitData = ((bitField >>  0) & (7ULL <<  0))
                | ((bitField >>  1) & (7ULL <<  3))
                | ((bitField >>  2) & (7ULL <<  6))
                | ((bitField >>  3) & (7ULL <<  9))
                | ((bitField >>  4) & (7ULL << 12))
                | ((bitField >>  5) & (7ULL << 15))
                | ((bitField >>  6) & (7ULL << 18))
                | ((bitField >>  7) & (7ULL << 21))
                | ((bitField >>  8) & (7ULL << 24))
                | ((bitField >>  9) & (7ULL << 27))
                | ((bitField >> 10) & (7ULL << 30))
                | ((bitField >> 11) & (7ULL << 33))
                | ((bitField >> 12) & (7ULL << 36))
                | ((bitField >> 13) & (7ULL << 39))
                | ((bitField >> 14) & (7ULL << 42))
                | ((bitField >> 15) & (7ULL << 45));
        cbf__HeapWriteExplicit(tree, cbf__CreateNode(nodeID >> 2, depth - 2), 48ULL, bitData);

        // 4-bits
        bitField = (bitField & 0x0F0F0F0F0F0F0F0FULL)
                 + ((bitField >>  4) & 0x0F0F0F0F0F0F0F0FULL);
        bitData = ((bitField >>  0) & (15ULL <<  0))
                | ((bitField >>  4) & (15ULL <<  4))
                | ((bitField >>  8) & (15ULL <<  8))
                | ((bitField >> 12) & (15ULL << 12))
                | ((bitField >> 16) & (15ULL << 16))
                | ((bitField >> 20) & (15ULL << 20))
                | ((bitField >> 24) & (15ULL << 24))
                | ((bitField >> 28) & (15ULL << 28));
        cbf__HeapWriteExplicit(tree, cbf__CreateNode(nodeID >> 3, depth - 3), 32ULL, bitData);

        // 5-bits
        bitField = (bitField & 0x00FF00FF00FF00FFULL)
                 + ((bitField >>  8) & 0x00FF00FF00FF00FFULL);
        bitData = ((bitField >>  0) & (31ULL <<  0))
                | ((bitField >> 11) & (31ULL <<  5))
                | ((bitField >> 22) & (31ULL << 10))
                | ((bitField >> 33) & (31ULL << 15));
        cbf__HeapWriteExplicit(tree, cbf__CreateNode(nodeID >> 4, depth - 4), 20ULL, bitData);

        // 6-bits
        bitField = (bitField & 0x0000FFFF0000FFFFULL)
                 + ((bitField >> 16) & 0x0000FFFF0000FFFFULL);
        bitData = ((bitField >>  0) & (63ULL << 0))
                | ((bitField >> 26) & (63ULL << 6));
        cbf__HeapWriteExplicit(tree, cbf__CreateNode(nodeID >> 5, depth - 5), 12ULL, bitData);

        // 7-bits
        bitField = (bitField & 0x00000000FFFFFFFFULL)
                 + ((bitField >> 32) & 0x00000000FFFFFFFFULL);
        bitData = bitField;
        cbf__HeapWriteExplicit(tree, cbf__CreateNode(nodeID >> 6, depth - 6),  7ULL, bitData);
    }
CBF_BARRIER
    depth-= 6;

    // iterate over elements atomically
    while (--depth >= 0) {
        uint64_t minNodeID = 1ULL << depth;
        uint64_t maxNodeID = 2ULL << depth;

CBF_PARALLEL_FOR
        for (uint64_t j = minNodeID; j < maxNodeID; ++j) {
            uint64_t x0 = cbf__HeapRead(tree, cbf__CreateNode(j << 1    , depth + 1));
            uint64_t x1 = cbf__HeapRead(tree, cbf__CreateNode(j << 1 | 1, depth + 1));

            cbf__HeapWrite(tree, cbf__CreateNode(j, depth), x0 + x1);
        }
CBF_BARRIER
    }
}


/*******************************************************************************
 * Bitfield Ctor
 *
 */
CBFDEF cbf_BitField *cbf_Create(int64_t size)
{
    cbf_BitField *cbf = (cbf_BitField *)CBF_MALLOC(sizeof(*cbf));
    int64_t heapDepth = cbf__FindMSB(cbf__NextPowerOfTwo(size));

    // the bitfield has to be at least 2^6 bits wide
    if (heapDepth < 6) heapDepth = 6;

    cbf->heap = (uint64_t *)CBF_MALLOC(cbf__HeapByteSize(heapDepth));
    cbf->heap[0] = 1ULL << heapDepth;

    cbf_Clear(cbf);

    return cbf;
}


/*******************************************************************************
 * Buffer Dtor
 *
 */
CBFDEF void cbf_Release(cbf_BitField *cbf)
{
    CBF_FREE(cbf->heap);
    CBF_FREE(cbf);
}


/*******************************************************************************
 * Update -- Split or merge each node in parallel
 *
 * The user provides an updater function that is responsible for
 * splitting or merging each node.
 *
 */
CBFDEF void
cbf_Update(cbf_BitField *cbt, cbf_UpdateCallback updater, const void *userData)
{
CBF_PARALLEL_FOR
    for (int64_t handle = 0; handle < cbf_BitCount(cbt); ++handle) {
        updater(cbt, cbf_DecodeBit(cbt, handle), userData);
    }
CBF_BARRIER

    cbf_Reduce(cbt);
}


/*******************************************************************************
 * Capacity -- Returns capacity of the bitfield in base 2 logarithm
 *
 */
CBFDEF int64_t cbf_HeapMaxDepth(const cbf_BitField *cbf)
{
    return cbf__FindLSB(cbf->heap[0]);
}
CBFDEF int64_t cbf_Size(const cbf_BitField *cbf)
{
    return 1LL << cbf_HeapMaxDepth(cbf);
}


/*******************************************************************************
 * BitCount -- Returns the number of bits set to one in the bit field
 *
 */
CBFDEF int64_t cbf_BitCount(const cbf_BitField *cbf)
{
    return cbf__HeapRead(cbf, cbf__CreateNode(1u, 0));
}


/*******************************************************************************
 * DecodeNode -- Returns the leaf node associated to index nodeID
 *
 * This is procedure is for iterating over the one-valued bits.
 *
 */
CBFDEF int64_t cbf_DecodeBit(const cbf_BitField *cbf, int64_t handle)
{
    CBF_ASSERT(handle < cbf_BitCount(cbf) && "handle > NodeCount");
    CBF_ASSERT(handle >= 0 && "handle < 0");

    cbf__Node node = cbf__CreateNode(1u, 0);
    int64_t bitFieldSize = cbf_Size(cbf);

    while (node.id < bitFieldSize) {
        cbf__Node leftChildNode = cbf__CreateNode(node.id<<= 1u, ++node.depth);
        uint64_t heapValue = cbf__HeapRead(cbf, leftChildNode);
        uint64_t b = (uint64_t)handle < heapValue ? 0u : 1u;

        node.id|= b;
        handle-= heapValue * b;
    }

    return (node.id ^ bitFieldSize);
}


/*******************************************************************************
 * EncodeNode -- Returns the handle associated with the corresponding bitID
 *
 * This does the inverse of the DecodeNode routine. Note that this mapping
 * has the property that any bit set to 0 will be mapped to the ID of the next
 * bit set to one in the bit field.
 *
 */
CBFDEF int64_t cbf_EncodeBit(const cbf_BitField *cbf, int64_t bitID)
{
    int64_t bitFieldSize = cbf_Size(cbf);
    cbf__Node node = cbf__CreateNode(bitID + bitFieldSize, cbf_HeapMaxDepth(cbf));
    int64_t handle = 0;

    while (node.id > 1u) {
        cbf__Node sibling = cbf__LeftSiblingNode(node);
        uint64_t bitCount = cbf__HeapRead(cbf, sibling);

        handle+= (node.id & 1u) * bitCount;
        node = cbf__ParentNode(node);
    }

    return handle;
}


#undef CBF_ATOMIC
#undef CBF_PARALLEL_FOR
#undef CBF_BARRIER
#endif

