/**
 * Entropy Coding for Synapse Wire Protocol
 *
 * After int8 quantization + delta encoding, activation tensors have lots of
 * near-zero values. Entropy coding compresses these efficiently.
 *
 * Strategy: Run-Length Encoding (RLE) of zero runs + raw values.
 * Simple, fast, no dependencies. Works great when sparsity > 30%.
 *
 * Wire format:
 *   [uint16 numRuns] [runs...] [float32 scale] [uint16 numRows] [float32 scales...]
 *   Each run: [uint8 type][uint16 length][data...]
 *     type 0: zero run (length zeros, no data)
 *     type 1: literal run (length values follow as int8)
 *
 * Theoretical: Shannon entropy of delta activations ≈ 100-200 bytes per token.
 * Current: 796 bytes (int8). Target: 200-400 bytes with RLE.
 */

/**
 * Compress int8 data using run-length encoding of zero runs.
 * @param {Int8Array} data
 * @param {number} zeroThreshold - values with |v| <= threshold treated as zero
 * @returns {{ compressed: Uint8Array, originalLength: number, ratio: number }}
 */
export function rleCompress(data, zeroThreshold = 0) {
  // Minimum zero run length to justify its own run (3-byte header overhead per run,
  // plus splitting a literal creates another 3-byte header = 6 bytes overhead).
  // Short zero runs get folded into adjacent literal runs as stored zeros.
  const MIN_ZERO_RUN = 4;

  // First pass: collect raw runs
  const rawRuns = [];
  let i = 0;

  while (i < data.length) {
    if (Math.abs(data[i]) <= zeroThreshold) {
      let runLen = 0;
      while (i < data.length && Math.abs(data[i]) <= zeroThreshold && runLen < 65535) {
        runLen++;
        i++;
      }
      rawRuns.push({ type: 0, length: runLen, start: i - runLen });
    } else {
      const start = i;
      let runLen = 0;
      while (i < data.length && Math.abs(data[i]) > zeroThreshold && runLen < 65535) {
        runLen++;
        i++;
      }
      rawRuns.push({ type: 1, length: runLen, start, data: data.slice(start, start + runLen) });
    }
  }

  // Second pass: coalesce short zero runs into adjacent literals
  const runs = [];
  for (let r = 0; r < rawRuns.length; r++) {
    const run = rawRuns[r];
    if (run.type === 0 && run.length < MIN_ZERO_RUN) {
      // Fold into a literal run — store the zeros as literal data
      const zeroData = new Int8Array(run.length); // all zeros
      if (runs.length > 0 && runs[runs.length - 1].type === 1 &&
          runs[runs.length - 1].length + run.length <= 65535) {
        // Append to previous literal
        const prev = runs[runs.length - 1];
        const merged = new Int8Array(prev.length + run.length);
        merged.set(prev.data);
        merged.set(zeroData, prev.length);
        prev.data = merged;
        prev.length = merged.length;
      } else {
        runs.push({ type: 1, length: run.length, data: zeroData });
      }
    } else {
      // Merge consecutive literals (can happen after folding)
      if (run.type === 1 && runs.length > 0 && runs[runs.length - 1].type === 1 &&
          runs[runs.length - 1].length + run.length <= 65535) {
        const prev = runs[runs.length - 1];
        const merged = new Int8Array(prev.length + run.length);
        merged.set(prev.data);
        merged.set(run.data, prev.length);
        prev.data = merged;
        prev.length = merged.length;
      } else {
        runs.push(run);
      }
    }
  }

  // Serialize: [uint16 numRuns] then each run
  let totalBytes = 2; // numRuns
  for (const run of runs) {
    totalBytes += 3; // type(1) + length(2)
    if (run.type === 1) totalBytes += run.length; // literal data
  }

  const compressed = new Uint8Array(totalBytes);
  const view = new DataView(compressed.buffer);
  view.setUint16(0, runs.length, true);

  let offset = 2;
  for (const run of runs) {
    compressed[offset] = run.type;
    view.setUint16(offset + 1, run.length, true);
    offset += 3;
    if (run.type === 1) {
      for (let j = 0; j < run.length; j++) {
        compressed[offset + j] = run.data[j] & 0xFF;
      }
      offset += run.length;
    }
  }

  return {
    compressed,
    originalLength: data.length,
    ratio: data.length / totalBytes,
  };
}

/**
 * Decompress RLE-encoded int8 data.
 * @param {Uint8Array} compressed
 * @param {number} originalLength
 * @returns {Int8Array}
 */
export function rleDecompress(compressed, originalLength) {
  const view = new DataView(compressed.buffer, compressed.byteOffset, compressed.byteLength);
  const numRuns = view.getUint16(0, true);
  const result = new Int8Array(originalLength);

  let offset = 2;
  let outIdx = 0;

  for (let r = 0; r < numRuns; r++) {
    const type = compressed[offset];
    const length = view.getUint16(offset + 1, true);
    offset += 3;

    if (type === 0) {
      // Zero run — already zero-initialized
      outIdx += length;
    } else {
      // Literal run
      for (let j = 0; j < length; j++) {
        result[outIdx++] = compressed[offset + j] << 24 >> 24; // sign-extend
      }
      offset += length;
    }
  }

  return result;
}

/**
 * Compress a quantized activation for wire transfer.
 * Applies RLE on top of int8 quantized data.
 *
 * @param {Int8Array} int8Data
 * @param {number} zeroThreshold - treat small values as zero (lossy but more compression)
 * @returns {{ compressed: Uint8Array, originalLength: number, ratio: number }}
 */
export function compressActivation(int8Data, zeroThreshold = 0) {
  return rleCompress(int8Data, zeroThreshold);
}

/**
 * Decompress an activation from wire format.
 */
export function decompressActivation(compressed, originalLength) {
  return rleDecompress(compressed, originalLength);
}

// ─── Wire-Level Payload Compression ──────────────────────────────
// Wraps RLE for use on packed quantized payloads (int8 data + scale bytes).
// Format: [uint32 originalLength] [rle compressed data...]

/**
 * Compress a packed quantized payload for wire transfer.
 * Returns null if compression doesn't save space (caller should send uncompressed).
 *
 * @param {ArrayBuffer} packedPayload - Output of packQuantized/packQuantizedPerChannel
 * @param {number} zeroThreshold - Values with |v| <= threshold treated as zero (0 = lossless)
 * @returns {{ compressed: ArrayBuffer, ratio: number } | null}
 */
export function compressPayload(packedPayload, zeroThreshold = 0) {
  const raw = new Int8Array(packedPayload);
  const { compressed, ratio } = rleCompress(raw, zeroThreshold);

  // Only compress if we actually save space (accounting for the 4-byte length prefix)
  if (compressed.byteLength + 4 >= raw.byteLength) {
    return null;
  }

  // Wrap: [uint32 originalLength] [compressed bytes...]
  const wrapped = new ArrayBuffer(4 + compressed.byteLength);
  new DataView(wrapped).setUint32(0, raw.byteLength, true);
  new Uint8Array(wrapped, 4).set(compressed);

  return { compressed: wrapped, ratio };
}

/**
 * Decompress a wire-compressed payload back to the original packed format.
 *
 * @param {Uint8Array} wrappedPayload - Compressed payload from the wire (after header strip)
 * @returns {Uint8Array} - Original packed quantized payload
 */
export function decompressPayload(wrappedPayload) {
  const buf = wrappedPayload.buffer.slice
    ? wrappedPayload.buffer.slice(wrappedPayload.byteOffset, wrappedPayload.byteOffset + wrappedPayload.byteLength)
    : wrappedPayload.buffer;
  const view = new DataView(buf);
  const originalLength = view.getUint32(0, true);
  const compressedBytes = new Uint8Array(buf, 4);
  const decompressed = rleDecompress(compressedBytes, originalLength);
  return new Uint8Array(decompressed.buffer);
}
