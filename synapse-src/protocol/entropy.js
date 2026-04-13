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
  const runs = [];
  let i = 0;

  while (i < data.length) {
    if (Math.abs(data[i]) <= zeroThreshold) {
      // Count zero run
      let runLen = 0;
      while (i < data.length && Math.abs(data[i]) <= zeroThreshold && runLen < 65535) {
        runLen++;
        i++;
      }
      runs.push({ type: 0, length: runLen });
    } else {
      // Count literal run
      const start = i;
      let runLen = 0;
      while (i < data.length && Math.abs(data[i]) > zeroThreshold && runLen < 65535) {
        runLen++;
        i++;
      }
      runs.push({ type: 1, length: runLen, data: data.slice(start, start + runLen) });
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
