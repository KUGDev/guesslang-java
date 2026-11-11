package com.guesslang;

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;

/** Row-major float32 tensor, and the reader for the {@code weights.bin} format written by export_java.py. */
final class Tensor {

    private static final byte[] MAGIC = {'G', 'L', 'W', '1'};
    private static final int MAX_RANK = 4;

    final int[] dims;
    final float[] data;

    Tensor(int[] dims, float[] data) {
        this.dims = dims;
        this.data = data;
    }

    /**
     * Format (big-endian): magic "GLW1", int32 count, then per tensor:
     * uint16 name length + UTF-8 name, int32 rank, int32 dims[rank], float32 data[product(dims)].
     */
    static Map<String, Tensor> readAll(InputStream input) throws IOException {
        DataInputStream in = new DataInputStream(input);
        byte[] magic = new byte[4];
        in.readFully(magic);
        if (!java.util.Arrays.equals(magic, MAGIC)) {
            throw new IOException("Not a guesslang weights file (bad magic)");
        }
        int count = in.readInt();
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        for (int t = 0; t < count; t++) {
            byte[] nameBytes = new byte[in.readUnsignedShort()];
            in.readFully(nameBytes);
            String name = new String(nameBytes, StandardCharsets.UTF_8);
            int rank = in.readInt();
            if (rank < 1 || rank > MAX_RANK) {
                throw new IOException("Bad rank " + rank + " for tensor " + name);
            }
            int[] dims = new int[rank];
            long size = 1;
            for (int i = 0; i < rank; i++) {
                dims[i] = in.readInt();
                if (dims[i] <= 0) {
                    throw new IOException("Bad dimension " + dims[i] + " for tensor " + name);
                }
                size *= dims[i];
            }
            if (size > Integer.MAX_VALUE / 4) {
                throw new IOException("Tensor too large: " + name);
            }
            byte[] raw = new byte[(int) size * 4];
            try {
                in.readFully(raw);
            } catch (EOFException e) {
                throw new IOException("Truncated weights file at tensor " + name, e);
            }
            float[] data = new float[(int) size];
            ByteBuffer.wrap(raw).order(ByteOrder.BIG_ENDIAN).asFloatBuffer().get(data);
            tensors.put(name, new Tensor(dims, data));
        }
        return tensors;
    }
}
