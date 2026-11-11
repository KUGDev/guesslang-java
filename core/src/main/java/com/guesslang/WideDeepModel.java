package com.guesslang;

import java.util.Map;

/**
 * Forward pass of the Guesslang wide &amp; deep network:
 * <pre>
 * deep: ids[512] -&gt; Embedding(10000x100) -&gt; mean over 512 positions -&gt; Dense 256 relu -&gt; Dense 128 relu
 * wide: tfidf[10000] -&gt; LayerNormalization
 * logits = Dense(concat(wide, deep))
 * </pre>
 * Dropout is a no-op at inference time. Accumulation is done in double precision.
 * Instances are immutable and thread-safe.
 */
final class WideDeepModel {

    private final Tensor embedding;
    private final Tensor dense0Kernel;
    private final float[] dense0Bias;
    private final Tensor dense1Kernel;
    private final float[] dense1Bias;
    private final float[] lnGamma;
    private final float[] lnBeta;
    private final double lnEpsilon;
    private final Tensor logitsKernel;
    private final float[] logitsBias;

    WideDeepModel(Map<String, Tensor> weights, double lnEpsilon) {
        this.embedding = require(weights, "embedding", 2);
        this.dense0Kernel = require(weights, "dense/kernel", 2);
        this.dense0Bias = require(weights, "dense/bias", 1).data;
        this.dense1Kernel = require(weights, "dense_1/kernel", 2);
        this.dense1Bias = require(weights, "dense_1/bias", 1).data;
        this.lnGamma = require(weights, "layer_normalization/gamma", 1).data;
        this.lnBeta = require(weights, "layer_normalization/beta", 1).data;
        this.logitsKernel = require(weights, "logits/kernel", 2);
        this.logitsBias = require(weights, "logits/bias", 1).data;
        this.lnEpsilon = lnEpsilon;

        check(dense0Kernel.dims[0] == embedding.dims[1], "dense input != embedding size");
        check(dense1Kernel.dims[0] == dense0Kernel.dims[1], "dense_1 input != dense output");
        check(logitsKernel.dims[0] == lnGamma.length + dense1Kernel.dims[1], "logits input != wide + deep size");
        check(lnBeta.length == lnGamma.length, "layer normalization gamma/beta size mismatch");
        check(dense0Bias.length == dense0Kernel.dims[1] && dense1Bias.length == dense1Kernel.dims[1]
                && logitsBias.length == logitsKernel.dims[1], "bias size mismatch");
    }

    private static Tensor require(Map<String, Tensor> weights, String name, int rank) {
        Tensor t = weights.get(name);
        if (t == null) {
            throw new IllegalArgumentException("Missing tensor: " + name);
        }
        if (t.dims.length != rank) {
            throw new IllegalArgumentException("Tensor " + name + " has rank " + t.dims.length + ", expected " + rank);
        }
        return t;
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException("Inconsistent weights: " + message);
        }
    }

    int vocabularySize() {
        return embedding.dims[0];
    }

    int wideSize() {
        return lnGamma.length;
    }

    int outputSize() {
        return logitsBias.length;
    }

    float[] logits(int[] deepIds, float[] wide) {
        if (wide.length != lnGamma.length) {
            throw new IllegalArgumentException("wide input size " + wide.length + " != " + lnGamma.length);
        }
        double[] deep = deep(deepIds);
        double[] wideNorm = layerNorm(wide);

        int outputs = logitsBias.length;
        double[] acc = new double[outputs];
        float[] k = logitsKernel.data;
        for (int i = 0; i < wideNorm.length; i++) {
            axpy(acc, wideNorm[i], k, i * outputs);
        }
        for (int i = 0; i < deep.length; i++) {
            axpy(acc, deep[i], k, (wideNorm.length + i) * outputs);
        }
        float[] result = new float[outputs];
        for (int o = 0; o < outputs; o++) {
            result[o] = (float) (acc[o] + logitsBias[o]);
        }
        return result;
    }

    private double[] deep(int[] ids) {
        int dim = embedding.dims[1];
        int vocab = embedding.dims[0];
        double[] pooled = new double[dim];
        float[] e = embedding.data;
        for (int id : ids) {
            if (id < 0 || id >= vocab) {
                throw new IllegalArgumentException("token id out of range: " + id);
            }
            int base = id * dim;
            for (int d = 0; d < dim; d++) {
                pooled[d] += e[base + d];
            }
        }
        for (int d = 0; d < dim; d++) {
            pooled[d] /= ids.length;
        }
        double[] hidden = denseRelu(pooled, dense0Kernel, dense0Bias);
        return denseRelu(hidden, dense1Kernel, dense1Bias);
    }

    private static double[] denseRelu(double[] in, Tensor kernel, float[] bias) {
        int outputs = kernel.dims[1];
        double[] acc = new double[outputs];
        for (int i = 0; i < in.length; i++) {
            axpy(acc, in[i], kernel.data, i * outputs);
        }
        for (int o = 0; o < outputs; o++) {
            acc[o] = Math.max(0.0, acc[o] + bias[o]);
        }
        return acc;
    }

    private double[] layerNorm(float[] x) {
        int n = x.length;
        double mean = 0;
        for (float v : x) {
            mean += v;
        }
        mean /= n;
        double var = 0;
        for (float v : x) {
            double d = v - mean;
            var += d * d;
        }
        var /= n;
        double inv = 1.0 / Math.sqrt(var + lnEpsilon);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = (x[i] - mean) * inv * lnGamma[i] + lnBeta[i];
        }
        return out;
    }

    private static void axpy(double[] acc, double a, float[] row, int offset) {
        if (a == 0.0) {
            return;
        }
        for (int o = 0; o < acc.length; o++) {
            acc[o] += a * row[offset + o];
        }
    }
}
