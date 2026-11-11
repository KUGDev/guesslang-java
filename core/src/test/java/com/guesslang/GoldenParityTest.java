package com.guesslang;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Compares the Java pipeline with the reference Keras model on cases exported by
 * {@code guesslang/tools/export_java.py}: vectorized inputs must match exactly (TF-IDF up to float
 * rounding) and logits within a small tolerance.
 */
class GoldenParityTest {

    private static final double WIDE_REL_TOLERANCE = 1e-5;
    private static final double LOGIT_TOLERANCE = 1e-3;

    private static Guesslang guesslang;
    private static JsonNode golden;

    @BeforeAll
    static void setUp() throws IOException {
        guesslang = Guesslang.load();
        try (InputStream in = new GZIPInputStream(GoldenParityTest.class.getResourceAsStream("/golden.json.gz"))) {
            golden = new ObjectMapper().readTree(in);
        }
    }

    @Test
    void labelsMatchExport() {
        List<String> labels = new ArrayList<>();
        golden.get("labels").forEach(n -> labels.add(n.asText()));
        assertEquals(labels, guesslang.extensions());
    }

    @TestFactory
    Stream<DynamicTest> goldenCases() {
        List<DynamicTest> tests = new ArrayList<>();
        for (JsonNode c : golden.get("cases")) {
            tests.add(DynamicTest.dynamicTest(c.get("name").asText(), () -> check(c)));
        }
        return tests.stream();
    }

    private static void check(JsonNode c) {
        String text = c.get("text").asText();
        List<String> tokens = TextVectorizer.tokenize(text);
        TextVectorizer vectorizer = guesslang.vectorizer();

        int[] deep = vectorizer.deepIds(tokens);
        assertArrayEquals(ints(c.get("deep_ids")), deep, "deep ids");

        float[] wide = vectorizer.wideTfidf(tokens);
        int[] expectedIdx = ints(c.get("wide_indices"));
        JsonNode expectedVal = c.get("wide_values");
        int nonZero = 0;
        for (float v : wide) {
            if (v != 0f) {
                nonZero++;
            }
        }
        assertEquals(expectedIdx.length, nonZero, "number of non-zero tf-idf entries");
        for (int i = 0; i < expectedIdx.length; i++) {
            double expected = expectedVal.get(i).asDouble();
            double actual = wide[expectedIdx[i]];
            assertTrue(Math.abs(expected - actual) <= WIDE_REL_TOLERANCE * Math.max(1.0, Math.abs(expected)),
                    "tf-idf[" + expectedIdx[i] + "] expected " + expected + " got " + actual);
        }

        float[] logits = guesslang.model().logits(deep, wide);
        JsonNode expectedLogits = c.get("logits");
        assertEquals(expectedLogits.size(), logits.length);
        int expectedTop = 0;
        int actualTop = 0;
        for (int i = 0; i < logits.length; i++) {
            double expected = expectedLogits.get(i).asDouble();
            assertTrue(Math.abs(expected - logits[i]) <= LOGIT_TOLERANCE,
                    "logit[" + i + "] expected " + expected + " got " + logits[i]);
            if (expected > expectedLogits.get(expectedTop).asDouble()) {
                expectedTop = i;
            }
            if (logits[i] > logits[actualTop]) {
                actualTop = i;
            }
        }
        assertEquals(expectedTop, actualTop, "top-1 label");
    }

    private static int[] ints(JsonNode array) {
        int[] out = new int[array.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = array.get(i).asInt();
        }
        return out;
    }
}
