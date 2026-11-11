package com.guesslang;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Re-implementation of the two Keras {@code TextVectorization} layers used by the model
 * ({@code standardize="lower_and_strip_punctuation"}, {@code split="whitespace"}).
 *
 * <p>Keras semantics that must be reproduced exactly (verified against TensorFlow):
 * <ul>
 *   <li>{@code tf.strings.lower} lowercases ASCII letters only;</li>
 *   <li>the punctuation regex removes ASCII punctuation only (all 32 printable symbols);</li>
 *   <li>{@code tf.strings.split} splits on ASCII whitespace only: space, \t, \n, \r, \u000B, \f.</li>
 * </ul>
 * Every special character is ASCII, so working on UTF-16 chars is equivalent to TF working on UTF-8 bytes.
 */
final class TextVectorizer {

    static final int DEEP_PADDING_ID = 0;
    static final int DEEP_OOV_ID = 1;
    static final int WIDE_OOV_ID = 0;

    private final Map<String, Integer> deepIndex;
    private final Map<String, Integer> wideIndex;
    private final float[] idf;
    private final int sequenceLength;
    private final int ngrams;

    TextVectorizer(List<String> deepVocab, List<String> wideVocab, float[] idf, int sequenceLength, int ngrams) {
        if (wideVocab.size() != idf.length) {
            throw new IllegalArgumentException("wide vocabulary size " + wideVocab.size() + " != idf size " + idf.length);
        }
        if (ngrams < 1) {
            throw new IllegalArgumentException("ngrams must be >= 1");
        }
        // Special tokens ("" and "[UNK]") can never be produced by the tokenizer, skip them anyway.
        this.deepIndex = index(deepVocab, 2);
        this.wideIndex = index(wideVocab, 1);
        this.idf = idf;
        this.sequenceLength = sequenceLength;
        this.ngrams = ngrams;
    }

    private static Map<String, Integer> index(List<String> vocab, int firstRegular) {
        Map<String, Integer> map = new HashMap<>(vocab.size() * 2);
        for (int i = firstRegular; i < vocab.size(); i++) {
            map.put(vocab.get(i), i);
        }
        return map;
    }

    int sequenceLength() {
        return sequenceLength;
    }

    int wideSize() {
        return idf.length;
    }

    /** Lowercase (ASCII), strip ASCII punctuation, split on ASCII whitespace. */
    static List<String> tokenize(CharSequence text) {
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (int i = 0, n = text.length(); i < n; i++) {
            char c = text.charAt(i);
            if (isAsciiWhitespace(c)) {
                if (current.length() > 0) {
                    tokens.add(current.toString());
                    current.setLength(0);
                }
            } else if (isAsciiPunctuation(c)) {
                // removed before splitting, so "a.b" becomes "ab"
                continue;
            } else if (c >= 'A' && c <= 'Z') {
                current.append((char) (c + ('a' - 'A')));
            } else {
                current.append(c);
            }
        }
        if (current.length() > 0) {
            tokens.add(current.toString());
        }
        return tokens;
    }

    private static boolean isAsciiWhitespace(char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\u000B' || c == '\f';
    }

    private static boolean isAsciiPunctuation(char c) {
        return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
    }

    /** Deep branch input: first {@code sequenceLength} token ids, OOV = 1, zero padded. */
    int[] deepIds(List<String> tokens) {
        int[] ids = new int[sequenceLength];
        int n = Math.min(tokens.size(), sequenceLength);
        for (int i = 0; i < n; i++) {
            ids[i] = deepIndex.getOrDefault(tokens.get(i), DEEP_OOV_ID);
        }
        return ids;
    }

    /** Wide branch input: dense TF-IDF over 1..ngrams-grams, OOV n-grams counted in index 0. */
    float[] wideTfidf(List<String> tokens) {
        int[] counts = new int[idf.length];
        StringBuilder sb = new StringBuilder();
        for (int width = 1; width <= ngrams; width++) {
            for (int start = 0; start + width <= tokens.size(); start++) {
                String gram;
                if (width == 1) {
                    gram = tokens.get(start);
                } else {
                    sb.setLength(0);
                    for (int j = 0; j < width; j++) {
                        if (j > 0) {
                            sb.append(' ');
                        }
                        sb.append(tokens.get(start + j));
                    }
                    gram = sb.toString();
                }
                counts[wideIndex.getOrDefault(gram, WIDE_OOV_ID)]++;
            }
        }
        float[] out = new float[idf.length];
        for (int i = 0; i < counts.length; i++) {
            if (counts[i] != 0) {
                out[i] = counts[i] * idf[i];
            }
        }
        return out;
    }
}
