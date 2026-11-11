package com.guesslang;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * Programming language detection with the Guesslang wide &amp; deep model, in pure Java.
 *
 * <p>Load once (loading reads ~7 MB of weights) and reuse: instances are immutable and thread-safe.
 * <pre>{@code
 * Guesslang guesslang = Guesslang.load();
 * List<Prediction> top = guesslang.predict(text, 3);
 * }</pre>
 */
public final class Guesslang {

    /** Classpath location of the bundled model. */
    public static final String DEFAULT_RESOURCE_DIR = "/com/guesslang/model/";

    private static final String PROPERTIES = "model.properties";
    private static final String LABELS = "labels.tsv";
    private static final String DEEP_VOCAB = "deep_vocab.txt";
    private static final String WIDE_VOCAB = "wide_vocab.txt";
    private static final String WEIGHTS = "weights.bin";
    private static final String SUPPORTED_FORMAT = "1";

    private final TextVectorizer vectorizer;
    private final WideDeepModel model;
    private final List<String> extensions;
    private final List<String> languages;
    private final Properties properties;

    private Guesslang(TextVectorizer vectorizer, WideDeepModel model,
                      List<String> extensions, List<String> languages, Properties properties) {
        this.vectorizer = vectorizer;
        this.model = model;
        this.extensions = extensions;
        this.languages = languages;
        this.properties = properties;
    }

    /** Loads the model bundled in this jar. */
    public static Guesslang load() throws IOException {
        return load(Guesslang.class, DEFAULT_RESOURCE_DIR);
    }

    /** Loads a model from classpath resources under {@code resourceDir} (e.g. {@code "/my/model/"}). */
    public static Guesslang load(Class<?> anchor, String resourceDir) throws IOException {
        String dir = resourceDir.endsWith("/") ? resourceDir : resourceDir + "/";
        return load(name -> {
            InputStream in = anchor.getResourceAsStream(dir + name);
            if (in == null) {
                throw new IOException("Model resource not found: " + dir + name);
            }
            return in;
        });
    }

    /** Loads a model from a directory produced by {@code tools/export_java.py}. */
    public static Guesslang load(Path directory) throws IOException {
        return load(name -> Files.newInputStream(directory.resolve(name)));
    }

    private interface Source {
        InputStream open(String name) throws IOException;
    }

    private static Guesslang load(Source source) throws IOException {
        Properties props = new Properties();
        props.load(new StringReader(readText(source, PROPERTIES)));
        String format = props.getProperty("format.version");
        if (!SUPPORTED_FORMAT.equals(format)) {
            throw new IOException("Unsupported model format version: " + format);
        }
        int sequenceLength = intProperty(props, "sequence.length");
        int ngrams = intProperty(props, "wide.ngrams");
        double epsilon = Double.parseDouble(requireProperty(props, "layer_normalization.epsilon"));

        List<String> extensions = new ArrayList<>();
        List<String> languages = new ArrayList<>();
        for (String line : lines(readText(source, LABELS))) {
            if (line.isEmpty()) {
                continue;
            }
            int tab = line.indexOf('\t');
            if (tab <= 0) {
                throw new IOException("Malformed labels line: " + line);
            }
            extensions.add(line.substring(0, tab));
            languages.add(line.substring(tab + 1));
        }

        Map<String, Tensor> weights;
        try (InputStream in = new BufferedInputStream(source.open(WEIGHTS), 1 << 16)) {
            weights = Tensor.readAll(in);
        }
        Tensor idf = weights.get("wide_idf");
        if (idf == null) {
            throw new IOException("Missing tensor: wide_idf");
        }

        List<String> deepVocab = vocabulary(readText(source, DEEP_VOCAB));
        List<String> wideVocab = vocabulary(readText(source, WIDE_VOCAB));
        WideDeepModel model;
        TextVectorizer vectorizer;
        try {
            model = new WideDeepModel(weights, epsilon);
            vectorizer = new TextVectorizer(deepVocab, wideVocab, idf.data, sequenceLength, ngrams);
        } catch (IllegalArgumentException e) {
            throw new IOException("Invalid model: " + e.getMessage(), e);
        }
        if (model.outputSize() != extensions.size()) {
            throw new IOException("Model has " + model.outputSize() + " outputs but " + extensions.size() + " labels");
        }
        if (deepVocab.size() != model.vocabularySize() || wideVocab.size() != model.wideSize()) {
            throw new IOException("Vocabulary sizes do not match the model weights");
        }
        return new Guesslang(vectorizer, model, List.copyOf(extensions), List.copyOf(languages), props);
    }

    private static String readText(Source source, String name) throws IOException {
        try (InputStream in = source.open(name)) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    /** Splits on '\n' only: tokens may contain other Unicode line separators. */
    private static List<String> lines(String text) {
        return Arrays.asList(text.split("\n", -1));
    }

    private static List<String> vocabulary(String text) {
        List<String> lines = lines(text);
        // file ends with a newline -> drop the trailing empty element
        return lines.subList(0, lines.size() - 1);
    }

    private static String requireProperty(Properties props, String key) throws IOException {
        String value = props.getProperty(key);
        if (value == null) {
            throw new IOException("Missing model property: " + key);
        }
        return value.trim();
    }

    private static int intProperty(Properties props, String key) throws IOException {
        return Integer.parseInt(requireProperty(props, key));
    }

    /** Language names in model output order. */
    public List<String> languages() {
        return languages;
    }

    /** Model labels (file extensions) in model output order. */
    public List<String> extensions() {
        return extensions;
    }

    /** Metadata of the loaded model (name, export date, hyper parameters). */
    public String modelProperty(String key) {
        return properties.getProperty(key);
    }

    /** Raw logits in model output order. */
    public float[] logits(String text) {
        List<String> tokens = TextVectorizer.tokenize(text);
        return model.logits(vectorizer.deepIds(tokens), vectorizer.wideTfidf(tokens));
    }

    /** Probabilities in model output order. */
    public double[] probabilities(String text) {
        return softmax(logits(text));
    }

    /** All languages, most probable first. */
    public List<Prediction> predict(String text) {
        return predict(text, extensions.size());
    }

    /** The {@code limit} most probable languages, most probable first. */
    public List<Prediction> predict(String text, int limit) {
        double[] probs = probabilities(text);
        List<Prediction> all = new ArrayList<>(probs.length);
        for (int i = 0; i < probs.length; i++) {
            all.add(new Prediction(languages.get(i), extensions.get(i), probs[i]));
        }
        all.sort(Comparator.comparingDouble(Prediction::probability).reversed());
        return Collections.unmodifiableList(all.subList(0, Math.max(0, Math.min(limit, all.size()))));
    }

    static double[] softmax(float[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v);
        }
        double sum = 0;
        double[] out = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            out[i] = Math.exp(logits[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < out.length; i++) {
            out[i] /= sum;
        }
        return out;
    }

    // Package-private hooks for parity tests.
    TextVectorizer vectorizer() {
        return vectorizer;
    }

    WideDeepModel model() {
        return model;
    }
}
