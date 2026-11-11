package com.guesslang.cli;

import com.guesslang.Guesslang;
import com.guesslang.Prediction;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Stream;

/** Command line wrapper around {@link Guesslang}, for manual checks and evaluation. */
public final class Main {

    private static final String USAGE = """
            Usage:
              guesslang [--top N] [--json] [--model DIR] [FILE]   guess the language of FILE (stdin if omitted)
              guesslang --eval DIR [--limit N] [--model DIR]      accuracy on files named <anything>.<extension>
            """;

    public static void main(String[] args) {
        try {
            System.exit(run(args, System.out));
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    static int run(String[] args, PrintStream out) throws IOException {
        int top = 3;
        boolean json = false;
        Path modelDir = null;
        Path evalDir = null;
        int limit = Integer.MAX_VALUE;
        Path file = null;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-h", "--help" -> {
                    out.print(USAGE);
                    return 0;
                }
                case "--top" -> top = Integer.parseInt(value(args, ++i));
                case "--json" -> json = true;
                case "--model" -> modelDir = Path.of(value(args, ++i));
                case "--eval" -> evalDir = Path.of(value(args, ++i));
                case "--limit" -> limit = Integer.parseInt(value(args, ++i));
                default -> {
                    if (args[i].startsWith("-") || file != null) {
                        throw new IllegalArgumentException("Unexpected argument: " + args[i]);
                    }
                    file = Path.of(args[i]);
                }
            }
        }

        long start = System.nanoTime();
        Guesslang guesslang = modelDir == null ? Guesslang.load() : Guesslang.load(modelDir);
        long loadMs = (System.nanoTime() - start) / 1_000_000;

        if (evalDir != null) {
            System.err.println("Model loaded in " + loadMs + " ms");
            return evaluate(guesslang, evalDir, limit, out);
        }

        String text = file == null
                ? new String(System.in.readAllBytes(), StandardCharsets.UTF_8)
                : Files.readString(file, StandardCharsets.UTF_8);
        List<Prediction> predictions = guesslang.predict(text, top);
        if (json) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < predictions.size(); i++) {
                Prediction p = predictions.get(i);
                sb.append(i == 0 ? "" : ",").append("{\"language\":\"").append(escape(p.language()))
                        .append("\",\"extension\":\"").append(escape(p.extension()))
                        .append("\",\"probability\":").append(p.probability()).append('}');
            }
            out.println(sb.append(']'));
        } else {
            for (Prediction p : predictions) {
                out.printf(Locale.ROOT, "%-12s %.4f%n", p.language(), p.probability());
            }
        }
        return 0;
    }

    private static String value(String[] args, int i) {
        if (i >= args.length) {
            throw new IllegalArgumentException("Missing value for " + args[i - 1]);
        }
        return args[i];
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    /** Per-extension accuracy; files that are not valid UTF-8 or have at most 30 chars are skipped like in training. */
    private static int evaluate(Guesslang guesslang, Path dir, int limitPerLanguage, PrintStream out) throws IOException {
        List<String> extensions = guesslang.extensions();
        Map<String, int[]> stats = new TreeMap<>(); // extension -> {total, correct}
        Map<String, Map<String, Integer>> confusions = new TreeMap<>();
        List<Path> files;
        try (Stream<Path> s = Files.list(dir)) {
            files = new ArrayList<>(s.sorted().toList());
        }
        long start = System.nanoTime();
        int processed = 0;
        for (Path f : files) {
            String name = f.getFileName().toString();
            String ext = name.substring(name.lastIndexOf('.') + 1);
            if (!extensions.contains(ext)) {
                continue;
            }
            int[] st = stats.computeIfAbsent(ext, k -> new int[2]);
            if (st[0] >= limitPerLanguage) {
                continue;
            }
            String text;
            try {
                text = StandardCharsets.UTF_8.newDecoder()
                        .onMalformedInput(CodingErrorAction.REPORT)
                        .decode(java.nio.ByteBuffer.wrap(Files.readAllBytes(f))).toString();
            } catch (CharacterCodingException e) {
                continue;
            }
            if (text.length() <= 30) {
                continue;
            }
            String predicted = guesslang.predict(text, 1).get(0).extension();
            st[0]++;
            processed++;
            if (predicted.equals(ext)) {
                st[1]++;
            } else {
                confusions.computeIfAbsent(ext, k -> new TreeMap<>()).merge(predicted, 1, Integer::sum);
            }
        }
        double seconds = (System.nanoTime() - start) / 1e9;

        int total = 0;
        int correct = 0;
        for (Map.Entry<String, int[]> e : stats.entrySet()) {
            int[] st = e.getValue();
            total += st[0];
            correct += st[1];
            String worst = confusions.getOrDefault(e.getKey(), Map.of()).entrySet().stream()
                    .sorted(Map.Entry.<String, Integer>comparingByValue().reversed()).limit(3)
                    .map(c -> c.getKey() + ":" + c.getValue()).reduce((a, b) -> a + ", " + b).orElse("");
            out.printf(Locale.ROOT, "%-11s %6.2f%%  %6d / %-6d %s%n", e.getKey(),
                    st[0] == 0 ? 0.0 : 100.0 * st[1] / st[0], st[1], st[0], worst);
        }
        out.printf(Locale.ROOT, "TOTAL       %6.2f%%  %6d / %d  (%.0f files/s)%n",
                total == 0 ? 0.0 : 100.0 * correct / total, correct, total, processed / seconds);
        return 0;
    }
}
