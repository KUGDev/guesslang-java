package com.guesslang;

/**
 * One language guess.
 *
 * @param language    human readable language name, e.g. {@code "COBOL"}
 * @param extension   model label (file extension), e.g. {@code "cbl"}
 * @param probability softmax probability in [0, 1]
 */
public record Prediction(String language, String extension, double probability) {
}
