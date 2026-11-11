# guesslang-java

Programming language detection with the [Guesslang](https://github.com/yoeo/guesslang) wide & deep model,
in pure Java.

The bundled model (`mainframe_model`, 58 languages including COBOL, JCL, HLASM, PL/I and REXX)
gives the same predictions as the Keras model. The test split accuracy is 89.45%, the same as the Python evaluation.

## Usage

```java
Guesslang guesslang = Guesslang.load();          // ~50 ms; load once, thread-safe
List<Prediction> top = guesslang.predict(text, 3);
top.get(0).language();     // "COBOL"
top.get(0).extension();    // "cbl"
top.get(0).probability();  // 0.997
```

Inference takes about 1 ms per file on one core. The model reads the first 512 tokens for the deep branch and the whole
text for the tf-idf branch, so callers should cap very large inputs (e.g. first 16 KB).

## Modules

| Module | Content |
|--------|---------|
| `core` | the library: `Guesslang`, `Prediction`, and the model in `src/main/resources/com/guesslang/model/` |
| `cli`  | command line wrapper for manual checks and evaluation |

```bash
./gradlew test                     # parity tests against the Keras model
./gradlew :cli:installDist
echo '//MYJOB JOB (ACCT),CLASS=A' | cli/build/install/guesslang/bin/guesslang --top 3
cli/build/install/guesslang/bin/guesslang --eval ../guesslangtools/generated_datasets/files/test
```

Java 21+.

## Copying into another project

Copy `core/src/main/java/com/guesslang/` and `core/src/main/resources/com/guesslang/model/`.
To keep the model somewhere else on the classpath, use `Guesslang.load(anchorClass, "/your/resource/dir/")`.
To load it from disk, use `Guesslang.load(Path)`.

## Updating the model

The model files are generated from a trained Keras model by `tools/export_java.py` in the
[guesslang](../guesslang) repository:

```bash
cd ../guesslang
.venv/bin/python tools/export_java.py \
    --model-dir mainframe_model \
    --out ../guesslang-java/core/src/main/resources/com/guesslang/model \
    --golden ../guesslang-java/core/src/test/resources/golden.json.gz \
    --test-dir ../guesslangtools/generated_datasets/files/test
cd ../guesslang-java && ./gradlew test
```

`golden.json.gz` holds the Keras outputs for about 300 test files and edge cases: token ids, tf-idf vectors and logits.
`GoldenParityTest` checks that the Java pipeline reproduces them.

### Model files

| File | Content |
|------|---------|
| `model.properties` | format version, sequence length, n-grams, layer norm epsilon |
| `labels.tsv` | `extension<TAB>language`, in model output order |
| `deep_vocab.txt` / `wide_vocab.txt` | vectorizer vocabularies, one token per line (`\n` separated) |
| `weights.bin` | float32 tensors, big-endian (`GLW1` format, see `Tensor.java`) |

### Preprocessing rules that matter

Reproduced from Keras `TextVectorization(standardize="lower_and_strip_punctuation", split="whitespace")`:

- only ASCII letters are lowercased;
- only ASCII punctuation is removed, including `_`, `#` and `+`, before splitting (`a.b` becomes `ab`);
- only ASCII whitespace splits tokens (space, `\t`, `\n`, `\r`, `\v`, `\f`);
- deep input: first 512 tokens, unknown token = 1, padding = 0, and the average includes padding positions;
- wide input: counts of 1- and 2-grams multiplied by idf; unknown n-grams are counted at index 0.
