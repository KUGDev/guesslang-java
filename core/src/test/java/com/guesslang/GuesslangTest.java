package com.guesslang;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GuesslangTest {

    private static Guesslang guesslang;

    @BeforeAll
    static void setUp() throws IOException {
        guesslang = Guesslang.load();
    }

    @Test
    void tokenizeFollowsKerasRules() {
        assertEquals(List.of("hello", "world"), TextVectorizer.tokenize("  Hello,\tWORLD!\n"));
        assertEquals(List.of("ab", "cd"), TextVectorizer.tokenize("a.b c_d"));
        // non-ASCII is neither lowercased nor treated as whitespace/punctuation
        assertEquals(List.of("ÄÖ", "a b", "«x»"), TextVectorizer.tokenize("ÄÖ a b «x»"));
        assertEquals(List.of(), TextVectorizer.tokenize(" \u000B\f\r\n"));
    }

    @Test
    void predictionsAreSortedProbabilities() {
        List<Prediction> all = guesslang.predict("print('hello')");
        assertEquals(guesslang.extensions().size(), all.size());
        double sum = all.stream().mapToDouble(Prediction::probability).sum();
        assertEquals(1.0, sum, 1e-9);
        for (int i = 1; i < all.size(); i++) {
            assertTrue(all.get(i - 1).probability() >= all.get(i).probability());
        }
        assertEquals(3, guesslang.predict("x", 3).size());
    }

    @Test
    void detectsMainframeLanguages() {
        assertTop("JCL", """
                //MYJOB    JOB (ACCT),'COMPILE',CLASS=A,MSGCLASS=X,NOTIFY=&SYSUID
                //STEP1    EXEC PGM=IGYCRCTL,PARM='LIST,MAP'
                //STEPLIB  DD DSN=IGY.SIGYCOMP,DISP=SHR
                //SYSPRINT DD SYSOUT=*
                //SYSIN    DD DSN=MY.COBOL.SRC(HELLO),DISP=SHR
                //SYSLIN   DD DSN=&&LOADSET,DISP=(MOD,PASS),UNIT=SYSDA,SPACE=(TRK,(3,3))
                """);
        assertTop("COBOL", """
                       IDENTIFICATION DIVISION.
                       PROGRAM-ID. HELLO.
                       DATA DIVISION.
                       WORKING-STORAGE SECTION.
                       01 WS-NAME PIC X(20) VALUE 'WORLD'.
                       PROCEDURE DIVISION.
                           DISPLAY 'HELLO ' WS-NAME.
                           STOP RUN.
                """);
    }

    private static void assertTop(String language, String text) {
        Prediction top = guesslang.predict(text, 1).get(0);
        assertEquals(language, top.language(), () -> "predictions: " + guesslang.predict(text, 5));
    }
}
