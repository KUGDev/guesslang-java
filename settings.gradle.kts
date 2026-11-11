plugins {
    // Downloads the JDK toolchain (Java 21) into ~/.gradle/jdks when it is not installed locally.
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}

rootProject.name = "guesslang-java"

include("core", "cli")
