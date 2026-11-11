plugins {
    `java-library`
}

// The core library has no runtime dependencies on purpose: it is meant to be
// embedded (or copied) into an IntelliJ plugin without native libraries.
dependencies {
    testImplementation(platform("org.junit:junit-bom:6.1.3"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testImplementation("com.fasterxml.jackson.core:jackson-databind:2.23.0")
}

tasks.test {
    useJUnitPlatform()
    testLogging {
        events("failed")
        showStandardStreams = true
    }
}
