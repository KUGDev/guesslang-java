plugins {
    application
}

dependencies {
    implementation(project(":core"))
}

application {
    mainClass.set("com.guesslang.cli.Main")
    applicationName = "guesslang"
}
