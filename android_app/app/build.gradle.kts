plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

val generatedPythonDir = layout.buildDirectory.dir("generated/raksha_python")

val syncRakshaPython by tasks.registering(Sync::class) {
    from(rootProject.projectDir.parentFile) {
        include("main.py", "gui.py", "src/**", "gui/**", "config/**")
        exclude("**/__pycache__/**", "**/*.pyc")
    }
    into(generatedPythonDir)
}

tasks.named("preBuild").configure {
    dependsOn(syncRakshaPython)
}

tasks.matching { it.name.contains("Python", ignoreCase = true) }.configureEach {
    dependsOn(syncRakshaPython)
}

android {
    namespace = "com.rakshakavach.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.rakshakavach.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

chaquopy {
    defaultConfig {
        version = "3.12"
        pip {
            install("PyYAML>=6.0.0")
            install("phonenumbers>=8.13.0")
            install("indic-transliteration>=2.3.50")
            install("regex>=2024.0.0")
            install("cryptography>=42.0.0")
        }
    }
    sourceSets {
        getByName("main") {
            srcDir(generatedPythonDir)
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
}
