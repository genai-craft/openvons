# onnxruntime は JNI から Java のクラスを名前で引く (convertToTensorInfo など)。
# R8 が縮小・難読化すると java_class == null で JNI が abort する (release だけ落ちる)。
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**
