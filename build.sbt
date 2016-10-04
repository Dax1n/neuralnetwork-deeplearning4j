name := """hello-scala"""

version := "1.0"

scalaVersion := "2.11.7"

libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.4" % "test"
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc3.9"
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.4.0"
libraryDependencies += "org.bytedeco" % "javacpp" % "1.2.4"

fork in run := true