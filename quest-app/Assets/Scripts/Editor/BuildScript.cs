using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;
using System.IO;

namespace PhosphoQuest.Editor
{
    public class BuildScript
    {
        [MenuItem("PhosphoQuest/Build Quest APK")]
        public static void BuildQuest()
        {
            string buildPath = Path.Combine(Application.dataPath, "../Builds");
            if (!Directory.Exists(buildPath))
                Directory.CreateDirectory(buildPath);

            // Configure player settings
            PlayerSettings.productName = "PhosphoQuest";
            PlayerSettings.companyName = "Phospho";
            PlayerSettings.applicationIdentifier = "ai.phospho.quest";
            
            // Quest-specific settings
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel29;
            PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto;
            
            // XR Settings
            PlayerSettings.virtualRealitySupported = true;
            
            // Graphics settings
            PlayerSettings.colorSpace = ColorSpace.Linear;
            
            // Find all scenes in the project
            string[] scenes = GetScenePaths();
            
            BuildPlayerOptions buildOptions = new BuildPlayerOptions
            {
                scenes = scenes,
                locationPathName = Path.Combine(buildPath, "PhosphoQuest.apk"),
                target = BuildTarget.Android,
                options = BuildOptions.None
            };

            Debug.Log("Starting Quest build...");
            BuildReport report = BuildPipeline.BuildPlayer(buildOptions);
            
            if (report.summary.result == BuildResult.Succeeded)
            {
                Debug.Log($"Build succeeded: {report.summary.outputPath}");
                Debug.Log($"Total size: {report.summary.totalSize / 1024 / 1024} MB");
                Debug.Log($"Build time: {report.summary.totalTime.TotalMinutes:F2} minutes");
            }
            else if (report.summary.result == BuildResult.Failed)
            {
                Debug.LogError("Build failed!");
                if (Application.isBatchMode)
                {
                    EditorApplication.Exit(1);
                }
            }
        }

        [MenuItem("PhosphoQuest/Build and Run on Quest")]
        public static void BuildAndRun()
        {
            string buildPath = Path.Combine(Application.dataPath, "../Builds");
            if (!Directory.Exists(buildPath))
                Directory.CreateDirectory(buildPath);

            // Configure same as BuildQuest
            PlayerSettings.productName = "PhosphoQuest";
            PlayerSettings.companyName = "Phospho";
            PlayerSettings.applicationIdentifier = "ai.phospho.quest";
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel29;

            BuildPlayerOptions buildOptions = new BuildPlayerOptions
            {
                scenes = GetScenePaths(),
                locationPathName = Path.Combine(buildPath, "PhosphoQuest.apk"),
                target = BuildTarget.Android,
                options = BuildOptions.AutoRunPlayer
            };

            BuildPipeline.BuildPlayer(buildOptions);
        }

        [MenuItem("PhosphoQuest/Configure Project Settings")]
        public static void ConfigureProjectSettings()
        {
            // Set company and product name
            PlayerSettings.companyName = "Phospho";
            PlayerSettings.productName = "PhosphoQuest";
            PlayerSettings.applicationIdentifier = "ai.phospho.quest";
            
            // Android settings
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel29;
            PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto;
            
            // Set to use IL2CPP
            PlayerSettings.SetScriptingBackend(BuildTargetGroup.Android, ScriptingImplementation.IL2CPP);
            
            // Enable ARM64 support
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            
            // Graphics settings
            PlayerSettings.colorSpace = ColorSpace.Linear;
            
            // XR Plugin Management would be configured here if using newer Unity versions
            // This would typically be done through the XR Plugin Management settings
            
            Debug.Log("Project settings configured for Quest development");
        }

        private static string[] GetScenePaths()
        {
            // For now, return the main scene
            // In a real project, you'd search for all scenes in the Assets folder
            return new string[] { "Assets/Scenes/MainScene.unity" };
        }
    }
}