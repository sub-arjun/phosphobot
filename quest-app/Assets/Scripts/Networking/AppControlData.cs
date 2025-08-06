using System;
using UnityEngine;

namespace PhosphoQuest.Networking
{
    [Serializable]
    public class AppControlData
    {
        public float x;
        public float y;
        public float z;
        public float rx;  // Pitch in degrees
        public float ry;  // Yaw in degrees
        public float rz;  // Roll in degrees
        public float open;  // 0 = closed, 1 = open
        public string source = "right";  // "left" or "right"
        public float timestamp;
        public float direction_x = 0f;  // For mobile robots
        public float direction_y = 0f;  // For mobile robots

        public AppControlData()
        {
            timestamp = Time.realtimeSinceStartup;
        }

        public string ToJson()
        {
            return JsonUtility.ToJson(this);
        }

        public static AppControlData FromJson(string json)
        {
            return JsonUtility.FromJson<AppControlData>(json);
        }
    }

    [Serializable]
    public class RobotStatus
    {
        public bool is_object_gripped;
        public string is_object_gripped_source;
        public int nb_actions_received;

        public static RobotStatus FromJson(string json)
        {
            return JsonUtility.FromJson<RobotStatus>(json);
        }
    }

    [Serializable]
    public class StatusResponse
    {
        public string status;
        public string message;
    }

    [Serializable]
    public class UDPServerInfo
    {
        public string host;
        public int port;
    }
}