using System.Collections;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.Interaction.Toolkit;
using PhosphoQuest.Networking;

namespace PhosphoQuest.Controllers
{
    public class HandTrackingController : MonoBehaviour
    {
        [Header("Controller Settings")]
        [SerializeField] private XRController rightController;
        [SerializeField] private XRController leftController;
        [SerializeField] private float sendRate = 30f; // Hz
        [SerializeField] private float deadzone = 0.1f;
        [SerializeField] private bool useGripButton = true;
        
        [Header("Robot Configuration")]
        [SerializeField] private string robotName = "so-100";
        [SerializeField] private bool isMobileRobot = false;
        
        [Header("Calibration")]
        [SerializeField] private bool autoCalibrate = true;
        [SerializeField] private float calibrationDelay = 2f;
        
        [Header("Debug")]
        [SerializeField] private bool showDebugVisuals = false;
        [SerializeField] private GameObject debugSphere;
        
        private float sendInterval;
        private float lastSendTime;
        private Vector3 initialRightPos;
        private Quaternion initialRightRot;
        private Vector3 initialLeftPos;
        private Quaternion initialLeftRot;
        private bool isCalibrated = false;
        private bool isControlEnabled = true;

        void Start()
        {
            sendInterval = 1f / sendRate;
            
            if (autoCalibrate)
            {
                StartCoroutine(CalibrateControllers());
            }
            
            // Subscribe to connection events
            PhosphoWebSocket.Instance.OnConnected += OnConnected;
            PhosphoWebSocket.Instance.OnDisconnected += OnDisconnected;
        }

        void OnDestroy()
        {
            if (PhosphoWebSocket.Instance != null)
            {
                PhosphoWebSocket.Instance.OnConnected -= OnConnected;
                PhosphoWebSocket.Instance.OnDisconnected -= OnDisconnected;
            }
        }

        void OnConnected()
        {
            Debug.Log("Connected to phosphobot - recalibrating controllers");
            if (autoCalibrate)
            {
                StartCoroutine(CalibrateControllers());
            }
        }

        void OnDisconnected()
        {
            Debug.Log("Disconnected from phosphobot");
        }

        IEnumerator CalibrateControllers()
        {
            isCalibrated = false;
            
            // Wait for controllers to be tracked
            yield return new WaitForSeconds(calibrationDelay);
            
            if (rightController)
            {
                initialRightPos = rightController.transform.position;
                initialRightRot = rightController.transform.rotation;
                Debug.Log($"Right controller calibrated at: {initialRightPos}");
            }
            
            if (leftController)
            {
                initialLeftPos = leftController.transform.position;
                initialLeftRot = leftController.transform.rotation;
                Debug.Log($"Left controller calibrated at: {initialLeftPos}");
            }
            
            isCalibrated = true;
            Debug.Log("Controllers calibrated");
        }

        public void RecalibrateControllers()
        {
            StartCoroutine(CalibrateControllers());
        }

        void Update()
        {
            if (!isCalibrated || !PhosphoWebSocket.Instance.IsConnected || !isControlEnabled)
                return;
            
            // Check for manual calibration trigger
            if (rightController && rightController.inputDevice.TryGetFeatureValue(CommonUsages.primaryButton, out bool aPressed) && aPressed)
            {
                RecalibrateControllers();
            }
            
            if (Time.time - lastSendTime >= sendInterval)
            {
                SendControllerData();
                lastSendTime = Time.time;
            }
        }

        void SendControllerData()
        {
            // Send right controller data
            if (rightController && rightController.enabled)
            {
                // Check if grip button is required
                if (useGripButton)
                {
                    rightController.inputDevice.TryGetFeatureValue(CommonUsages.gripButton, out float gripValue);
                    if (gripValue < 0.5f) return; // Skip if grip not pressed
                }
                
                SendHandData(rightController, "right", initialRightPos, initialRightRot);
            }
            
            // Send left controller data (for dual-arm setups)
            if (leftController && leftController.enabled)
            {
                // Check if grip button is required
                if (useGripButton)
                {
                    leftController.inputDevice.TryGetFeatureValue(CommonUsages.gripButton, out float gripValue);
                    if (gripValue < 0.5f) return; // Skip if grip not pressed
                }
                
                SendHandData(leftController, "left", initialLeftPos, initialLeftRot);
            }
        }

        void SendHandData(XRController controller, string source, Vector3 initialPos, Quaternion initialRot)
        {
            var data = new AppControlData();
            data.source = source;
            
            // Get relative position
            Vector3 relativePos = controller.transform.position - initialPos;
            
            // Scale the movement (1 meter movement = 10cm robot movement)
            relativePos *= 0.1f;
            
            // Convert Unity coordinates to robot coordinates
            ConvertToRobotCoordinates(relativePos, controller.transform.rotation, 
                                      out data.x, out data.y, out data.z,
                                      out data.rx, out data.ry, out data.rz);
            
            // Get trigger value for gripper
            controller.inputDevice.TryGetFeatureValue(CommonUsages.trigger, out float triggerValue);
            data.open = 1f - triggerValue; // Invert for phosphobot
            
            // Get joystick for mobile robots
            if (isMobileRobot)
            {
                controller.inputDevice.TryGetFeatureValue(CommonUsages.primary2DAxis, out Vector2 joystick);
                data.direction_x = Mathf.Abs(joystick.x) > deadzone ? joystick.x : 0f;
                data.direction_y = Mathf.Abs(joystick.y) > deadzone ? joystick.y : 0f;
            }
            
            // Update debug visual if enabled
            if (showDebugVisuals && debugSphere != null)
            {
                debugSphere.transform.position = controller.transform.position;
                debugSphere.transform.localScale = Vector3.one * (0.05f + triggerValue * 0.05f);
            }
            
            PhosphoWebSocket.Instance.SendControlData(data);
        }

        void ConvertToRobotCoordinates(Vector3 position, Quaternion rotation,
                                       out float x, out float y, out float z,
                                       out float rx, out float ry, out float rz)
        {
            // Unity to Robot position conversion (Y/Z swap)
            x = position.x;
            y = position.z;
            z = position.y;
            
            // Get euler angles
            Vector3 euler = rotation.eulerAngles;
            
            // Robot-specific orientation mapping
            switch (robotName)
            {
                case "agilex-piper":
                    rx = euler.y;
                    ry = euler.x;
                    rz = euler.z;
                    break;
                case "wx-250s":
                case "koch-v1.1":
                case "so-100":
                default:
                    rx = -euler.x;
                    ry = -euler.z;
                    rz = -euler.y;
                    break;
            }
            
            // Normalize angles to [-180, 180]
            rx = NormalizeAngle(rx);
            ry = NormalizeAngle(ry);
            rz = NormalizeAngle(rz);
        }

        float NormalizeAngle(float angle)
        {
            while (angle > 180f) angle -= 360f;
            while (angle < -180f) angle += 360f;
            return angle;
        }

        public void SetControlEnabled(bool enabled)
        {
            isControlEnabled = enabled;
        }
    }
}