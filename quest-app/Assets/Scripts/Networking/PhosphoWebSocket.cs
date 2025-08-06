using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using NativeWebSocket;
using PhosphoQuest.Utils;

namespace PhosphoQuest.Networking
{
    public class PhosphoWebSocket : SingletonBehaviour<PhosphoWebSocket>
    {
        [Header("Connection Settings")]
        [SerializeField] private string serverIP = "192.168.1.100";
        [SerializeField] private int serverPort = 8080;
        [SerializeField] private bool useWebSocket = true;
        [SerializeField] private bool autoReconnect = true;
        [SerializeField] private float reconnectDelay = 3f;
        
        [Header("Debug")]
        [SerializeField] private bool debugMode = false;
        
        private WebSocket websocket;
        private bool isInitialized = false;
        private Queue<AppControlData> sendQueue = new Queue<AppControlData>();
        private float lastRateLogTime;
        private int messagesSentThisSecond;
        
        public event Action OnConnected;
        public event Action OnDisconnected;
        public event Action<string> OnError;
        public event Action<RobotStatus> OnStatusReceived;
        
        public bool IsConnected => websocket != null && websocket.State == WebSocketState.Open;
        public bool IsInitialized => isInitialized;
        public string ServerURL => $"{serverIP}:{serverPort}";

        protected override void Awake()
        {
            base.Awake();
            DontDestroyOnLoad(gameObject);
        }

        void Start()
        {
            // Load saved settings
            serverIP = PlayerPrefs.GetString("ServerIP", serverIP);
            serverPort = PlayerPrefs.GetInt("ServerPort", serverPort);
            
            StartCoroutine(InitializeConnection());
        }

        public void UpdateConnectionSettings(string ip, int port)
        {
            if (serverIP != ip || serverPort != port)
            {
                serverIP = ip;
                serverPort = port;
                PlayerPrefs.SetString("ServerIP", serverIP);
                PlayerPrefs.SetInt("ServerPort", serverPort);
                PlayerPrefs.Save();
                
                StartCoroutine(Reconnect());
            }
        }

        IEnumerator InitializeConnection()
        {
            // First initialize robot position
            yield return StartCoroutine(InitializeRobot());
            
            // Then connect WebSocket
            if (useWebSocket)
            {
                yield return StartCoroutine(ConnectWebSocket());
            }
        }

        IEnumerator InitializeRobot()
        {
            string url = $"http://{serverIP}:{serverPort}/move/init";
            
            if (debugMode) Debug.Log($"Initializing robot at: {url}");
            
            using (var www = UnityWebRequest.Post(url, ""))
            {
                www.timeout = 5;
                yield return www.SendWebRequest();
                
                if (www.result == UnityWebRequest.Result.Success)
                {
                    Debug.Log("Robot initialized successfully");
                    isInitialized = true;
                }
                else
                {
                    Debug.LogError($"Failed to initialize robot: {www.error}");
                    OnError?.Invoke(www.error);
                }
            }
        }

        IEnumerator ConnectWebSocket()
        {
            string wsUrl = $"ws://{serverIP}:{serverPort}/move/teleop/ws";
            
            if (debugMode) Debug.Log($"Connecting to WebSocket: {wsUrl}");
            
            websocket = new WebSocket(wsUrl);

            websocket.OnOpen += () =>
            {
                Debug.Log("WebSocket connected!");
                OnConnected?.Invoke();
            };

            websocket.OnError += (e) =>
            {
                Debug.LogError($"WebSocket error: {e}");
                OnError?.Invoke(e);
            };

            websocket.OnClose += (e) =>
            {
                Debug.Log($"WebSocket closed: {e}");
                OnDisconnected?.Invoke();
                
                if (autoReconnect && Application.isPlaying)
                {
                    StartCoroutine(ReconnectAfterDelay());
                }
            };

            websocket.OnMessage += (bytes) =>
            {
                var message = System.Text.Encoding.UTF8.GetString(bytes);
                if (debugMode) Debug.Log($"Received: {message}");
                
                try
                {
                    var status = RobotStatus.FromJson(message);
                    OnStatusReceived?.Invoke(status);
                }
                catch (Exception e)
                {
                    if (debugMode) Debug.LogError($"Failed to parse message: {e.Message}");
                }
            };

            yield return websocket.Connect();
        }

        IEnumerator Reconnect()
        {
            if (websocket != null && websocket.State == WebSocketState.Open)
            {
                yield return websocket.Close();
            }
            
            yield return StartCoroutine(InitializeConnection());
        }

        IEnumerator ReconnectAfterDelay()
        {
            yield return new WaitForSeconds(reconnectDelay);
            yield return StartCoroutine(ConnectWebSocket());
        }

        void Update()
        {
            #if !UNITY_WEBGL || UNITY_EDITOR
            websocket?.DispatchMessageQueue();
            #endif
            
            // Process send queue
            while (sendQueue.Count > 0 && IsConnected)
            {
                var data = sendQueue.Dequeue();
                SendControlDataImmediate(data);
            }
            
            // Log send rate if debugging
            if (debugMode && Time.time - lastRateLogTime >= 1f)
            {
                if (messagesSentThisSecond > 0)
                {
                    Debug.Log($"Send rate: {messagesSentThisSecond} msgs/sec");
                }
                lastRateLogTime = Time.time;
                messagesSentThisSecond = 0;
            }
        }

        public void SendControlData(AppControlData data)
        {
            if (!isInitialized)
            {
                if (debugMode) Debug.LogWarning("Robot not initialized yet");
                return;
            }

            if (IsConnected)
            {
                SendControlDataImmediate(data);
            }
            else
            {
                // Queue for later if not connected
                if (sendQueue.Count < 100) // Prevent queue overflow
                {
                    sendQueue.Enqueue(data);
                }
            }
        }

        private void SendControlDataImmediate(AppControlData data)
        {
            websocket.SendText(data.ToJson());
            messagesSentThisSecond++;
        }

        private async void OnApplicationPause(bool pauseStatus)
        {
            if (!pauseStatus && websocket != null && websocket.State == WebSocketState.Closed)
            {
                await websocket.Connect();
            }
        }

        private async void OnApplicationQuit()
        {
            if (websocket != null && websocket.State == WebSocketState.Open)
            {
                await websocket.Close();
            }
        }

        private void OnDestroy()
        {
            if (websocket != null && websocket.State == WebSocketState.Open)
            {
                websocket.Close();
            }
        }
    }
}