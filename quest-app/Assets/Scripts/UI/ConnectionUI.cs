using UnityEngine;
using UnityEngine.UI;
using TMPro;
using PhosphoQuest.Networking;

namespace PhosphoQuest.UI
{
    public class ConnectionUI : MonoBehaviour
    {
        [Header("UI Panels")]
        [SerializeField] private GameObject connectionPanel;
        [SerializeField] private GameObject statusPanel;
        [SerializeField] private GameObject errorPanel;
        
        [Header("Connection UI Elements")]
        [SerializeField] private TMP_InputField ipInput;
        [SerializeField] private TMP_InputField portInput;
        [SerializeField] private Button connectButton;
        [SerializeField] private TextMeshProUGUI connectionStatusText;
        
        [Header("Status UI Elements")]
        [SerializeField] private TextMeshProUGUI robotStatusText;
        [SerializeField] private TextMeshProUGUI grippedText;
        [SerializeField] private TextMeshProUGUI actionsText;
        [SerializeField] private Button disconnectButton;
        [SerializeField] private Image connectionIndicator;
        
        [Header("Error UI Elements")]
        [SerializeField] private TextMeshProUGUI errorText;
        [SerializeField] private Button retryButton;
        
        [Header("Colors")]
        [SerializeField] private Color connectedColor = Color.green;
        [SerializeField] private Color disconnectedColor = Color.red;
        [SerializeField] private Color connectingColor = Color.yellow;
        
        private bool isConnecting = false;

        void Start()
        {
            // Load saved connection settings
            ipInput.text = PlayerPrefs.GetString("ServerIP", "192.168.1.100");
            portInput.text = PlayerPrefs.GetString("ServerPort", "8080");
            
            // Setup button listeners
            connectButton.onClick.AddListener(OnConnectClicked);
            disconnectButton.onClick.AddListener(OnDisconnectClicked);
            retryButton.onClick.AddListener(OnRetryClicked);
            
            // Subscribe to events
            PhosphoWebSocket.Instance.OnConnected += OnConnected;
            PhosphoWebSocket.Instance.OnDisconnected += OnDisconnected;
            PhosphoWebSocket.Instance.OnError += OnError;
            PhosphoWebSocket.Instance.OnStatusReceived += OnStatusReceived;
            
            // Initial UI state
            ShowConnectionPanel();
        }

        void OnDestroy()
        {
            // Unsubscribe from events
            if (PhosphoWebSocket.Instance != null)
            {
                PhosphoWebSocket.Instance.OnConnected -= OnConnected;
                PhosphoWebSocket.Instance.OnDisconnected -= OnDisconnected;
                PhosphoWebSocket.Instance.OnError -= OnError;
                PhosphoWebSocket.Instance.OnStatusReceived -= OnStatusReceived;
            }
        }

        void OnConnectClicked()
        {
            if (isConnecting) return;
            
            // Validate input
            if (string.IsNullOrEmpty(ipInput.text))
            {
                ShowError("Please enter a valid IP address");
                return;
            }
            
            if (!int.TryParse(portInput.text, out int port) || port < 1 || port > 65535)
            {
                ShowError("Please enter a valid port number (1-65535)");
                return;
            }
            
            isConnecting = true;
            connectionStatusText.text = "Connecting...";
            connectionStatusText.color = connectingColor;
            connectButton.interactable = false;
            
            // Update connection settings
            PhosphoWebSocket.Instance.UpdateConnectionSettings(ipInput.text, port);
        }

        void OnDisconnectClicked()
        {
            // For now, we'll just show the connection panel
            // In a full implementation, you'd properly disconnect
            ShowConnectionPanel();
        }

        void OnRetryClicked()
        {
            ShowConnectionPanel();
            OnConnectClicked();
        }

        void OnConnected()
        {
            isConnecting = false;
            ShowStatusPanel();
            
            robotStatusText.text = $"Connected to {PhosphoWebSocket.Instance.ServerURL}";
            connectionIndicator.color = connectedColor;
        }

        void OnDisconnected()
        {
            isConnecting = false;
            ShowConnectionPanel();
            
            connectionStatusText.text = "Disconnected";
            connectionStatusText.color = disconnectedColor;
            connectButton.interactable = true;
        }

        void OnError(string error)
        {
            isConnecting = false;
            ShowErrorPanel(error);
            connectButton.interactable = true;
        }

        void OnStatusReceived(RobotStatus status)
        {
            grippedText.text = status.is_object_gripped ? 
                $"Object Gripped ({status.is_object_gripped_source})" : 
                "No Object";
            actionsText.text = $"Actions/sec: {status.nb_actions_received}";
            
            // Flash connection indicator
            if (connectionIndicator != null)
            {
                connectionIndicator.color = connectedColor;
                CancelInvoke(nameof(ResetConnectionIndicator));
                Invoke(nameof(ResetConnectionIndicator), 0.1f);
            }
        }

        void ResetConnectionIndicator()
        {
            if (connectionIndicator != null)
            {
                connectionIndicator.color = connectedColor * 0.7f;
            }
        }

        void ShowConnectionPanel()
        {
            connectionPanel.SetActive(true);
            statusPanel.SetActive(false);
            errorPanel.SetActive(false);
        }

        void ShowStatusPanel()
        {
            connectionPanel.SetActive(false);
            statusPanel.SetActive(true);
            errorPanel.SetActive(false);
        }

        void ShowErrorPanel(string error)
        {
            connectionPanel.SetActive(false);
            statusPanel.SetActive(false);
            errorPanel.SetActive(true);
            errorText.text = $"Connection Error:\n{error}";
        }

        void ShowError(string message)
        {
            connectionStatusText.text = message;
            connectionStatusText.color = disconnectedColor;
        }
    }
}