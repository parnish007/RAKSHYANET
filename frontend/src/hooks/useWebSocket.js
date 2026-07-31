import { useEffect, useState, useRef, useCallback } from 'react';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';
const RECONNECT_MS   = 3000;
const MAX_HISTORY    = 100;

// Message types mirroring backend WSMessage constants
export const MSG_EVENT_PROCESSED      = 'EVENT_PROCESSED';
export const MSG_REOPTIMIZATION_START = 'REOPTIMIZATION_START';
export const MSG_REOPTIMIZATION_DONE  = 'REOPTIMIZATION_DONE';
export const MSG_HITL_SUBMITTED       = 'HITL_SUBMITTED';
export const MSG_HITL_APPROVED        = 'HITL_APPROVED';
export const MSG_HITL_REJECTED        = 'HITL_REJECTED';

/**
 * useWebSocket — connect to the RakshyaNet WebSocket server.
 *
 * Returns:
 *   messages        — array of all received WSMessage objects (last 100)
 *   isConnected     — live connection state
 *   lastMessage     — most recent message (or null)
 *   sendMessage     — fn(object) — serialises and sends via WS
 *   getByType       — fn(type) — filters messages by type
 *   clearMessages   — resets the message buffer
 */
export function useWebSocket(url = WS_URL) {
  const [messages,    setMessages]    = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);

  const wsRef          = useRef(null);
  const reconnectTimer = useRef(null);
  const isMounted      = useRef(true);

  const connect = useCallback(() => {
    // Don't open a second connection
    if (wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING) return;

    try {
      const socket = new WebSocket(url);
      wsRef.current = socket;

      socket.onopen = () => {
        if (!isMounted.current) return;
        setIsConnected(true);
      };

      socket.onmessage = (event) => {
        if (!isMounted.current) return;
        try {
          const msg = JSON.parse(event.data);
          const stamped = { ...msg, _ts: Date.now() };
          setLastMessage(stamped);
          setMessages(prev => {
            const next = [...prev, stamped];
            return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
          });
        } catch {
          // silently skip malformed frames
        }
      };

      socket.onclose = () => {
        if (!isMounted.current) return;
        setIsConnected(false);
        reconnectTimer.current = setTimeout(connect, RECONNECT_MS);
      };

      socket.onerror = () => {
        socket.close();
      };
    } catch {
      reconnectTimer.current = setTimeout(connect, RECONNECT_MS);
    }
  }, [url]);

  useEffect(() => {
    isMounted.current = true;
    connect();

    return () => {
      isMounted.current = false;
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendMessage = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const getByType = useCallback((type) =>
    messages.filter(m => m.type === type),
  [messages]);

  const clearMessages = useCallback(() => setMessages([]), []);

  return { messages, isConnected, lastMessage, sendMessage, getByType, clearMessages };
}
