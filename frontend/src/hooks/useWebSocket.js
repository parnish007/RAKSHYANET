import { useEffect, useState, useRef, useCallback } from 'react';

// A serverless deployment has no WebSocket endpoint to point at, so
// frontend/.env.production deliberately sets no VITE_WS_URL. Falling back to
// localhost there would have an https page open a ws:// connection the browser
// blocks as mixed content: three doomed attempts and twenty-one seconds of
// backoff to reach a conclusion already known at build time. An absent URL in a
// production build IS the conclusion, so it is reported immediately.
const WS_URL = import.meta.env.VITE_WS_URL
  || (import.meta.env.PROD ? null : 'ws://localhost:8000/ws');
const RECONNECT_BASE_MS = 3000;
const MAX_RECONNECT_ATTEMPTS = 3;
const MAX_HISTORY = 100;

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
 *   transport       — connecting, live, or unavailable
 */
export function useWebSocket(url = WS_URL) {
  const [messages,    setMessages]    = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const [transport,   setTransport]   = useState('connecting');

  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const reconnectAttempts = useRef(0);
  const isMounted = useRef(true);

  const connect = useCallback(() => {
    // Nothing configured to connect to. That is a settled answer, not a
    // failure, so it resolves at once instead of through the retry ladder.
    if (!url) {
      setTransport('unavailable');
      return;
    }

    // Don't open a second connection
    if (wsRef.current?.readyState === WebSocket.OPEN
        || wsRef.current?.readyState === WebSocket.CONNECTING) return;

    const retry = () => {
      if (!isMounted.current) return;
      if (reconnectAttempts.current >= MAX_RECONNECT_ATTEMPTS) {
        setTransport('unavailable');
        return;
      }

      const delay = RECONNECT_BASE_MS * (2 ** reconnectAttempts.current);
      reconnectAttempts.current += 1;
      setTransport('connecting');
      reconnectTimer.current = setTimeout(connect, delay);
    };

    try {
      const socket = new WebSocket(url);
      wsRef.current = socket;
      setTransport('connecting');

      socket.onopen = () => {
        if (!isMounted.current || wsRef.current !== socket) return;
        reconnectAttempts.current = 0;
        setIsConnected(true);
        setTransport('live');
      };

      socket.onmessage = (event) => {
        if (!isMounted.current || wsRef.current !== socket) return;
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
        if (!isMounted.current || wsRef.current !== socket) return;
        wsRef.current = null;
        setIsConnected(false);
        retry();
      };

      socket.onerror = () => {
        socket.close();
      };
    } catch {
      wsRef.current = null;
      setIsConnected(false);
      retry();
    }
  }, [url]);

  useEffect(() => {
    isMounted.current = true;
    reconnectAttempts.current = 0;
    setIsConnected(false);
    setTransport('connecting');
    connect();

    return () => {
      isMounted.current = false;
      clearTimeout(reconnectTimer.current);
      const socket = wsRef.current;
      wsRef.current = null;
      socket?.close();
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

  return {
    messages,
    isConnected,
    lastMessage,
    sendMessage,
    getByType,
    clearMessages,
    transport,
  };
}
