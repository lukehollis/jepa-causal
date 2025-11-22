import { useState, useRef } from 'react'
import './App.css'
import { AttentionViz } from './Visualizer'

function App() {
  const [connected, setConnected] = useState(false)
  const [attnData, setAttnData] = useState<number[][][] | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  const connect = () => {
    const ws = new WebSocket('ws://localhost:8000/ws/inference')
    
    ws.onopen = () => {
      console.log('Connected')
      setConnected(true)
      
      // Send config
      ws.send(JSON.stringify({
        data_dims: [3, 8, 32, 32], // Example config
        patch_size: 8,
        tubelet_size: 2,
        rep_dim: 128
      }))
    }

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data)
      if (msg.type === 'representation') {
        if (msg.attention) {
          setAttnData(msg.attention)
        }
      } else if (msg.type === 'complete') {
        console.log('Stream complete')
      }
    }

    ws.onclose = () => {
      console.log('Disconnected')
      setConnected(false)
    }

    wsRef.current = ws
  }

  return (
    <div className="app-container">
        <button type="button" onClick={connect} disabled={connected} style={{ position: "fixed", top: "10px", right: "10px", zIndex: 2, backgroundColor: "#000", color: "#fff", border: "none", padding: "10px 20px", cursor: "pointer", borderRadius: "5px", boxShadow: "0 2px 5px rgba(0, 0, 0, 0.2)"}}>
          {connected ? 'Streaming...' : 'Connect & Run Inference'}
        </button>
      
        <AttentionViz data={attnData} />
    </div>
  )
}

export default App
