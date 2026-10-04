const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const http = require('http'); 
const { Server } = require('socket.io'); 
require('dotenv').config();

const app = express();
const server = http.createServer(app); 

const io = new Server(server, {
    cors: { origin: "*" } 
});

app.use(cors());
app.use(express.json());

app.use((req, res, next) => {
    req.io = io;
    next();
});

// Import and use routes
const alertRoutes = require('./routes/alertRoutes');
app.use('/api/v1/alerts', alertRoutes);

// NEW: Omni-Agent Chat Proxy Route
// This routes questions from the React dashboard to the Python LangChain server
app.post('/api/v1/agent/chat', async (req, res) => {
    try {
        const response = await fetch('http://host.docker.internal:8000/api/agent/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req.body)
        });
        
        if (!response.ok) throw new Error(`Python server error: ${response.status}`);
        
        const data = await response.json();
        res.status(200).json(data);
    } catch (error) {
        console.error("Agent proxy error:", error);
        res.status(502).json({ 
            reply: "[CONNECTION ERROR]: Omni-Agent offline. Ensure Python FastAPI is running on port 8000." 
        });
    }
});

mongoose.connect(process.env.MONGO_URI)
    .then(() => console.log('✅ Connected to MongoDB successfully!'))
    .catch((err) => console.log('❌ MongoDB connection error:', err));

io.on('connection', (socket) => {
    console.log('⚡ Target terminal connected via WebSockets');
});

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
    console.log(`🚀 Server and WebSockets running on port ${PORT}`);
});