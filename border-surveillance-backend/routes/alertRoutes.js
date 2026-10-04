const express = require('express');
const router = express.Router();
const Alert = require('../models/Alert');
const { logAlertToBlockchain } = require('../services/blockchainService');

// POST: Receive a new alert from the Python AI
router.post('/', async (req, res) => {
    try {
        const newAlert = new Alert(req.body);
        const savedAlert = await newAlert.save();
        
        // Broadcast initial unverified alert via WebSockets instantly
        req.io.emit('NEW_ALERT', savedAlert);

        // If threat level is high or critical, anchor it to the blockchain asynchronously
        if (savedAlert.threatLevel === 'CRITICAL' || savedAlert.threatLevel === 'HIGH') {
            logAlertToBlockchain(savedAlert, req.io); 
        }
        
        res.status(201).json(savedAlert);
    } catch (error) {
        res.status(400).json({ message: "Failed to save alert", error: error.message });
    }
});

// GET: Fetch the latest alerts for the React dashboard
router.get('/', async (req, res) => {
    try {
        const alerts = await Alert.find().sort({ createdAt: -1 }).limit(50);
        res.status(200).json(alerts);
    } catch (error) {
        res.status(500).json({ message: "Server error", error: error.message });
    }
});

// PUT: Acknowledge/Resolve an alert
router.put('/:id/acknowledge', async (req, res) => {
    try {
        const updatedAlert = await Alert.findByIdAndUpdate(
            req.params.id, 
            { threatLevel: 'RESOLVED' }, // Downgrades the threat
            { new: true }
        );
        
        if (!updatedAlert) {
            return res.status(404).json({ message: "Alert not found" });
        }
        
        res.status(200).json(updatedAlert);
    } catch (error) {
        console.error("Error updating alert:", error);
        res.status(500).json({ error: "Server Error" });
    }
});

module.exports = router;