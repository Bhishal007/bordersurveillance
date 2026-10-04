const ethers = require('ethers');
const Alert = require('../models/Alert');

// The minimal ABI (Application Binary Interface) needed to talk to your specific contract
const contractABI = [
    "function logIncident(string memory _alertId, bytes32 _dataHash, string memory _threatLevel) public"
];

async function logAlertToBlockchain(alert, io) {
    try {
        console.log(`⛓️ Submitting alert ${alert._id} fingerprint to blockchain...`);

        // 1. Connect to the local Hardhat node using your .env variables
        const provider = new ethers.JsonRpcProvider(process.env.BLOCKCHAIN_RPC_URL);
        const wallet = new ethers.Wallet(process.env.PRIVATE_KEY, provider);
        
        // 2. Connect to the deployed smart contract
        const contract = new ethers.Contract(process.env.CONTRACT_ADDRESS, contractABI, wallet);

        // 3. Create a cryptographic SHA-256 hash of the alert data
        const dataString = JSON.stringify({
            cameraId: alert.cameraId,
            eventType: alert.eventType,
            threatLevel: alert.threatLevel,
            objectClass: alert.objectClass,
            confidence: alert.confidence
        });
        const dataHash = ethers.keccak256(ethers.toUtf8Bytes(dataString));

        // 4. Execute the smart contract function to log the data permanently
        const tx = await contract.logIncident(
            alert._id.toString(), 
            dataHash, 
            alert.threatLevel
        );
        
        // 5. Wait for the Hardhat node to mine the block
        const receipt = await tx.wait();
        console.log(`✅ Blockchain transaction confirmed: ${receipt.hash}`);

        // 6. Update the MongoDB document with the blockchain evidence
        const updatedAlert = await Alert.findByIdAndUpdate(
            alert._id,
            {
                $set: {
                    "blockchain.verified": true,
                    "blockchain.txHash": receipt.hash,
                    "blockchain.blockNumber": receipt.blockNumber,
                    "blockchain.dataHash": dataHash
                }
            },
          { returnDocument: 'after' }
        );

        // 7. Instantly push the verified update to the React frontend via WebSockets
        if (io) {
            io.emit('ALERT_VERIFIED', updatedAlert);
        }

    } catch (error) {
        console.error("❌ Blockchain logging failed:", error);
    }
}

module.exports = { logAlertToBlockchain };