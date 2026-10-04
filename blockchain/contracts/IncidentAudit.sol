// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract IncidentAudit {
    struct Incident {
        string alertId;
        bytes32 dataHash;
        uint256 timestamp;
        string threatLevel;
        address recordedBy;
    }

    mapping(string => Incident) public incidents;
    event IncidentLogged(string indexed alertId, bytes32 dataHash, uint256 timestamp);

    function logIncident(string memory _alertId, bytes32 _dataHash, string memory _threatLevel) public {
        incidents[_alertId] = Incident(_alertId, _dataHash, block.timestamp, _threatLevel, msg.sender);
        emit IncidentLogged(_alertId, _dataHash, block.timestamp);
    }
}