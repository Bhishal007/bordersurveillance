import { buildModule } from "@nomicfoundation/hardhat-ignition/modules";

export default buildModule("IncidentAuditModule", (m) => {
  const audit = m.contract("IncidentAudit");
  return { audit };
});