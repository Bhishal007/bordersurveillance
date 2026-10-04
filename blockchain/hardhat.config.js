import { defineConfig } from "hardhat/config";
import hardhatToolboxMochaEthers from "@nomicfoundation/hardhat-toolbox-mocha-ethers";

export default defineConfig({
  solidity: "0.8.24",
  plugins: [hardhatToolboxMochaEthers]
});