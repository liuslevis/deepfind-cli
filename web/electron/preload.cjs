const { contextBridge, ipcRenderer } = require("electron");

const tokenArgument = process.argv.find((argument) => argument.startsWith("--deepfind-api-token="));
const apiToken = tokenArgument ? tokenArgument.slice("--deepfind-api-token=".length) : "";

contextBridge.exposeInMainWorld("deepfindDesktop", {
  apiToken,
  zoom: (direction) => {
    if (direction === "in" || direction === "out") {
      ipcRenderer.send("deepfind:zoom", direction);
    }
  },
});
