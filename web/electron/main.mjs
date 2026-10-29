import { randomBytes } from "node:crypto";
import { spawn } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { app, BrowserWindow, dialog, ipcMain, Menu, shell } from "electron";

const electronDir = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = process.env.DEEPFIND_REPOSITORY_ROOT
  ? path.resolve(process.env.DEEPFIND_REPOSITORY_ROOT)
  : path.resolve(electronDir, "..", "..");
const rendererUrl = process.env.DEEPFIND_RENDERER_URL ?? (!app.isPackaged ? "http://127.0.0.1:5173" : undefined);
const apiToken = randomBytes(32).toString("hex");

let backendProcess;
let mainWindow;
let backendOrigin;
let quitting = false;
let applicationMenu;
let menuBarVisible = true;

const ZOOM_STEP = 0.5;
const MIN_ZOOM_LEVEL = -5;
const MAX_ZOOM_LEVEL = 5;

function ignoreBrokenPipe(stream) {
  stream.on("error", (error) => {
    if (error.code !== "EPIPE") {
      throw error;
    }
  });
}

ignoreBrokenPipe(process.stdout);
ignoreBrokenPipe(process.stderr);

function reservePort(preferredPort) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(preferredPort ?? 0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : preferredPort;
      server.close(() => resolve(port));
    });
  });
}

function developmentPython() {
  const executable = process.platform === "win32"
    ? path.join(repositoryRoot, ".venv", "Scripts", "python.exe")
    : path.join(repositoryRoot, ".venv", "bin", "python");

  if (existsSync(executable)) {
    return { command: executable, args: ["-m", "deepfind.web_api"] };
  }
  return {
    command: process.platform === "win32" ? "uv.exe" : "uv",
    args: ["run", "--project", repositoryRoot, "python", "-m", "deepfind.web_api"],
  };
}

function packagedBackend() {
  const executable = process.platform === "win32"
    ? path.join(process.resourcesPath, "backend", "deepfind-backend.exe")
    : path.join(process.resourcesPath, "backend", "deepfind-backend");
  return { command: executable, args: [] };
}

function startBackend(port) {
  const backend = app.isPackaged ? packagedBackend() : developmentPython();
  const userData = app.getPath("userData");
  mkdirSync(userData, { recursive: true });
  const env = {
    ...process.env,
    DEEPFIND_ENV_FILE: path.join(app.isPackaged ? userData : repositoryRoot, ".env"),
    DEEPFIND_PROJECT_DIR: app.isPackaged ? userData : repositoryRoot,
    DEEPFIND_WEB_DIST_DIR: app.isPackaged
      ? path.join(process.resourcesPath, "web-dist")
      : path.join(repositoryRoot, "web", "dist"),
    DEEPFIND_WEB_TOKEN: apiToken,
  };

  backendProcess = spawn(
    backend.command,
    [...backend.args, "--host", "127.0.0.1", "--port", String(port)],
    {
      cwd: app.isPackaged ? userData : repositoryRoot,
      env,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    },
  );

  backendProcess.stdout.on("data", (chunk) => console.log(`[backend] ${chunk.toString().trimEnd()}`));
  backendProcess.stderr.on("data", (chunk) => console.error(`[backend] ${chunk.toString().trimEnd()}`));
  backendProcess.on("error", (error) => console.error("Failed to start DeepFind backend", error));
}

async function waitForBackend(origin) {
  const deadline = Date.now() + 60_000;
  let lastError;

  while (Date.now() < deadline) {
    if (backendProcess?.exitCode !== null) {
      throw new Error(`DeepFind backend exited with code ${backendProcess.exitCode}.`);
    }
    try {
      const response = await fetch(`${origin}/api/health`);
      if (response.ok) {
        return;
      }
      lastError = new Error(`Health check returned ${response.status}.`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
  }

  throw new Error(`Timed out waiting for the DeepFind backend: ${lastError?.message ?? "unknown error"}`);
}

function isSafeExternalUrl(url) {
  try {
    const protocol = new URL(url).protocol;
    return protocol === "https:" || protocol === "http:";
  } catch {
    return false;
  }
}

function toggleMenuBar(window) {
  if (process.platform === "darwin") {
    // macOS always shows the menu bar, so swap out the application menu instead.
    applicationMenu ??= Menu.getApplicationMenu();
    menuBarVisible = !menuBarVisible;
    Menu.setApplicationMenu(menuBarVisible ? applicationMenu : Menu.buildFromTemplate([]));
    return;
  }

  window.setMenuBarVisibility(!window.isMenuBarVisible());
}

function registerMenuBarShortcut(window) {
  window.webContents.on("before-input-event", (event, input) => {
    if (input.type !== "keyDown" || input.key !== "F12") {
      return;
    }
    event.preventDefault();
    if (input.isAutoRepeat || input.control || input.alt || input.meta || input.shift) {
      return;
    }
    toggleMenuBar(window);
  });
}

ipcMain.on("deepfind:zoom", (event, direction) => {
  if ((direction !== "in" && direction !== "out") || event.sender.isDestroyed()) {
    return;
  }
  const current = event.sender.getZoomLevel();
  const next = direction === "in" ? current + ZOOM_STEP : current - ZOOM_STEP;
  event.sender.setZoomLevel(Math.min(MAX_ZOOM_LEVEL, Math.max(MIN_ZOOM_LEVEL, next)));
});

async function createWindow() {
  if (!backendProcess || backendProcess.exitCode !== null || !backendOrigin) {
    const port = await reservePort(rendererUrl ? 8000 : undefined);
    backendOrigin = `http://127.0.0.1:${port}`;
    startBackend(port);
    await waitForBackend(backendOrigin);
  }

  mainWindow = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 900,
    minHeight: 640,
    show: false,
    backgroundColor: "#0b0f14",
    webPreferences: {
      preload: path.join(electronDir, "preload.cjs"),
      additionalArguments: [`--deepfind-api-token=${apiToken}`],
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  // Each new window starts with the menu bar showing again.
  menuBarVisible = true;
  if (process.platform === "darwin" && applicationMenu) {
    Menu.setApplicationMenu(applicationMenu);
  }
  registerMenuBarShortcut(mainWindow);

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (isSafeExternalUrl(url)) {
      void shell.openExternal(url);
    }
    return { action: "deny" };
  });
  mainWindow.webContents.on("will-navigate", (event, url) => {
    const allowedOrigin = new URL(rendererUrl ?? backendOrigin).origin;
    if (new URL(url).origin !== allowedOrigin) {
      event.preventDefault();
      if (isSafeExternalUrl(url)) {
        void shell.openExternal(url);
      }
    }
  });
  mainWindow.once("ready-to-show", () => mainWindow?.show());
  await mainWindow.loadURL(rendererUrl ?? backendOrigin);
}

function stopBackend() {
  if (backendProcess && backendProcess.exitCode === null) {
    backendProcess.kill();
  }
  backendProcess = undefined;
}

app.on("before-quit", () => {
  quitting = true;
  stopBackend();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (!quitting && BrowserWindow.getAllWindows().length === 0) {
    void createWindow().catch(handleStartupError);
  }
});

function handleStartupError(error) {
  console.error(error);
  dialog.showErrorBox("DeepFind failed to start", error instanceof Error ? error.message : String(error));
  app.quit();
}

app.whenReady().then(createWindow).catch(handleStartupError);
