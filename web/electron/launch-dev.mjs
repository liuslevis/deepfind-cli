import { spawn, spawnSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import electronPath from "electron";

const electronDir = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(electronDir, "..");
const repositoryRoot = path.resolve(webDir, "..");

function stageWindowsRuntime() {
  const sourceDist = path.dirname(electronPath);
  const version = readFileSync(path.join(sourceDist, "version"), "utf8").trim();
  const stageRoot = path.join(process.env.LOCALAPPDATA ?? os.tmpdir(), "DeepFind", "electron-dev");
  const runtimeDir = path.join(stageRoot, `runtime-${version}`);
  const appDir = path.join(stageRoot, "app");
  const stagedExecutable = path.join(runtimeDir, "electron.exe");

  mkdirSync(stageRoot, { recursive: true });
  if (!existsSync(stagedExecutable)) {
    cpSync(sourceDist, runtimeDir, { recursive: true });
  }

  mkdirSync(appDir, { recursive: true });
  cpSync(path.join(electronDir, "main.mjs"), path.join(appDir, "main.mjs"));
  cpSync(path.join(electronDir, "preload.cjs"), path.join(appDir, "preload.cjs"));
  writeFileSync(
    path.join(appDir, "package.json"),
    JSON.stringify(
      {
        name: "deepfind-electron-dev",
        productName: "DeepFind",
        version: "0.1.0",
        type: "module",
        main: "main.mjs",
      },
      null,
      2,
    ),
  );

  const acl = spawnSync(
    "icacls",
    [stageRoot, "/grant", "*S-1-15-2-2:(OI)(CI)(RX)", "/T", "/C", "/Q"],
    { stdio: "ignore", windowsHide: true },
  );
  if (acl.status !== 0) {
    throw new Error(`Failed to grant Electron AppContainer access to ${stageRoot}.`);
  }

  return { executable: stagedExecutable, appDir };
}

const runtime = process.platform === "win32"
  ? stageWindowsRuntime()
  : { executable: electronPath, appDir: webDir };
const child = spawn(runtime.executable, [runtime.appDir], {
  env: {
    ...process.env,
    DEEPFIND_RENDERER_URL: "http://127.0.0.1:5173",
    DEEPFIND_REPOSITORY_ROOT: repositoryRoot,
  },
  stdio: "inherit",
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}

child.on("error", (error) => {
  console.error(error);
  process.exit(1);
});
child.on("exit", (code) => process.exit(code ?? 1));
