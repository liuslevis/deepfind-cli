import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const electronDir = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(electronDir, "..");
const repositoryRoot = path.resolve(webDir, "..");
const osName = process.platform === "win32" ? "win" : process.platform === "darwin" ? "mac" : "linux";
const targetDir = path.join(webDir, "backend-dist", `${osName}-${process.arch}`);
const workDir = path.join(webDir, ".electron-build", "pyinstaller");
const entrypoint = path.join(repositoryRoot, "desktop", "backend.py");
const uv = process.platform === "win32" ? "uv.exe" : "uv";

const result = spawnSync(
  uv,
  [
    "run",
    "--project",
    repositoryRoot,
    "--extra",
    "desktop",
    "pyinstaller",
    "--noconfirm",
    "--clean",
    "--onedir",
    "--name",
    "deepfind-backend",
    "--distpath",
    targetDir,
    "--workpath",
    workDir,
    "--specpath",
    path.join(webDir, ".electron-build"),
    "--paths",
    repositoryRoot,
    "--collect-submodules",
    "deepfind",
    entrypoint,
  ],
  {
    cwd: repositoryRoot,
    stdio: "inherit",
  },
);

if (result.error) {
  throw result.error;
}
process.exit(result.status ?? 1);
