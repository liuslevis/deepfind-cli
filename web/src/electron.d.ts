export {};

declare global {
  interface Window {
    deepfindDesktop?: {
      apiToken: string;
    };
  }
}
