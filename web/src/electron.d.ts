export {};

declare global {
  interface Window {
    deepfindDesktop?: {
      apiToken: string;
      zoom: (direction: "in" | "out") => void;
    };
  }
}
