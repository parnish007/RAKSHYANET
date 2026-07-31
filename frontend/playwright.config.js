import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  workers: 1,
  // A hosted Gemma extraction takes ~13s and a full pipeline run chains several
  // model calls. The old 45s budget assumed the deterministic fallback answered.
  timeout: 150_000,
  use: {
    baseURL: 'http://127.0.0.1:5173',
    browserName: 'chromium',
    channel: 'chrome',
    headless: true,
    launchOptions: {
      executablePath: 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
      args: ['--enable-unsafe-swiftshader', '--enable-webgl'],
    },
  },
  projects: [{
    name: 'desktop',
    use: { viewport: { width: 1600, height: 1000 } },
  }],
});
