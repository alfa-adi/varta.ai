// @ts-check
import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for Varta.ai browser streaming validation.
 *
 * Uses the deterministic stub server (tests/e2e/stub_server.py) as the backend.
 * Chromium is launched with fake media device flags so getUserMedia returns
 * a controllable audio stream without requiring a physical microphone.
 */
export default defineConfig({
  testDir: './tests',
  /* Run tests sequentially — real browser WebSocket lifecycle is order-sensitive */
  fullyParallel: false,
  /* Fail the build on CI if you accidentally left test.only in the source */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 1 : 0,
  /* One worker — serial execution for lifecycle correctness */
  workers: 1,
  /* Reporter */
  reporter: [
    ['list'],
    ['json', { outputFile: '../test/results/browser-streaming/results.json' }],
  ],
  /* Global timeout per test */
  timeout: 60_000,
  /* Shared settings for all projects */
  use: {
    /* Base URL for the stub server */
    baseURL: 'http://127.0.0.1:8000',
    /* Capture trace on first retry */
    trace: 'on-first-retry',
    /* Capture video for every test */
    video: 'on',
    /* Capture screenshot on failure */
    screenshot: 'only-on-failure',
  },

  projects: [
    {
      name: 'chromium-fake-mic',
      use: {
        ...devices['Desktop Chrome'],
        /* Grant microphone permission for the app origin */
        permissions: ['microphone'],
        /* Chromium flags for deterministic fake audio device */
        launchOptions: {
          args: [
            '--use-fake-ui-for-media-stream',
            '--use-fake-device-for-media-stream',
            '--autoplay-policy=no-user-gesture-required',
          ],
        },
      },
    },
  ],

  /* Start the deterministic stub server before running tests */
  webServer: {
    command: process.platform === 'win32'
      ? 'venv\\Scripts\\python.exe -m tests.e2e.stub_server --port 8000'
      : 'venv/bin/python -m tests.e2e.stub_server --port 8000',
    port: 8000,
    cwd: '..',
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
