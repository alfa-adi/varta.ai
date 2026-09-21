mkdir puppeteer_temp -ErrorAction SilentlyContinue
cd puppeteer_temp
npm init -y
npm install puppeteer
node ../scripts/take_screenshots.js
