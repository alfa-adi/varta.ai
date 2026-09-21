import puppeteer from 'puppeteer';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const outDir = path.join(__dirname, 'screenshots');

if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir);
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function run() {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  
  // Mobile viewport iPhone 13
  await page.setViewport({ width: 390, height: 844 });
  
  const baseUrl = 'http://localhost:5173';

  console.log('1. Dashboard with bottom nav');
  await page.goto(baseUrl, { waitUntil: 'networkidle0' });
  await sleep(1000);
  await page.screenshot({ path: path.join(outDir, 'screenshot_1_dashboard.png') });

  console.log('2. FAB Action Sheet');
  await page.click('button[aria-label="Action Menu"]'); // The central FAB
  await sleep(500);
  await page.screenshot({ path: path.join(outDir, 'screenshot_2_fab_action.png') });
  
  // close sheet by clicking outside/close
  await page.keyboard.press('Escape');
  await sleep(500);

  console.log('3. More Sheet');
  await page.click('button[aria-label="More"]'); // The more tab
  await sleep(500);
  await page.screenshot({ path: path.join(outDir, 'screenshot_3_more_sheet.png') });
  
  // click 'Patients' to close and navigate
  await page.keyboard.press('Escape');
  await sleep(500);
  
  console.log('4. Patients Directory with filter sheet');
  await page.click('button[aria-label="Patients"]');
  await sleep(1000);
  // open filter sheet (assumes button has text 'Filters' or similar icon)
  // we'll try to find the filter button
  const filterBtn = await page.$('button[title="Filters"]');
  if (filterBtn) {
    await filterBtn.click();
  } else {
    // fallback, try to click the filter icon by its typical position or class
    // In DirectoryToolbar it is the last button in the row of search
    const buttons = await page.$$('button');
    for (const b of buttons) {
      const className = await page.evaluate(el => el.className, b);
      if (className.includes('bg-surface-card')) {
        await b.click();
        break;
      }
    }
  }
  await sleep(500);
  await page.screenshot({ path: path.join(outDir, 'screenshot_5_patients_dir.png') });
  await page.keyboard.press('Escape');
  await sleep(500);

  console.log('5. Patient Profile');
  // click the first patient card
  const patientCard = await page.$('.grid > div');
  if (patientCard) {
    await patientCard.click();
    await sleep(1000);
    await page.screenshot({ path: path.join(outDir, 'screenshot_4_patient_profile.png') });
  }

  console.log('6. Calendar appointment action sheet');
  await page.click('button[aria-label="Calendar"]');
  await sleep(1000);
  const aptCard = await page.$('.absolute.right-8'); // the appointment block
  if (aptCard) {
    await aptCard.click();
    await sleep(500);
    await page.screenshot({ path: path.join(outDir, 'screenshot_6_calendar_action.png') });
    await page.keyboard.press('Escape');
    await sleep(500);
  }

  console.log('7. Global Search');
  // The search icon is in TopBar
  await page.goto(baseUrl, { waitUntil: 'networkidle0' });
  await sleep(1000);
  const searchBtn = await page.$('button[aria-label="Global Search"]');
  if (searchBtn) {
    await searchBtn.click();
    await sleep(500);
    await page.screenshot({ path: path.join(outDir, 'screenshot_7_global_search.png') });
    await page.keyboard.press('Escape');
    await sleep(500);
  }

  console.log('8. New Patient full-screen form');
  await page.click('button[aria-label="Action Menu"]');
  await sleep(500);
  // find "Add Patient" in action sheet
  const textBtns = await page.$$('button');
  for (const b of textBtns) {
    const text = await page.evaluate(el => el.textContent, b);
    if (text?.includes('Add Patient')) {
      await b.click();
      break;
    }
  }
  await sleep(500);
  await page.screenshot({ path: path.join(outDir, 'screenshot_8_new_patient.png') });
  await page.keyboard.press('Escape');
  await sleep(500);

  console.log('9. Settings section deep-link');
  await page.goto(baseUrl + '/settings/profile', { waitUntil: 'networkidle0' });
  await sleep(1000);
  await page.screenshot({ path: path.join(outDir, 'screenshot_9_settings_detail.png') });

  console.log('10. Reports detail deep-link');
  await page.goto(baseUrl + '/reports/1', { waitUntil: 'networkidle0' });
  await sleep(1000);
  await page.screenshot({ path: path.join(outDir, 'screenshot_10_reports_detail.png') });

  await browser.close();
  console.log('Done generating screenshots.');
}

run().catch(console.error);
