import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const payload = JSON.parse(await fs.readFile("outputs/workbook_payload.json", "utf8"));
const templateDir = "附件/附件5";
const outputDir = "outputs";
const previewDir = ".tmp/result_previews";
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

function excelColumn(number) {
  let value = number;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function asDate(value) {
  if (!value) return null;
  const [year, month, day] = value.split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, day));
}

function convertDateColumn(rows) {
  return rows.map((row) => [asDate(row[0]), ...row.slice(1)]);
}

async function tileRows(sheet, sourceRange, firstRow, rowCount, blockHeight, width) {
  const lastColumn = excelColumn(width);
  for (let offset = 0; offset < rowCount; offset += blockHeight) {
    const height = Math.min(blockHeight, rowCount - offset);
    const source = height === blockHeight
      ? sheet.getRange(sourceRange)
      : sheet.getRange(sourceRange).resize(height, width);
    const destination = sheet.getRange(
      `A${firstRow + offset}:${lastColumn}${firstRow + offset + height - 1}`,
    );
    destination.copyFrom(source, "all");
  }
}

async function writePlanSheet(sheet, rows) {
  const converted = convertDateColumn(rows);
  const lastColumn = excelColumn(converted[0].length);
  sheet.getRange(`A2:${lastColumn}${converted.length + 1}`).values = converted;
  sheet.getRange(`A2:A${converted.length + 1}`).setNumberFormat("yyyy-mm-dd");
  sheet.getRange(`B2:${lastColumn}${converted.length + 1}`).setNumberFormat("0.000");
}

async function writeStorageSheet(sheet, rows) {
  await tileRows(sheet, "A2:F7", 2, rows.length, 6, 6);
  const converted = convertDateColumn(rows);
  sheet.getRange(`A2:F${rows.length + 1}`).values = converted;
  sheet.getRange(`A2:A${rows.length + 1}`).setNumberFormat("yyyy-mm-dd");
  sheet.getRange(`C2:D${rows.length + 1}`).setNumberFormat("0.000");
  sheet.getRange(`F2:F${rows.length + 1}`).setNumberFormat("0.000");
}

async function writeEmergencySheet(sheet, rows) {
  await tileRows(sheet, "A2:C4", 2, rows.length, 3, 3);
  const converted = convertDateColumn(rows);
  sheet.getRange(`A2:C${rows.length + 1}`).values = converted;
  sheet.getRange(`A2:A${rows.length + 1}`).setNumberFormat("yyyy-mm-dd");
  sheet.getRange(`C2:C${rows.length + 1}`).setNumberFormat("0.000");
}

async function verifyAndExport(workbook, name, previewSpecs) {
  workbook.recalculate();
  const inspect = await workbook.inspect({
    kind: "workbook,sheet,table",
    maxChars: 5000,
    tableMaxRows: 4,
    tableMaxCols: 8,
    tableMaxCellChars: 60,
  });
  console.log(`VERIFY ${name}\n${inspect.ndjson}`);
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 30 },
    summary: `${name} formula error scan`,
  });
  console.log(`ERRORS ${name}\n${errors.ndjson}`);
  for (const [sheetName, range, label] of previewSpecs) {
    const preview = await workbook.render({ sheetName, range, scale: 1.5, format: "png" });
    await fs.writeFile(
      path.join(previewDir, `${name}_${label}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
  const output = await SpreadsheetFile.exportXlsx(workbook);
  const outputPath = path.join(outputDir, name);
  await output.save(outputPath);
  const reopened = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
  const savedCheck = await reopened.inspect({
    kind: "sheet",
    include: "id,name",
    maxChars: 3000,
  });
  console.log(`SAVED ${name}\n${savedCheck.ndjson}`);
}

async function buildResult1() {
  const name = "result1.xlsx";
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(`${templateDir}/${name}`));
  const plan = workbook.worksheets.getItem("计划购电量");
  plan.getRange("B2:B145").values = payload.result1.plan.map((value) => [value]);
  plan.getRange("B2:B145").setNumberFormat("0.000");
  const storage = workbook.worksheets.getItem("充放电量");
  storage.getRange("B2:C7").values = payload.result1.storage.map((row) => row.slice(1));
  storage.getRange("E2:E3").values = [[payload.result1.soc0], [payload.result1.soc24]];
  storage.getRange("B2:C7").setNumberFormat("0.000");
  storage.getRange("E2:E3").setNumberFormat("0.000");
  await verifyAndExport(workbook, name, [
    ["计划购电量", "A1:B20", "plan_top"],
    ["计划购电量", "A135:B145", "plan_bottom"],
    ["充放电量", "A1:E7", "storage"],
  ]);
}

async function buildYearWorkbook(name, data, hasAdjustment) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(`${templateDir}/${name}`));
  await writePlanSheet(workbook.worksheets.getItem("计划购电量"), data.plan_rows);
  if (hasAdjustment) {
    await writePlanSheet(workbook.worksheets.getItem("调整购电量"), data.adjust_rows);
  }
  await writeStorageSheet(workbook.worksheets.getItem("充放电量"), data.storage_rows);
  await writeEmergencySheet(workbook.worksheets.getItem("紧急购电量"), data.emergency_rows);
  const storageLast = data.storage_rows.length + 1;
  const emergencyLast = data.emergency_rows.length + 1;
  const specs = [
    ["计划购电量", "A1:L8", "plan_left"],
    ["计划购电量", "EN1:EQ8", "plan_right"],
    ["充放电量", "A1:F14", "storage_top"],
    ["充放电量", `A${storageLast - 11}:F${storageLast}`, "storage_bottom"],
    ["紧急购电量", "A1:C12", "emergency_top"],
    ["紧急购电量", `A${Math.max(1, emergencyLast - 10)}:C${emergencyLast}`, "emergency_bottom"],
  ];
  if (hasAdjustment) {
    specs.push(["调整购电量", "A1:L8", "adjust_left"]);
    specs.push(["调整购电量", "EN1:EQ8", "adjust_right"]);
  }
  await verifyAndExport(workbook, name, specs);
}

const target = process.argv[2] ?? "all";
if (target === "all") {
  await buildResult1();
  await buildYearWorkbook("result2.xlsx", payload.result2, false);
  await buildYearWorkbook("result3.xlsx", payload.result3, true);
  await buildYearWorkbook("result4-2.xlsx", payload["result4-2"], false);
  await buildYearWorkbook("result4-3.xlsx", payload["result4-3"], true);
} else if (target === "result2") {
  await buildYearWorkbook("result2.xlsx", payload.result2, false);
} else {
  throw new Error(`Unknown workbook target: ${target}`);
}
