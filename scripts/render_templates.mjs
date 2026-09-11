import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const jobs = [
  ["附件/附件5/result1.xlsx", "计划购电量", "A1:B20", "result1_plan"],
  ["附件/附件5/result1.xlsx", "充放电量", "A1:E7", "result1_storage"],
  ["附件/附件5/result3.xlsx", "计划购电量", "A1:L8", "result3_plan_left"],
  ["附件/附件5/result3.xlsx", "计划购电量", "EN1:EQ8", "result3_plan_right"],
  ["附件/附件5/result3.xlsx", "调整购电量", "A1:L8", "result3_adjust_left"],
  ["附件/附件5/result3.xlsx", "充放电量", "A1:F26", "result3_storage"],
  ["附件/附件5/result3.xlsx", "紧急购电量", "A1:C11", "result3_emergency"],
];

const outDir = ".tmp/template_renders";
await fs.mkdir(outDir, { recursive: true });

const cache = new Map();
for (const [file, sheetName, range, label] of jobs) {
  let workbook = cache.get(file);
  if (!workbook) {
    workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(file));
    cache.set(file, workbook);
  }
  const preview = await workbook.render({ sheetName, range, scale: 2, format: "png" });
  const outputPath = path.join(outDir, `${label}.png`);
  await fs.writeFile(outputPath, new Uint8Array(await preview.arrayBuffer()));
  console.log(outputPath);
}
