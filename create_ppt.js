const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "AI Eval Platform";
pres.title = "人工智能医疗器械设计开发要点";

// ============================================================
// Color palette
// ============================================================
const C = {
  darkBg:    "0F172A",
  darkBg2:   "1E293B",
  primary:   "2563EB",
  primaryL:  "3B82F6",
  secondary: "0D9488",
  accent:    "7C3AED",
  lightBg:   "F8FAFC",
  cardBg:    "FFFFFF",
  textDark:  "1E293B",
  textBody:  "334155",
  textMuted: "64748B",
  border:    "E2E8F0",
  highlight: "DBEAFE",
  green:     "059669",
  amber:     "D97706",
  red:       "DC2626",
  white:     "FFFFFF",
};

const FONT_TITLE = "Arial";
const FONT_BODY  = "Calibri";

// ============================================================
// Helper: add header bar to content slides
// ============================================================
function addHeader(slide, title, sec) {
  // Top bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.primary },
  });
  // Section tag
  slide.addText(`第${sec}章`, {
    x: 0.5, y: 0.25, w: 1.2, h: 0.35,
    fontSize: 10, fontFace: FONT_BODY, color: C.white,
    fill: { color: C.primary }, align: "center", valign: "middle",
    margin: 0,
  });
  // Title
  slide.addText(title, {
    x: 1.85, y: 0.2, w: 7.5, h: 0.45,
    fontSize: 20, fontFace: FONT_TITLE, color: C.textDark, bold: true,
    margin: 0, valign: "middle",
  });
  // Bottom line
  slide.addShape(pres.shapes.LINE, {
    x: 0.5, y: 0.72, w: 9, h: 0,
    line: { color: C.border, width: 1 },
  });
}

// ============================================================
// Helper: footer
// ============================================================
function addFooter(slide, num) {
  slide.addText(`人工智能医疗器械设计开发要点  |  ${num}`, {
    x: 0.5, y: 5.25, w: 9, h: 0.3,
    fontSize: 8, fontFace: FONT_BODY, color: C.textMuted, align: "center",
    margin: 0,
  });
}

// ============================================================
// Helper: content slide with header + footer
// ============================================================
function contentSlide(sec, title, num) {
  const slide = pres.addSlide();
  slide.background = { color: C.lightBg };
  addHeader(slide, title, sec);
  addFooter(slide, num);
  return slide;
}

// ============================================================
// Helper: bullet row
// ============================================================
function bulletRow(slide, items, x, y, w, h, opts = {}) {
  const arr = items.map((t, i) => ({
    text: t,
    options: {
      bullet: opts.noBullet ? false : true,
      breakLine: i < items.length - 1,
      fontSize: opts.fontSize || 13,
      color: opts.color || C.textBody,
      paraSpaceAfter: 6,
    },
  }));
  slide.addText(arr, { x, y, w, h, fontFace: FONT_BODY, valign: "top" });
}

// ============================================================
// TITLE SLIDE
// ============================================================
const s1 = pres.addSlide();
s1.background = { color: C.darkBg };
// Decorative top bar
s1.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.primary },
});
// Dot pattern
for (let i = 0; i < 12; i++) {
  s1.addShape(pres.shapes.OVAL, {
    x: 0.4 + i * 0.8, y: 0.55, w: 0.08, h: 0.08,
    fill: { color: C.primaryL, transparency: 60 },
  });
}
// Main title
s1.addText("人工智能医疗器械\n设计开发要点", {
  x: 0.8, y: 1.0, w: 8.4, h: 2.0,
  fontSize: 38, fontFace: FONT_TITLE, color: C.white, bold: true,
  align: "center", valign: "middle", lineSpacingMultiple: 1.3,
});
// Subtitle
s1.addText("AI Medical Device Design & Development", {
  x: 0.8, y: 2.9, w: 8.4, h: 0.5,
  fontSize: 16, fontFace: FONT_BODY, color: C.primaryL,
  align: "center", charSpacing: 4,
});
// Divider line
s1.addShape(pres.shapes.LINE, {
  x: 3.5, y: 3.55, w: 3, h: 0,
  line: { color: C.secondary, width: 2 },
});
// Bottom info
s1.addText("基于 NMPA《人工智能医疗器械注册审查指导原则》", {
  x: 0.8, y: 4.2, w: 8.4, h: 0.5,
  fontSize: 12, fontFace: FONT_BODY, color: C.textMuted, align: "center",
});

// ============================================================
// TABLE OF CONTENTS
// ============================================================
const sToc = pres.addSlide();
sToc.background = { color: C.lightBg };
addFooter(sToc, "目录");

sToc.addText("目 录", {
  x: 0.8, y: 0.35, w: 5, h: 0.6,
  fontSize: 26, fontFace: FONT_TITLE, color: C.textDark, bold: true, margin: 0,
});
sToc.addShape(pres.shapes.LINE, {
  x: 0.8, y: 1.0, w: 2.5, h: 0,
  line: { color: C.primary, width: 3 },
});

const chapters = [
  { num: "01", title: "人工智能发展和应用", items: ["人工智能整体发展", "AI在医疗器械上的发展", "国家政策与纲要时政支持"] },
  { num: "02", title: "法规和监管", items: ["中国 NMPA 监管体系", "美国 FDA 监管框架", "欧盟 MDR+AI Act 监管", "通用人工智能与医疗器械 AI 法规"] },
  { num: "03", title: "人工智能医疗器械设计开发要点", items: ["全生命周期流程", "风险治理", "AI 素养要求"] },
];

chapters.forEach((ch, i) => {
  const yBase = 1.35 + i * 1.35;
  sToc.addShape(pres.shapes.RECTANGLE, {
    x: 0.8, y: yBase, w: 8.4, h: 1.15,
    fill: { color: C.cardBg },
    shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.08 },
  });
  // Number circle
  sToc.addShape(pres.shapes.OVAL, {
    x: 1.1, y: yBase + 0.2, w: 0.7, h: 0.7,
    fill: { color: i === 0 ? C.primary : i === 1 ? C.secondary : C.accent },
  });
  sToc.addText(ch.num, {
    x: 1.1, y: yBase + 0.2, w: 0.7, h: 0.7,
    fontSize: 20, fontFace: FONT_TITLE, color: C.white, bold: true,
    align: "center", valign: "middle", margin: 0,
  });
  // Chapter title
  sToc.addText(ch.title, {
    x: 2.1, y: yBase + 0.12, w: 6.5, h: 0.4,
    fontSize: 15, fontFace: FONT_TITLE, color: C.textDark, bold: true, margin: 0, valign: "middle",
  });
  // Items
  const itemText = ch.items.map((it, j) => ({
    text: it,
    options: { bullet: true, breakLine: j < ch.items.length - 1, fontSize: 11, color: C.textMuted, paraSpaceAfter: 2 },
  }));
  sToc.addText(itemText, {
    x: 2.1, y: yBase + 0.52, w: 6.5, h: 0.55,
    fontFace: FONT_BODY, valign: "top",
  });
});

// ============================================================
// CHAPTER 1: 人工智能发展和应用
// ============================================================

// --- Chapter 1 divider ---
const ch1Div = pres.addSlide();
ch1Div.background = { color: C.darkBg };
ch1Div.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.primary },
});
ch1Div.addText("01", {
  x: 0.8, y: 1.2, w: 2, h: 1.2,
  fontSize: 60, fontFace: FONT_TITLE, color: C.primary, bold: true, margin: 0,
});
ch1Div.addText("人工智能发展和应用", {
  x: 0.8, y: 2.4, w: 8, h: 0.7,
  fontSize: 28, fontFace: FONT_TITLE, color: C.white, bold: true, margin: 0,
});
ch1Div.addShape(pres.shapes.LINE, {
  x: 0.8, y: 3.2, w: 2, h: 0, line: { color: C.secondary, width: 2.5 },
});
ch1Div.addText("AI Development · Medical Applications · National Policies", {
  x: 0.8, y: 3.4, w: 8, h: 0.5,
  fontSize: 12, fontFace: FONT_BODY, color: C.textMuted, margin: 0, charSpacing: 2,
});

// --- Slide 1-1: 人工智能发展历程 ---
const s11 = contentSlide(1, "人工智能整体发展历程", "1");
// Timeline
const milestones = [
  { year: "1956", event: "达特茅斯会议，AI 概念诞生" },
  { year: "2012", event: "深度学习突破（AlexNet）" },
  { year: "2017", event: "Transformer 架构提出" },
  { year: "2017", event: "中国《新一代人工智能发展规划》" },
  { year: "2022", event: "ChatGPT 发布，生成式 AI 爆发" },
  { year: "2024", event: "AI 医疗器械全球获批超 1000 个" },
];
milestones.forEach((m, i) => {
  const xBase = 0.5 + i * 1.55;
  // Line
  s11.addShape(pres.shapes.LINE, {
    x: xBase + 0.15, y: 1.9, w: 1.4, h: 0,
    line: { color: C.border, width: 2 },
  });
  // Dot
  s11.addShape(pres.shapes.OVAL, {
    x: xBase + 0.5, y: 1.75, w: 0.25, h: 0.25,
    fill: { color: i < 3 ? C.textMuted : C.primary },
  });
  // Year
  s11.addText(m.year, {
    x: xBase, y: 1.3, w: 1.3, h: 0.35,
    fontSize: 13, fontFace: FONT_TITLE, color: i < 3 ? C.textMuted : C.primary,
    bold: true, align: "center", margin: 0,
  });
  // Event
  s11.addText(m.event, {
    x: xBase - 0.05, y: 2.2, w: 1.5, h: 0.7,
    fontSize: 9, fontFace: FONT_BODY, color: C.textBody, align: "center",
    margin: 0, valign: "top",
  });
});
// AI 三步走战略 box
s11.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 3.15, w: 9, h: 0.85,
  fill: { color: C.highlight },
});
s11.addText("中国《新一代人工智能发展规划》\"三步走\"战略：2020 年与世界同步 → 2025 年部分领先 → 2030 年总体领先", {
  x: 0.7, y: 3.2, w: 8.6, h: 0.75,
  fontSize: 11, fontFace: FONT_BODY, color: C.primary, margin: 0, valign: "middle",
});
// Key stats
s11.addText([
  { text: "核心产业规模目标：", options: { bold: true, breakLine: true } },
  { text: "2025 年超 4000 亿元  |  2030 年超 1 万亿元", options: { fontSize: 12, color: C.textDark } },
  { text: "带动相关产业规模超 10 万亿元", options: { fontSize: 11, color: C.textMuted, breakLine: true } },
], { x: 0.5, y: 4.15, w: 9, h: 0.9, fontFace: FONT_BODY, valign: "top" });

// --- Slide 1-2: AI 医疗器械应用场景 ---
const s12 = contentSlide(1, "AI 在医疗器械上的七大应用场景", "2");
const apps = [
  { icon: "🩻", title: "医学影像诊断", desc: "肺结节、乳腺癌、眼底病变等 AI 辅助识别（占比约 40%）" },
  { icon: "💊", title: "药物发现与研发", desc: "AI 加速靶点发现、分子筛选、临床试验优化" },
  { icon: "📋", title: "临床决策支持", desc: "基于大模型的辅助诊断和治疗推荐" },
  { icon: "🧬", title: "基因组学与精准医疗", desc: "AI 分析基因数据指导个体化治疗方案" },
  { icon: "⌚", title: "远程监测与可穿戴", desc: "AI 驱动的健康监测和实时预警" },
  { icon: "🤖", title: "手术机器人", desc: "AI 辅助手术规划和机器人控制" },
  { icon: "🏥", title: "医院运营管理", desc: "AI 优化诊疗流程和资源调度" },
];
apps.forEach((a, i) => {
  const row = Math.floor(i / 2);
  const col = i % 2;
  const xBase = 0.5 + col * 4.6;
  const yBase = 1.0 + row * 1.35;
  s12.addShape(pres.shapes.RECTANGLE, {
    x: xBase, y: yBase, w: 4.3, h: 1.15,
    fill: { color: C.cardBg },
    shadow: { type: "outer", blur: 3, offset: 1, color: "000000", opacity: 0.06 },
  });
  s12.addText(a.icon, {
    x: xBase + 0.2, y: yBase + 0.15, w: 0.6, h: 0.5,
    fontSize: 22, align: "center", valign: "middle", margin: 0,
  });
  s12.addText(a.title, {
    x: xBase + 0.9, y: yBase + 0.1, w: 3.2, h: 0.35,
    fontSize: 13, fontFace: FONT_TITLE, color: C.textDark, bold: true, margin: 0, valign: "middle",
  });
  s12.addText(a.desc, {
    x: xBase + 0.9, y: yBase + 0.45, w: 3.2, h: 0.55,
    fontSize: 10, fontFace: FONT_BODY, color: C.textMuted, margin: 0, valign: "top",
  });
});

// --- Slide 1-3: 市场规模 ---
const s13 = contentSlide(1, "全球及中国 AI 医疗器械市场规模", "3");
s13.addText("全球市场", {
  x: 0.5, y: 1.0, w: 4.3, h: 0.4,
  fontSize: 16, fontFace: FONT_TITLE, color: C.primary, bold: true, margin: 0,
});
// Global market data cards
const globalData = [
  { label: "2023 年市场规模", value: "~200 亿美元", color: C.primary },
  { label: "2030 年预测", value: ">1,880 亿美元", color: C.secondary },
  { label: "CAGR (2023-2030)", value: "37-42%", color: C.accent },
];
globalData.forEach((d, i) => {
  const xB = 0.5 + i * 3.1;
  s13.addShape(pres.shapes.RECTANGLE, {
    x: xB, y: 1.5, w: 2.8, h: 1.0,
    fill: { color: C.cardBg },
    shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.06 },
  });
  s13.addText(d.label, {
    x: xB, y: 1.55, w: 2.8, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, color: C.textMuted, align: "center", margin: 0,
  });
  s13.addText(d.value, {
    x: xB, y: 1.85, w: 2.8, h: 0.5,
    fontSize: 22, fontFace: FONT_TITLE, color: d.color, bold: true, align: "center", margin: 0, valign: "middle",
  });
});

s13.addText("中国市场", {
  x: 0.5, y: 2.8, w: 4.3, h: 0.4,
  fontSize: 16, fontFace: FONT_TITLE, color: C.secondary, bold: true, margin: 0,
});
const chinaData = [
  { label: "2023 年市场规模", value: "~60 亿元" },
  { label: "2028 年预测", value: ">300 亿元" },
  { label: "获批 AI 产品数", value: ">100 个" },
];
chinaData.forEach((d, i) => {
  const xB = 0.5 + i * 3.1;
  s13.addShape(pres.shapes.RECTANGLE, {
    x: xB, y: 3.3, w: 2.8, h: 1.0,
    fill: { color: C.cardBg },
    shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.06 },
  });
  s13.addText(d.label, {
    x: xB, y: 3.35, w: 2.8, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, color: C.textMuted, align: "center", margin: 0,
  });
  s13.addText(d.value, {
    x: xB, y: 3.65, w: 2.8, h: 0.5,
    fontSize: 22, fontFace: FONT_TITLE, color: C.secondary, bold: true, align: "center", margin: 0, valign: "middle",
  });
});
// Note
s13.addText("数据来源：前瞻产业研究院、Frost & Sullivan、NMPA 公开数据", {
  x: 0.5, y: 4.7, w: 9, h: 0.3,
  fontSize: 8, fontFace: FONT_BODY, color: C.textMuted, italic: true, margin: 0,
});

// --- Slide 1-4: 国家政策支持 ---
const s14 = contentSlide(1, "国家政策支持人工智能发展", "4");
const policies = [
  { title: "《新一代人工智能发展规划》", doc: "国发〔2017〕35 号", key: "\"三步走\"战略，AI 核心产业规模 2030 年超 1 万亿元" },
  { title: "十四五规划 & 2035 远景目标", doc: "2021 年 3 月", key: "明确 AI 关键算法研发，推进智能医疗和健康中国建设" },
  { title: "\"十四五\"医药工业发展规划", doc: "工信部联规〔2022〕1 号", key: "推动 AI 在药物研发和医疗器械领域的创新应用" },
  { title: "《医疗器械监督管理条例》", doc: "国务院令第 739 号 (2021)", key: "为 AI 医疗器械注册审批提供法律基础框架" },
  { title: "场景创新指导意见", doc: "国科发规〔2022〕199 号", key: "明确 AI 医疗为重点应用场景" },
];
policies.forEach((p, i) => {
  const yB = 1.0 + i * 0.82;
  // Accent bar
  s14.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: yB, w: 0.08, h: 0.65,
    fill: { color: [C.primary, C.secondary, C.accent, C.green, C.amber][i] },
  });
  s14.addText(p.title, {
    x: 0.75, y: yB, w: 2.8, h: 0.32,
    fontSize: 12, fontFace: FONT_TITLE, color: C.textDark, bold: true, margin: 0, valign: "middle",
  });
  s14.addText(p.doc, {
    x: 3.6, y: yB, w: 1.8, h: 0.32,
    fontSize: 10, fontFace: FONT_BODY, color: C.primary, margin: 0, valign: "middle",
  });
  s14.addText(p.key, {
    x: 0.75, y: yB + 0.32, w: 8.5, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, color: C.textMuted, margin: 0, valign: "top",
  });
});

// ============================================================
// 习近平关于AI的重要讲话（嵌入第一章）
// ============================================================

// --- Slide 1-5: 2017 十九大 · 首次写入 ---
const s15 = contentSlide(1, '习近平关于AI的重要讲话与指示（一）', '5');
s15.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.0, w: 9, h: 1.2,
  fill: { color: C.cardBg },
  shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
});
s15.addText('"推动互联网、大数据、人工智能和实体经济深度融合。"', {
  x: 0.7, y: 1.05, w: 8.6, h: 0.6,
  fontSize: 16, fontFace: FONT_TITLE, color: C.primary, bold: true,
  italic: true, align: "center", valign: "middle", margin: 0,
});
s15.addText('——2017年10月 中国共产党第十九次全国代表大会报告', {
  x: 0.7, y: 1.6, w: 8.6, h: 0.3,
  fontSize: 10, fontFace: FONT_BODY, color: C.textMuted, align: "center", margin: 0,
});
s15.addText('⭐ 首次将"人工智能"写入党代会报告', {
  x: 0.7, y: 1.9, w: 8.6, h: 0.25,
  fontSize: 11, fontFace: FONT_BODY, color: C.accent, bold: true, align: "center", margin: 0,
});
// Key events 2017-2018
const xjEvents1 = [
  { year: '2017.07', event: '中央深改委审议通过《新一代人工智能发展规划》"三步走"战略', color: C.primary },
  { year: '2017.10', event: '十九大：AI与实体经济深度融合', color: C.primary },
  { year: '2018.05', event: '两院院士大会：AI是"大国竞争的战略制高点"', color: C.secondary },
  { year: '2018.10', event: '政治局第九次集体学习（首次AI专题）：AI是"重要驱动力量"', color: C.accent },
];
xjEvents1.forEach((e, i) => {
  const yB = 2.55 + i * 0.65;
  s15.addShape(pres.shapes.OVAL, {
    x: 0.6, y: yB + 0.08, w: 0.3, h: 0.3,
    fill: { color: e.color },
  });
  s15.addText(e.year, {
    x: 1.1, y: yB, w: 1.5, h: 0.4,
    fontSize: 10, fontFace: FONT_TITLE, color: e.color, bold: true, margin: 0, valign: "middle",
  });
  s15.addText(e.event, {
    x: 2.7, y: yB, w: 6.5, h: 0.4,
    fontSize: 11, fontFace: FONT_BODY, color: C.textBody, margin: 0, valign: "middle",
  });
});

// --- Slide 1-6: 2019-2020 头雁效应 + 城市治理 ---
const s16 = contentSlide(1, '习近平关于AI的重要讲话与指示（二）', '6');
s16.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.0, w: 4.3, h: 2.2,
  fill: { color: C.cardBg },
  shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
});
s16.addText('2019 · 头雁效应', {
  x: 0.7, y: 1.1, w: 3.9, h: 0.3,
  fontSize: 14, fontFace: FONT_TITLE, color: C.primary, bold: true, margin: 0,
});
s16.addText('"人工智能是引领这一轮科技革命和产业变革的战略性技术，具有溢出带动性很强的\'头雁\'效应。"', {
  x: 0.7, y: 1.45, w: 3.9, h: 0.8,
  fontSize: 10, fontFace: FONT_BODY, color: C.textBody, margin: 0, italic: true, valign: "top",
});
s16.addText('—— 2019年8月 世界人工智能大会（WAIC）贺信', {
  x: 0.7, y: 2.2, w: 3.9, h: 0.25,
  fontSize: 9, fontFace: FONT_BODY, color: C.textMuted, margin: 0,
});

s16.addShape(pres.shapes.RECTANGLE, {
  x: 5.2, y: 1.0, w: 4.3, h: 2.2,
  fill: { color: C.cardBg },
  shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
});
s16.addText('2020 · 城市治理 + AI', {
  x: 5.4, y: 1.1, w: 3.9, h: 0.3,
  fontSize: 14, fontFace: FONT_TITLE, color: C.secondary, bold: true, margin: 0,
});
s16.addText('"通过大数据、云计算、人工智能等手段推进城市治理现代化，大城市也可以变得更\'聪明\'。"', {
  x: 5.4, y: 1.45, w: 3.9, h: 0.8,
  fontSize: 10, fontFace: FONT_BODY, color: C.textBody, margin: 0, italic: true, valign: "top",
});
s16.addText('—— 2020年3月 考察杭州城市大脑运营指挥中心', {
  x: 5.4, y: 2.2, w: 3.9, h: 0.25,
  fontSize: 9, fontFace: FONT_BODY, color: C.textMuted, margin: 0,
});

// Timeline bottom
s16.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 3.5, w: 9, h: 0.05,
  fill: { color: C.highlight },
});
const xjEvents2 = [
  { year: '2019.08', event: 'WAIC贺信：AI"头雁效应"', detail: 'AI对其他技术的牵引带动作用' },
  { year: '2020.03', event: '杭州考察：城市大脑' , detail: 'AI赋能城市治理现代化' },
  { year: '2020.10', event: '深圳特区40周年' , detail: '发展AI等新型基础设施建设' },
  { year: '2020.11', event: '第三届进博会' , detail: '与各国加强AI前沿领域合作' },
];
xjEvents2.forEach((e, i) => {
  const xB = 0.5 + i * 2.35;
  s16.addShape(pres.shapes.RECTANGLE, {
    x: xB, y: 3.75, w: 2.15, h: 1.1,
    fill: { color: C.cardBg },
  });
  s16.addText(e.year, {
    x: xB + 0.1, y: 3.78, w: 1.95, h: 0.25,
    fontSize: 10, fontFace: FONT_TITLE, color: C.primary, bold: true, margin: 0,
  });
  s16.addText(e.event, {
    x: xB + 0.1, y: 4.0, w: 1.95, h: 0.3,
    fontSize: 9, fontFace: FONT_BODY, color: C.textDark, bold: true, margin: 0,
  });
  s16.addText(e.detail, {
    x: xB + 0.1, y: 4.3, w: 1.95, h: 0.3,
    fontSize: 8, fontFace: FONT_BODY, color: C.textMuted, margin: 0,
  });
});

// --- Slide 1-7: 2021-2022 十四五 + 二十大 ---
const s17 = contentSlide(1, '习近平关于AI的重要讲话与指示（三）', '7');
s17.addText('2021 · 十四五规划开局', {
  x: 0.5, y: 1.0, w: 4.3, h: 0.35,
  fontSize: 14, fontFace: FONT_TITLE, color: C.primary, bold: true, margin: 0,
});
s17.addText([
  { text: '"瞄准人工智能、量子信息、集成电路等前沿领域"', options: { bullet: true, breakLine: true, fontSize: 11, color: C.textBody } },
  { text: '中国第一个将AI作为核心发展方向的五年规划', options: { bullet: true, breakLine: true, fontSize: 11, color: C.textMuted } },
  { text: '两院院士大会提出"高水平科技自立自强"', options: { bullet: true, fontSize: 11, color: C.textDark, bold: true } },
], { x: 0.5, y: 1.45, w: 4.3, h: 1.5, fontFace: FONT_BODY, valign: "top", paraSpaceAfter: 6 });

s17.addShape(pres.shapes.RECTANGLE, {
  x: 5.2, y: 1.0, w: 4.3, h: 2.0,
  fill: { color: C.cardBg },
  shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
});
s17.addText('2022 · 二十大报告', {
  x: 5.4, y: 1.1, w: 3.9, h: 0.35,
  fontSize: 14, fontFace: FONT_TITLE, color: C.accent, bold: true, margin: 0,
});
s17.addText('"构建人工智能等一批新的增长引擎"', {
  x: 5.4, y: 1.5, w: 3.9, h: 0.5,
  fontSize: 13, fontFace: FONT_BODY, color: C.textDark, italic: true, margin: 0, align: "center", valign: "middle",
});
s17.addText('从2017年"深度融合" → 2022年"新增长引擎"\n战略定位的重大升级', {
  x: 5.4, y: 2.0, w: 3.9, h: 0.6,
  fontSize: 10, fontFace: FONT_BODY, color: C.textMuted, align: "center", margin: 0, valign: "top",
});

// --- Slide 1-8: 2023-2024 AGI · 新质生产力 · 治理 ---
const s18 = contentSlide(1, '习近平关于AI的重要讲话与指示（四）', '8');
// 2023
s18.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.0, w: 4.3, h: 2.6,
  fill: { color: C.cardBg },
  shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
});
s18.addText('2023 · 里程碑之年', {
  x: 0.7, y: 1.1, w: 3.9, h: 0.3,
  fontSize: 14, fontFace: FONT_TITLE, color: C.primary, bold: true, margin: 0,
});
const y2023items = [
  { text: '政治局会议：首次提出"通用人工智能（AGI）"', options: { bullet: true, breakLine: true, fontSize: 10, bold: true, color: C.textDark } },
  { text: '"要重视通用人工智能发展，营造创新生态，重视防范风险"', options: { bullet: true, breakLine: true, fontSize: 9, italic: true, color: C.textMuted } },
  { text: '中央财经委："把握AI新科技革命浪潮"', options: { bullet: true, breakLine: true, fontSize: 10, color: C.textBody } },
  { text: 'WAIC贺信："以人为本、智能向善"', options: { bullet: true, breakLine: true, fontSize: 10, bold: true, color: C.primary } },
  { text: '"构建开放、公正、有效的AI治理机制"', options: { bullet: true, fontSize: 10, color: C.textBody } },
];
s18.addText(y2023items, {
  x: 0.7, y: 1.5, w: 3.9, h: 2.0, fontFace: FONT_BODY, valign: "top", paraSpaceAfter: 4,
});

// 2024
s18.addShape(pres.shapes.RECTANGLE, {
  x: 5.2, y: 1.0, w: 4.3, h: 2.6,
  fill: { color: C.cardBg },
  shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
});
s18.addText('2024 · 新质生产力 + 发展与治理并重', {
  x: 5.4, y: 1.1, w: 3.9, h: 0.3,
  fontSize: 12, fontFace: FONT_TITLE, color: C.secondary, bold: true, margin: 0,
});
const y2024items = [
  { text: '政治局学习：AI定位为"新质生产力"核心引擎', options: { bullet: true, breakLine: true, fontSize: 10, bold: true, color: C.textDark } },
  { text: '全国科技大会：AI是"战略制高点"', options: { bullet: true, breakLine: true, fontSize: 10, color: C.textBody } },
  { text: 'WAIC贺信：连续第六年致贺信', options: { bullet: true, breakLine: true, fontSize: 10, color: C.textBody } },
  { text: '二十届三中全会：首次AI发展与治理并重写入', options: { bullet: true, fontSize: 10, bold: true, color: C.accent } },
];
s18.addText(y2024items, {
  x: 5.4, y: 1.5, w: 3.9, h: 2.0, fontFace: FONT_BODY, valign: "top", paraSpaceAfter: 4,
});

// Bottom note
s18.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 3.85, w: 9, h: 0.55,
  fill: { color: C.highlight },
});
s18.addText('2025-2026：政治局再次专题研究AI发展与安全 | "十五五"规划：AI是"抢占未来竞争优势的战略方向" | AI是"发展新质生产力的核心引擎"', {
  x: 0.7, y: 3.9, w: 8.6, h: 0.45,
  fontSize: 10, fontFace: FONT_BODY, color: C.primary, margin: 0, valign: "middle",
});

// --- Slide 1-9: 概念演变时间轴 ---
const s19 = contentSlide(1, 'AI战略定位：关键概念演变脉络', '9');
const evolution = [
  { year: '2017', concept: '深度融合', desc: 'AI与实体经济融合', color: C.primary },
  { year: '2018', concept: '战略制高点', desc: '大国竞争的战略领域', color: C.primaryL },
  { year: '2019', concept: '头雁效应', desc: '带动其他技术发展', color: C.secondary },
  { year: '2021', concept: '科技自立自强', desc: 'AI是突破关键', color: C.secondary },
  { year: '2022', concept: '新增长引擎', desc: '战略性新兴产业', color: C.accent },
  { year: '2023', concept: '通用AI/向善', desc: 'AGI+以人为本', color: C.accent },
  { year: '2024', concept: '新质生产力', desc: '核心引擎', color: C.red },
  { year: '2026', concept: '核心引擎', desc: '新质生产力核心', color: C.primary },
];
evolution.forEach((e, i) => {
  const xB = 0.3 + i * 1.2;
  // Vertical line
  if (i < evolution.length - 1) {
    s19.addShape(pres.shapes.LINE, {
      x: xB + 0.85, y: 1.5, w: 0.9, h: 0,
      line: { color: C.border, width: 1.5 },
    });
  }
  // Dot
  s19.addShape(pres.shapes.OVAL, {
    x: xB + 0.55, y: 1.35, w: 0.3, h: 0.3,
    fill: { color: e.color },
  });
  // Year
  s19.addText(e.year, {
    x: xB, y: 1.1, w: 1.1, h: 0.25,
    fontSize: 9, fontFace: FONT_TITLE, color: e.color, bold: true, align: "center", margin: 0,
  });
  // Concept
  s19.addShape(pres.shapes.RECTANGLE, {
    x: xB, y: 1.8, w: 1.1, h: 0.8,
    fill: { color: e.color, transparency: 85 },
    line: { color: e.color, width: 0.5 },
  });
  s19.addText(e.concept, {
    x: xB, y: 1.85, w: 1.1, h: 0.4,
    fontSize: 9, fontFace: FONT_TITLE, color: e.color, bold: true, align: "center", valign: "middle", margin: 0,
  });
  s19.addText(e.desc, {
    x: xB, y: 2.2, w: 1.1, h: 0.3,
    fontSize: 7, fontFace: FONT_BODY, color: C.textMuted, align: "center", margin: 0, valign: "top",
  });
});

// Global governance summary
s19.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 3.0, w: 9, h: 1.5,
  fill: { color: C.cardBg },
  shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
});
s19.addText('AI全球治理的中国主张', {
  x: 0.7, y: 3.1, w: 8.6, h: 0.3,
  fontSize: 13, fontFace: FONT_TITLE, color: C.primary, bold: true, margin: 0,
});
const govItems = [
  { num: '①', text: '以人为本、智能向善' },
  { num: '②', text: '开放合作、共治共享' },
  { num: '③', text: '统筹发展与安全' },
  { num: '④', text: '构建开放、公正、有效的全球AI治理机制' },
  { num: '⑤', text: '推动AI赋能可持续发展目标' },
  { num: '⑥', text: '反对技术霸权，主张各国平等参与' },
];
govItems.forEach((g, i) => {
  const col = i % 3;
  const row = Math.floor(i / 3);
  const xB = 0.7 + col * 3.0;
  const yB = 3.5 + row * 0.45;
  s19.addText(g.num, {
    x: xB, y: yB, w: 0.3, h: 0.35,
    fontSize: 11, fontFace: FONT_TITLE, color: C.primary, bold: true, margin: 0, valign: "middle",
  });
  s19.addText(g.text, {
    x: xB + 0.35, y: yB, w: 2.6, h: 0.35,
    fontSize: 9, fontFace: FONT_BODY, color: C.textBody, margin: 0, valign: "middle",
  });
});

// ============================================================
// 中国AI政策、纲要、时政支持总览（嵌入第一章）
// ============================================================

// --- Policy Slide 1: 核心政策文件与纲要 ---
const p1 = contentSlide(1, '中国AI核心政策文件与纲要', '10');
const policyList = [
  { year: '2017', name: '《新一代人工智能发展规划》', org: '国务院', key: '"三步走"战略：2020同步→2025部分领先→2030总体领先', color: C.primary },
  { year: '2021', name: '十四五规划 & 2035远景目标', org: '全国人大', key: 'AI列为前沿科技攻关重点方向，首个AI核心五年规划', color: C.primaryL },
  { year: '2022', name: '"十四五"医药工业发展规划', org: '工信部', key: '推动AI在药物研发和医疗器械领域的创新应用', color: C.secondary },
  { year: '2022', name: 'AI医疗器械注册审查指导原则', org: 'NMPA', key: '"算力-算法-数据"三位一体评价框架', color: C.secondary },
  { year: '2024', name: '二十届三中全会《决定》', org: '中共中央', key: '首次AI发展与治理并重写入中央全会决定', color: C.accent },
  { year: '2025', name: '《"人工智能+"行动意见》', org: '国务院', key: '强化算力统筹，推动AI全环节落地，发展智能终端', color: C.accent },
  { year: '2025', name: '"人工智能+制造"专项行动', org: '工信部等八部门', key: '2027年推广500个典型应用场景', color: C.red },
  { year: '2025', name: '"数据要素×"三年行动计划', org: '国家数据局', key: '打造高质量数据集，培育数据产业', color: C.red },
];
policyList.forEach((p, i) => {
  const yB = 0.95 + i * 0.53;
  // Left accent
  p1.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: yB, w: 0.06, h: 0.42,
    fill: { color: p.color },
  });
  // Year badge
  p1.addShape(pres.shapes.RECTANGLE, {
    x: 0.7, y: yB + 0.05, w: 0.55, h: 0.3,
    fill: { color: p.color },
  });
  p1.addText(p.year, {
    x: 0.7, y: yB + 0.05, w: 0.55, h: 0.3,
    fontSize: 8, fontFace: FONT_TITLE, color: C.white, bold: true, align: "center", valign: "middle", margin: 0,
  });
  // Policy name
  p1.addText(p.name, {
    x: 1.4, y: yB, w: 2.8, h: 0.22,
    fontSize: 10, fontFace: FONT_TITLE, color: C.textDark, bold: true, margin: 0, valign: "middle",
  });
  p1.addText(p.org, {
    x: 1.4, y: yB + 0.22, w: 2.8, h: 0.2,
    fontSize: 8, fontFace: FONT_BODY, color: C.textMuted, margin: 0, valign: "middle",
  });
  // Key content
  p1.addText(p.key, {
    x: 4.3, y: yB, w: 5.2, h: 0.42,
    fontSize: 9, fontFace: FONT_BODY, color: C.textBody, margin: 0, valign: "middle",
  });
});
// Bottom note
p1.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 4.65, w: 9, h: 0.45,
  fill: { color: C.highlight },
});
p1.addText('2026年"十五五"规划：AI作为"抢占未来竞争优势的战略方向"  |  AI企业超6,000家，核心产业规模突破1.2万亿元（2025年）', {
  x: 0.7, y: 4.68, w: 8.6, h: 0.4,
  fontSize: 10, fontFace: FONT_BODY, color: C.primary, margin: 0, valign: "middle",
});

// --- Policy Slide 2: 时政支持与最新举措 ---
const p2 = contentSlide(1, '时政支持与最新举措', '11');
// Left column: 顶层设计
p2.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.0, w: 4.3, h: 3.2,
  fill: { color: C.cardBg },
  shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
});
p2.addText('顶层设计与战略部署', {
  x: 0.7, y: 1.1, w: 3.9, h: 0.3,
  fontSize: 14, fontFace: FONT_TITLE, color: C.primary, bold: true, margin: 0,
});
const topItems = [
  { text: '政治局多次集体学习AI专题（2018、2023、2025）', options: { bullet: true, breakLine: true, fontSize: 10, bold: true } },
  { text: '习近平连续7年向WAIC致贺信（2018-2025）', options: { bullet: true, breakLine: true, fontSize: 10 } },
  { text: '二十届三中全会：AI发展与治理并重', options: { bullet: true, breakLine: true, fontSize: 10 } },
  { text: '"新质生产力"核心引擎定位（2024.01）', options: { bullet: true, breakLine: true, fontSize: 10 } },
  { text: '"十五五"规划：AI是战略方向', options: { bullet: true, breakLine: true, fontSize: 10 } },
  { text: 'AI安全治理专题指示（2026.02）', options: { bullet: true, fontSize: 10 } },
];
p2.addText(topItems, {
  x: 0.7, y: 1.5, w: 3.9, h: 2.5, fontFace: FONT_BODY, color: C.textBody, valign: "top", paraSpaceAfter: 5,
});

// Right column: 产业支持
p2.addShape(pres.shapes.RECTANGLE, {
  x: 5.2, y: 1.0, w: 4.3, h: 3.2,
  fill: { color: C.cardBg },
  shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
});
p2.addText('产业支持与生态建设', {
  x: 5.4, y: 1.1, w: 3.9, h: 0.3,
  fontSize: 14, fontFace: FONT_TITLE, color: C.secondary, bold: true, margin: 0,
});
const indItems = [
  { text: '42个万卡智算集群建成，算力超1,590 EFLOPS', options: { bullet: true, breakLine: true, fontSize: 10 } },
  { text: 'AI核心产业规模突破1.2万亿元（2025）', options: { bullet: true, breakLine: true, fontSize: 10, bold: true } },
  { text: 'AI企业超6,000家，专利占全球60%', options: { bullet: true, breakLine: true, fontSize: 10 } },
  { text: '7个国家级数据标注基地建设', options: { bullet: true, breakLine: true, fontSize: 10 } },
  { text: '高质量行业数据集超500个（医疗、工业等）', options: { bullet: true, breakLine: true, fontSize: 10 } },
  { text: '"人工智能+制造"：2027年500个典型场景', options: { bullet: true, fontSize: 10 } },
];
p2.addText(indItems, {
  x: 5.4, y: 1.5, w: 3.9, h: 2.5, fontFace: FONT_BODY, color: C.textBody, valign: "top", paraSpaceAfter: 5,
});

// Bottom: 核心数据亮点
p2.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 4.45, w: 9, h: 0.65,
  fill: { color: C.highlight },
});
p2.addText('📊 核心数据亮点', {
  x: 0.7, y: 4.5, w: 2, h: 0.25,
  fontSize: 10, fontFace: FONT_TITLE, color: C.primary, bold: true, margin: 0,
});
p2.addText('2025年AI核心产业规模突破1.2万亿元 · 国产开源大模型全球下载超100亿次 · 日均Token消耗30万亿（18个月增长300倍）', {
  x: 0.7, y: 4.75, w: 8.6, h: 0.3,
  fontSize: 9, fontFace: FONT_BODY, color: C.textDark, margin: 0, valign: "middle",
});

// ============================================================
// CHAPTER 2: 法规和监管
// ============================================================

// --- Chapter 2 divider ---
const ch2Div = pres.addSlide();
ch2Div.background = { color: C.darkBg };
ch2Div.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.secondary },
});
ch2Div.addText("02", {
  x: 0.8, y: 1.2, w: 2, h: 1.2,
  fontSize: 60, fontFace: FONT_TITLE, color: C.secondary, bold: true, margin: 0,
});
ch2Div.addText("法规和监管", {
  x: 0.8, y: 2.4, w: 8, h: 0.7,
  fontSize: 28, fontFace: FONT_TITLE, color: C.white, bold: true, margin: 0,
});
ch2Div.addShape(pres.shapes.LINE, {
  x: 0.8, y: 3.2, w: 2, h: 0, line: { color: C.secondary, width: 2.5 },
});
ch2Div.addText("China NMPA · US FDA · EU MDR + AI Act · Global Governance", {
  x: 0.8, y: 3.4, w: 8, h: 0.5,
  fontSize: 11, fontFace: FONT_BODY, color: C.textMuted, margin: 0, charSpacing: 2,
});

// --- Slide 2-1: 中国 NMPA ---
const s21 = contentSlide(2, "中国 NMPA 监管体系", "12");
const nmpaItems = [
  { title: "《人工智能医疗器械注册审查指导原则》", detail: "2022 年第 50 号通告 — 核心指南", note: "定义 AI 医疗器械，提出\"算力-算法-数据\"三位一体评价框架" },
  { title: "《医疗器械软件注册技术审查指导原则》", detail: "2022 年修订版", note: "明确软件医疗器械分类规则和注册要求" },
  { title: "医疗器械分类目录（2023 年更新）", detail: "新增 AI 辅助诊断软件分类条目", note: "AI 医学影像类产品普遍归为 II 类或 III 类" },
  { title: "AI 辅助检测临床评价指导原则", detail: "2024 年征求意见稿", note: "针对 AI 辅助检测类产品的临床评价具体要求" },
];
nmpaItems.forEach((item, i) => {
  const yB = 1.0 + i * 1.05;
  s21.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: yB, w: 9, h: 0.9,
    fill: { color: C.cardBg },
    shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
  });
  // Number
  s21.addShape(pres.shapes.OVAL, {
    x: 0.7, y: yB + 0.2, w: 0.45, h: 0.45,
    fill: { color: C.primary },
  });
  s21.addText(String(i + 1), {
    x: 0.7, y: yB + 0.2, w: 0.45, h: 0.45,
    fontSize: 16, fontFace: FONT_TITLE, color: C.white, bold: true, align: "center", valign: "middle", margin: 0,
  });
  s21.addText(item.title, {
    x: 1.35, y: yB + 0.08, w: 7.5, h: 0.3,
    fontSize: 12, fontFace: FONT_TITLE, color: C.textDark, bold: true, margin: 0, valign: "middle",
  });
  s21.addText(item.detail, {
    x: 1.35, y: yB + 0.4, w: 3.5, h: 0.25,
    fontSize: 10, fontFace: FONT_BODY, color: C.primary, margin: 0, valign: "middle",
  });
  s21.addText(item.note, {
    x: 4.9, y: yB + 0.4, w: 4.4, h: 0.25,
    fontSize: 10, fontFace: FONT_BODY, color: C.textMuted, margin: 0, valign: "middle",
  });
});
// Bottom stats
s21.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 4.3, w: 9, h: 0.55,
  fill: { color: C.highlight },
});
s21.addText("截至 2025 年初：已有超过 100 个 AI 医疗器械产品获得 NMPA 注册批准  |  2024 年 NMPA 进一步优化 AI 软件注册流程", {
  x: 0.7, y: 4.35, w: 8.6, h: 0.45,
  fontSize: 11, fontFace: FONT_BODY, color: C.primary, margin: 0, valign: "middle",
});

// --- Slide 2-2: 美国 FDA ---
const s22 = contentSlide(2, "美国 FDA 监管框架", "13");
// Left column: AI/ML SaMD
s22.addText("AI/ML-Based SaMD 框架", {
  x: 0.5, y: 1.0, w: 4.5, h: 0.35,
  fontSize: 14, fontFace: FONT_TITLE, color: C.textDark, bold: true, margin: 0,
});
const pillars = [
  "定制化监管框架 (Tailored Framework)",
  "良好的机器学习实践 (GMLP)",
  "患者为中心的透明度",
  "真实世界性能监测",
];
pillars.forEach((p, i) => {
  const yB = 1.5 + i * 0.45;
  s22.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: yB, w: 0.08, h: 0.35,
    fill: { color: C.primary },
  });
  s22.addText(`${i + 1}. ${p}`, {
    x: 0.75, y: yB, w: 4.2, h: 0.35,
    fontSize: 11, fontFace: FONT_BODY, color: C.textBody, margin: 0, valign: "middle",
  });
});

// Right column: PCCP
s22.addShape(pres.shapes.RECTANGLE, {
  x: 5.2, y: 0.85, w: 4.3, h: 4.15,
  fill: { color: C.cardBg },
  shadow: { type: "outer", blur: 3, offset: 1, color: "000000", opacity: 0.06 },
});
s22.addText("预定变更控制计划\n(PCCP)", {
  x: 5.4, y: 0.95, w: 3.9, h: 0.55,
  fontSize: 14, fontFace: FONT_TITLE, color: C.primary, bold: true, margin: 0, align: "center",
});
s22.addText("允许 AI/ML 医疗器械在上市后通过 PCCP 进行持续学习和迭代更新", {
  x: 5.4, y: 1.55, w: 3.9, h: 0.4,
  fontSize: 10, fontFace: FONT_BODY, color: C.textMuted, margin: 0, align: "center",
});
s22.addShape(pres.shapes.LINE, {
  x: 5.6, y: 2.05, w: 3.5, h: 0, line: { color: C.border, width: 1 },
});
const pccpElements = [
  { title: "变更描述", desc: "变更类型和范围的详细描述" },
  { title: "修改协议", desc: "开发、验证、验证方法" },
  { title: "影响评估", desc: "对安全性和有效性的影响评估" },
];
pccpElements.forEach((e, i) => {
  s22.addText(`${e.title}：${e.desc}`, {
    x: 5.4, y: 2.25 + i * 0.55, w: 3.9, h: 0.4,
    fontSize: 10, fontFace: FONT_BODY, color: C.textBody, margin: 0, valign: "middle",
    bullet: true,
  });
});

// Bottom stats
s22.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 4.2, w: 9, h: 0.6,
  fill: { color: C.highlight },
});
s22.addText([
  { text: "FDA 已批准超过 1,000 个 AI/ML 医疗器械", options: { bold: true, breakLine: true, fontSize: 12, color: C.primary } },
  { text: "2024 年新增约 250 个 (增长 40%)  |  放射学占比 ~75%  |  ~97% 为 510(k) 途径", options: { fontSize: 10, color: C.textMuted } },
], { x: 0.7, y: 4.25, w: 8.6, h: 0.5, fontFace: FONT_BODY, valign: "middle" });

// --- Slide 2-3: 欧盟 ---
const s23 = contentSlide(2, "欧盟监管体系：MDR + EU AI Act", "14");
// MDR section
s23.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.0, w: 4.3, h: 3.3,
  fill: { color: C.cardBg },
  shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
});
s23.addText("MDR (EU 2017/745)", {
  x: 0.7, y: 1.1, w: 3.9, h: 0.35,
  fontSize: 14, fontFace: FONT_TITLE, color: C.secondary, bold: true, margin: 0,
});
s23.addText([
  { text: "全面实施：2021 年 5 月 26 日", options: { bullet: true, breakLine: true, fontSize: 11 } },
  { text: "Rule 11：专门针对医疗软件分类", options: { bullet: true, breakLine: true, fontSize: 11 } },
  { text: "更严格的临床评价 (Annex XIV)", options: { bullet: true, breakLine: true, fontSize: 11 } },
  { text: "公告机构 (Notified Body) 监督强化", options: { bullet: true, fontSize: 11 } },
], { x: 0.7, y: 1.55, w: 3.9, h: 2.5, fontFace: FONT_BODY, color: C.textBody, valign: "top", paraSpaceAfter: 4 });

// EU AI Act section
s23.addShape(pres.shapes.RECTANGLE, {
  x: 5.2, y: 1.0, w: 4.3, h: 3.3,
  fill: { color: C.cardBg },
  shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
});
s23.addText("EU AI Act (2024)", {
  x: 5.4, y: 1.1, w: 3.9, h: 0.35,
  fontSize: 14, fontFace: FONT_TITLE, color: C.accent, bold: true, margin: 0,
});
s23.addText([
  { text: "2024 年 8 月 1 日正式生效", options: { bullet: true, breakLine: true, fontSize: 11 } },
  { text: "2026 年 8 月全面实施", options: { bullet: true, breakLine: true, fontSize: 11 } },
  { text: "AI 医疗器械归为\"高风险 AI 系统\"", options: { bullet: true, breakLine: true, fontSize: 11, bold: true, color: C.red } },
  { text: "须同时满足 EU AI Act + MDR/IVDR", options: { bullet: true, fontSize: 11 } },
], { x: 5.4, y: 1.55, w: 3.9, h: 2.5, fontFace: FONT_BODY, color: C.textBody, valign: "top", paraSpaceAfter: 4 });

// Bottom note
s23.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 4.5, w: 9, h: 0.5,
  fill: { color: C.highlight },
});
s23.addText("监管协同：MDR 已注册的 AI 医疗器械被视为已符合 EU AI Act 的高风险 AI 系统要求", {
  x: 0.7, y: 4.55, w: 8.6, h: 0.4,
  fontSize: 11, fontFace: FONT_BODY, color: C.primary, margin: 0, valign: "middle",
});

// --- Slide 2-4: 三地法规对比 ---
const s24 = contentSlide(2, "中美欧 AI 医疗器械法规对比", "15");
const headers = ["维度", "中国 (NMPA)", "美国 (FDA)", "欧盟 (EU)"];
const rows = [
  ["核心法规", "AI 医疗器械注册审查指导原则 (2022)", "AI/ML SaMD Action Plan (2021/2023)", "MDR + EU AI Act (2024)"],
  ["分类方法", "基于风险 (II 类 / III 类)", "基于风险 (II 类 / III 类)", "基于风险 (Class IIa/IIb/III)"],
  ["更新机制", "指导原则定期修订", "PCCP 预定变更控制计划", "通过 MDR 修订 + AI Act"],
  ["临床评价", "临床试验或真实世界数据", "临床性能数据", "临床评价 (Annex XIV)"],
  ["AI 专门指南", "已发布专门指导原则", "AI/ML 行动计划 + GMLP", "MDCG 指南 (2024 年更新中)"],
];

// Table
const tblRows = [headers, ...rows].map((row, ri) =>
  row.map((cell, ci) => ({
    text: cell,
    options: {
      fontSize: ri === 0 ? 10 : 9,
      fontFace: FONT_BODY,
      color: ri === 0 ? C.white : (ci === 0 ? C.textDark : C.textBody),
      bold: ri === 0 || ci === 0,
      fill: { color: ri === 0 ? C.primary : (ci === 0 ? C.highlight : C.cardBg) },
      align: ci === 0 ? "left" : "center",
      valign: "middle",
    },
  }))
);

s24.addTable(tblRows, {
  x: 0.5, y: 1.0, w: 9,
  colW: [1.2, 2.6, 2.6, 2.6],
  rowH: [0.4, 0.6, 0.6, 0.6, 0.6, 0.6],
  border: { pt: 0.5, color: C.border },
});

// --- Slide 2-5: 国际参考 ---
const s25 = contentSlide(2, "关键国际文件与标准", "16");
const intlItems = [
  "IMDRF \"Software as a Medical Device\" (SaMD) 框架 (2014-2020)",
  "WHO《Ethics and Governance of AI for Health》(2021)",
  "IEEE 2801-2022 医学人工智能数据集质量管理",
  "ISO/IEC 22989 人工智能概念与术语",
  "ISO/IEC 23053 AI 系统框架",
  "ISO 14971:2019 医疗器械风险管理",
];
s25.addText(intlItems.map((t, i) => ({
  text: t,
  options: { bullet: true, breakLine: i < intlItems.length - 1, fontSize: 12, color: C.textBody, paraSpaceAfter: 8 },
})), {
  x: 0.8, y: 1.1, w: 8.4, h: 3.5,
  fontFace: FONT_BODY, valign: "top",
});

// ============================================================
// CHAPTER 3: 人工智能医疗器械设计开发要点
// ============================================================

// --- Chapter 3 divider ---
const ch3Div = pres.addSlide();
ch3Div.background = { color: C.darkBg };
ch3Div.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent },
});
ch3Div.addText("03", {
  x: 0.8, y: 1.2, w: 2, h: 1.2,
  fontSize: 60, fontFace: FONT_TITLE, color: C.accent, bold: true, margin: 0,
});
ch3Div.addText("人工智能医疗器械\n设计开发要点", {
  x: 0.8, y: 2.4, w: 8, h: 1.0,
  fontSize: 28, fontFace: FONT_TITLE, color: C.white, bold: true, margin: 0,
});
ch3Div.addShape(pres.shapes.LINE, {
  x: 0.8, y: 3.6, w: 2, h: 0, line: { color: C.accent, width: 2.5 },
});
ch3Div.addText("Lifecycle · Risk Management · AI Literacy", {
  x: 0.8, y: 3.8, w: 8, h: 0.5,
  fontSize: 12, fontFace: FONT_BODY, color: C.textMuted, margin: 0, charSpacing: 2,
});

// --- Slide 3-1: 全生命周期流程 ---
const s31 = contentSlide(3, "AI 医疗器械全生命周期流程", "17");
s31.addText("基于 NMPA《人工智能医疗器械注册审查指导原则》", {
  x: 0.5, y: 0.85, w: 9, h: 0.3,
  fontSize: 10, fontFace: FONT_BODY, color: C.textMuted, italic: true, margin: 0,
});
const steps = [
  "需求分析", "数据采集", "数据预处理", "模型设计\n/选型", "模型训练",
  "模型验证", "模型确认", "临床评价", "注册申报", "上市后监管", "持续更新\n迭代",
];
// Flow: 3 rows
const row1 = steps.slice(0, 5);
const row2 = steps.slice(5, 9);
const row3 = steps.slice(9);
const allRows = [row1, row2, row3];
allRows.forEach((row, ri) => {
  const totalW = row.length * 1.6 + (row.length - 1) * 0.15;
  const startX = (10 - totalW) / 2;
  const yBase = 1.25 + ri * 1.25;
  row.forEach((s, ci) => {
    const xB = startX + ci * 1.75;
    s31.addShape(pres.shapes.RECTANGLE, {
      x: xB, y: yBase, w: 1.6, h: 0.85,
      fill: { color: ri === 2 ? C.highlight : C.cardBg },
      shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.06 },
    });
    // Accent top bar
    s31.addShape(pres.shapes.RECTANGLE, {
      x: xB, y: yBase, w: 1.6, h: 0.04,
      fill: { color: ri === 0 ? C.primary : ri === 1 ? C.secondary : C.accent },
    });
    s31.addText(s, {
      x: xB, y: yBase + 0.08, w: 1.6, h: 0.75,
      fontSize: s.length > 4 ? 9 : 10, fontFace: FONT_BODY, color: C.textDark,
      align: "center", valign: "middle", margin: 0, bold: true,
    });
    // Arrow (except last)
    if (ci < row.length - 1) {
      s31.addText("→", {
        x: xB + 1.6, y: yBase, w: 0.15, h: 0.85,
        fontSize: 14, color: C.textMuted, align: "center", valign: "middle", margin: 0,
      });
    }
  });
});
// Down arrows
s31.addText("↓", { x: 4.7, y: 2.15, w: 0.5, h: 0.3, fontSize: 16, color: C.textMuted, align: "center", margin: 0 });
s31.addText("↓", { x: 4.7, y: 3.35, w: 0.5, h: 0.3, fontSize: 16, color: C.textMuted, align: "center", margin: 0 });

// Note
s31.addText("关键设计环节：需求分析 → 数据管理 → 算法设计 → 验证确认 → 临床评价 → 持续更新", {
  x: 0.5, y: 4.55, w: 9, h: 0.35,
  fontSize: 11, fontFace: FONT_BODY, color: C.primary, align: "center", margin: 0,
});

// --- Slide 3-2: 数据管理 ---
const s32 = contentSlide(3, "数据管理关键要素", "18");
const dataItems = [
  { title: "数据来源合法性", desc: "受试者知情同意、伦理审查、数据授权链完整" },
  { title: "数据标注规范性", desc: "标注指南、标注者资质、多重标注一致性审核" },
  { title: "数据多样性", desc: "覆盖不同人群、不同设备、不同采集条件" },
  { title: "数据偏差控制", desc: "避免选择偏倚、测量偏倚、标注偏倚，代表目标人群分布" },
  { title: "数据安全与隐私", desc: "符合《个人信息保护法》《数据安全法》，数据脱敏与加密" },
];
dataItems.forEach((item, i) => {
  const yB = 1.0 + i * 0.8;
  s32.addShape(pres.shapes.OVAL, {
    x: 0.5, y: yB + 0.1, w: 0.4, h: 0.4,
    fill: { color: [C.primary, C.secondary, C.accent, C.green, C.amber][i] },
  });
  s32.addText(String(i + 1), {
    x: 0.5, y: yB + 0.1, w: 0.4, h: 0.4,
    fontSize: 14, fontFace: FONT_TITLE, color: C.white, bold: true, align: "center", valign: "middle", margin: 0,
  });
  s32.addText(item.title, {
    x: 1.1, y: yB, w: 3, h: 0.3,
    fontSize: 13, fontFace: FONT_TITLE, color: C.textDark, bold: true, margin: 0, valign: "middle",
  });
  s32.addText(item.desc, {
    x: 1.1, y: yB + 0.35, w: 7.5, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, color: C.textMuted, margin: 0, valign: "top",
  });
});

// --- Slide 3-3: 验证与确认 ---
const s33 = contentSlide(3, "算法验证与临床确认", "19");
// Left: 算法性能验证
s33.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.0, w: 4.3, h: 3.0,
  fill: { color: C.cardBg },
  shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
});
s33.addText("算法性能验证", {
  x: 0.7, y: 1.1, w: 3.9, h: 0.35,
  fontSize: 14, fontFace: FONT_TITLE, color: C.primary, bold: true, margin: 0,
});
s33.addText([
  { text: "数据集划分 (训练/验证/测试)", options: { bullet: true, breakLine: true, fontSize: 11 } },
  { text: "性能指标 (敏感度、特异度、AUC等)", options: { bullet: true, breakLine: true, fontSize: 11 } },
  { text: "对抗性测试与边界条件测试", options: { bullet: true, breakLine: true, fontSize: 11 } },
  { text: "鲁棒性评估 (噪声、图像质量变化)", options: { bullet: true, breakLine: true, fontSize: 11 } },
  { text: "亚组分析 (年龄、性别、疾病亚型)", options: { bullet: true, fontSize: 11 } },
], { x: 0.7, y: 1.6, w: 3.9, h: 2.2, fontFace: FONT_BODY, color: C.textBody, valign: "top", paraSpaceAfter: 4 });

// Right: 临床确认
s33.addShape(pres.shapes.RECTANGLE, {
  x: 5.2, y: 1.0, w: 4.3, h: 3.0,
  fill: { color: C.cardBg },
  shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
});
s33.addText("临床确认", {
  x: 5.4, y: 1.1, w: 3.9, h: 0.35,
  fontSize: 14, fontFace: FONT_TITLE, color: C.secondary, bold: true, margin: 0,
});
s33.addText([
  { text: "临床试验或临床评价", options: { bullet: true, breakLine: true, fontSize: 11 } },
  { text: "与金标准/现有方法对比", options: { bullet: true, breakLine: true, fontSize: 11 } },
  { text: "多中心验证 (至少 3 家中心)", options: { bullet: true, breakLine: true, fontSize: 11 } },
  { text: "前瞻性和回顾性研究设计", options: { bullet: true, breakLine: true, fontSize: 11 } },
  { text: "样本量计算需基于统计功效", options: { bullet: true, fontSize: 11 } },
], { x: 5.4, y: 1.6, w: 3.9, h: 2.2, fontFace: FONT_BODY, color: C.textBody, valign: "top", paraSpaceAfter: 4 });

// Bottom note
s33.addText("性能指标参考：敏感度 ≥ 90%  |  特异度 ≥ 85%  |  AUC ≥ 0.95  (具体指标由临床需求决定)", {
  x: 0.5, y: 4.2, w: 9, h: 0.35,
  fontSize: 10, fontFace: FONT_BODY, color: C.textMuted, italic: true, align: "center", margin: 0,
});

// --- Slide 3-4: 风险管理 ---
const s34 = contentSlide(3, "AI 医疗器械风险管理", "20");
s34.addText("基于 ISO 14971:2019，风险分析 → 风险评价 → 风险控制 → 综合剩余风险评价 → 风险管理报告", {
  x: 0.5, y: 0.85, w: 9, h: 0.3,
  fontSize: 10, fontFace: FONT_BODY, color: C.textMuted, italic: true, margin: 0,
});
const risks = [
  { risk: "算法偏差 (Bias)", impact: "误诊/漏诊", control: "多样化数据采集 + 亚组分析 + 公平性评估" },
  { risk: "数据集漂移", impact: "模型性能退化", control: "持续监测 + 自适应校准" },
  { risk: "对抗性攻击", impact: "安全性事件", control: "对抗训练 + 输入验证" },
  { risk: "黑箱问题", impact: "临床信任度低", control: "XAI 技术 + 决策路径可视化" },
  { risk: "过拟合/欠拟合", impact: "泛化能力不足", control: "独立测试集 + 交叉验证 + 正则化" },
  { risk: "概念漂移", impact: "模型逐渐失效", control: "定期重训练 + 触发式更新" },
  { risk: "人类过度依赖", impact: "自动化偏差", control: "界面设计 + 人机协同机制" },
];

// Risk table
const rHeaders = ["风险类别", "潜在影响", "控制措施"];
const rRows = risks.map(r => [
  { text: r.risk, options: { fontSize: 9, fontFace: FONT_BODY, color: C.textDark, bold: true, fill: { color: C.highlight }, valign: "middle" } },
  { text: r.impact, options: { fontSize: 9, fontFace: FONT_BODY, color: C.textBody, fill: { color: C.cardBg }, valign: "middle", align: "center" } },
  { text: r.control, options: { fontSize: 9, fontFace: FONT_BODY, color: C.textBody, fill: { color: C.cardBg }, valign: "middle" } },
]);

const tblData = [
  rHeaders.map(h => ({ text: h, options: { fontSize: 9, fontFace: FONT_BODY, color: C.white, bold: true, fill: { color: C.primary }, valign: "middle", align: "center" } })),
  ...rRows,
];
s34.addTable(tblData, {
  x: 0.5, y: 1.15, w: 9,
  colW: [2.0, 1.8, 5.2],
  rowH: [0.35, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
  border: { pt: 0.5, color: C.border },
});

// --- Slide 3-5: AI 素养 ---
const s35 = contentSlide(3, "AI 素养要求", "21");
// 4 quadrants
const literacyItems = [
  { title: "技术能力", items: ["ML/DL 理论基础", "模型开发框架", "数据科学能力", "AI 安全与鲁棒性"], color: C.primary },
  { title: "医疗器械专业", items: ["NMPA 注册要求", "ISO 13485 QMS", "ISO 14971 风险管理", "临床评价方法"], color: C.secondary },
  { title: "交叉学科能力", items: ["医学基础知识", "数据标注规范", "医学统计学", "临床路径理解"], color: C.accent },
  { title: "伦理与合规", items: ["AI 伦理原则", "数据隐私法规", "知情同意审查", "可解释性要求"], color: C.green },
];
literacyItems.forEach((item, i) => {
  const col = i % 2;
  const row = Math.floor(i / 2);
  const xB = 0.5 + col * 4.6;
  const yB = 1.0 + row * 2.0;
  s35.addShape(pres.shapes.RECTANGLE, {
    x: xB, y: yB, w: 4.3, h: 1.8,
    fill: { color: C.cardBg },
    shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
  });
  // Top accent
  s35.addShape(pres.shapes.RECTANGLE, {
    x: xB, y: yB, w: 4.3, h: 0.05,
    fill: { color: item.color },
  });
  s35.addText(item.title, {
    x: xB + 0.2, y: yB + 0.15, w: 3.9, h: 0.3,
    fontSize: 13, fontFace: FONT_TITLE, color: item.color, bold: true, margin: 0, valign: "middle",
  });
  s35.addText(item.items.map((t, j) => ({
    text: t,
    options: { bullet: true, breakLine: j < item.items.length - 1, fontSize: 11, color: C.textBody, paraSpaceAfter: 3 },
  })), {
    x: xB + 0.2, y: yB + 0.5, w: 3.9, h: 1.2,
    fontFace: FONT_BODY, valign: "top",
  });
});

// --- Slide 3-6: 质量管理 ---
const s36 = contentSlide(3, "质量管理体系建设要点", "22");
const qmsItems = [
  { title: "设计控制", desc: "AI 特有的设计输入（数据、算法、性能指标），建立完整的设计历史文档" },
  { title: "软件验证与确认", desc: "AI 模型验证的特殊性（统计方法、数据要求、独立测试集）" },
  { title: "变更管理", desc: "模型更新机制（PCCP 机制与传统变更控制的结合）" },
  { title: "供应商管理", desc: "第三方数据提供商、AI 算法供应商的质量控制与审核" },
  { title: "文档化要求", desc: "数据管理文档、模型开发文档、验证确认文档、风险管理文档、临床评价文档" },
];
qmsItems.forEach((item, i) => {
  const yB = 1.0 + i * 0.82;
  s36.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: yB, w: 0.08, h: 0.6,
    fill: { color: C.secondary },
  });
  s36.addText(item.title, {
    x: 0.75, y: yB, w: 2.2, h: 0.3,
    fontSize: 12, fontFace: FONT_TITLE, color: C.textDark, bold: true, margin: 0, valign: "middle",
  });
  s36.addText(item.desc, {
    x: 0.75, y: yB + 0.3, w: 8.5, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, color: C.textMuted, margin: 0, valign: "top",
  });
});

// --- Slide 3-7: 中国特色要求 ---
const s37 = contentSlide(3, "中国特色要求", "23");
const cnItems = [
  { icon: "🔒", title: "国产化和自主可控", desc: "AI 医疗器械核心技术自主创新要求，鼓励国产替代" },
  { icon: "🔬", title: "中检院评测", desc: "医疗器械注册检验中的 AI 性能评测，中检院 AI 实验室建设" },
  { icon: "🏷️", title: "UDI 唯一标识", desc: "AI 软件医疗器械的 UDI 追溯体系" },
  { icon: "🛡️", title: "网络安全等保", desc: "AI 医疗系统的网络安全等级保护要求" },
  { icon: "📊", title: "数据安全", desc: "《数据安全法》《个人信息保护法》对医疗健康数据的特殊要求" },
];
cnItems.forEach((item, i) => {
  const col = i % 3;
  const row = Math.floor(i / 3);
  const xB = 0.5 + col * 3.15;
  const yB = 1.0 + row * 1.95;
  s37.addShape(pres.shapes.RECTANGLE, {
    x: xB, y: yB, w: 2.9, h: 1.7,
    fill: { color: C.cardBg },
    shadow: { type: "outer", blur: 2, offset: 1, color: "000000", opacity: 0.05 },
  });
  s37.addText(item.icon, {
    x: xB, y: yB + 0.15, w: 2.9, h: 0.5,
    fontSize: 28, align: "center", margin: 0,
  });
  s37.addText(item.title, {
    x: xB + 0.15, y: yB + 0.6, w: 2.6, h: 0.3,
    fontSize: 12, fontFace: FONT_TITLE, color: C.textDark, bold: true, align: "center", margin: 0,
  });
  s37.addText(item.desc, {
    x: xB + 0.15, y: yB + 0.95, w: 2.6, h: 0.6,
    fontSize: 10, fontFace: FONT_BODY, color: C.textMuted, align: "center", margin: 0, valign: "top",
  });
});

// ============================================================
// SUMMARY / CLOSING SLIDE
// ============================================================
const sEnd = pres.addSlide();
sEnd.background = { color: C.darkBg };
sEnd.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.primary },
});
sEnd.addText("谢谢", {
  x: 0.8, y: 1.0, w: 8.4, h: 1.0,
  fontSize: 48, fontFace: FONT_TITLE, color: C.white, bold: true, align: "center", margin: 0,
});
sEnd.addShape(pres.shapes.LINE, {
  x: 4, y: 2.1, w: 2, h: 0,
  line: { color: C.primary, width: 2.5 },
});
sEnd.addText("基于 NMPA《人工智能医疗器械注册审查指导原则》", {
  x: 0.8, y: 2.5, w: 8.4, h: 0.4,
  fontSize: 13, fontFace: FONT_BODY, color: C.textMuted, align: "center", margin: 0,
});
sEnd.addText("及中美欧 AI 医疗器械法规体系整理", {
  x: 0.8, y: 2.9, w: 8.4, h: 0.4,
  fontSize: 13, fontFace: FONT_BODY, color: C.textMuted, align: "center", margin: 0,
});
sEnd.addText("参考来源", {
  x: 0.8, y: 3.6, w: 8.4, h: 0.3,
  fontSize: 11, fontFace: FONT_BODY, color: C.primary, align: "center", margin: 0,
});
sEnd.addText([
  { text: "NMPA / FDA / European Commission · 国务院 · ISO 14971 · WHO · IMDRF", options: { fontSize: 9, color: C.textMuted } },
], {
  x: 0.8, y: 3.9, w: 8.4, h: 0.3,
  fontFace: FONT_BODY, align: "center", margin: 0,
});

// ============================================================
// SAVE
// ============================================================
const outPath = process.argv[2] || "AI_Medical_Device_Design.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log(`✅ PPTX saved: ${outPath}`);
}).catch(err => {
  console.error("❌ Error:", err);
  process.exit(1);
});
