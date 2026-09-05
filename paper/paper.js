import { HybridPolicy } from "./hybrid.js";

// ---------------------------------------------------------------- paper UI
const panes = [...document.querySelectorAll(".formula-pane")];
document.querySelectorAll(".formula-tabs [role=tab]").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".formula-tabs [role=tab]").forEach((item) => {
      const active = item === tab;
      item.setAttribute("aria-selected", String(active));
      item.tabIndex = active ? 0 : -1;
    });
    panes.forEach((pane) => {
      const active = pane.id === `pane-${tab.dataset.pane}`;
      pane.hidden = !active;
      pane.classList.toggle("active", active);
    });
  });
});

document.querySelectorAll(".question").forEach((details) => {
  details.addEventListener("toggle", () => {
    if (!details.open) return;
    document.querySelectorAll(".question").forEach((other) => {
      if (other !== details) other.open = false;
    });
  });
});

const loopCopyEn = {
  rollout: ["32,768 transitions", "256 parallel environments sample 128 frames each; training samples, display uses argmax."],
  gae: ["GAE(γ=.999, λ=.95)", "Bootstrap from the last state, then compute advantages in reverse."],
  ppo: ["32 optimizer steps", "4 epochs × 8 minibatches; clipped update, gradient norm ≤ 0.5."],
  league: ["10% / 10% / 80%", "Laika / Killfield / seven frozen ancestors, redrawn every round."],
  publish: ["Atomic snapshot", "Checkpoint, schema and web weights ship as one checkable version."],
};
const loopCopy = {
  rollout: ["32,768 transitions", "256 个并行环境各采样 128 帧；训练 sample，展示 argmax。"],
  gae: ["GAE(γ=.999, λ=.95)", "从最后状态 bootstrap，再逆序计算 advantage。"],
  ppo: ["32 optimizer steps", "4 个 epoch × 8 minibatch；clip 更新，梯度范数 ≤ 0.5。"],
  league: ["10% / 10% / 80%", "Laika / Killfield / 七个冻结祖先，每局重新抽取。"],
  publish: ["Atomic snapshot", "checkpoint、schema 与网页权重作为一个可校验版本发布。"],
};
document.querySelectorAll(".training-loop button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".training-loop button").forEach((item) => item.classList.toggle("active", item === button));
    const [title, body] = (lang === "en" ? loopCopyEn : loopCopy)[button.dataset.loop];
    document.querySelector("#loop-output b").textContent = title;
    document.querySelector("#loop-output span").textContent = body;
  });
});

const problemCopyEn = {
  twitch: ["The terminal style score must travel ~200 frames.", "Entropy pushes the sample every single frame; the gradient paths are asymmetric."],
  idle: ["Switching fell, but idling rose to 48%.", "Holding an action is easiest to satisfy by doing nothing — a new local optimum."],
  wall: ["Plain persistence cannot tell persisting from being stuck.", "Only when the action really pushes into a wall should the wall loss produce gradient."],
  dodge: ["A feedforward net sees danger but cannot compare nine actions.", "Give every movement a same-scale counterfactual, and align it explicitly at the actor."],
  ammo: ["One bad shot occupies a slot for a long time.", "The terminal result is too far from the trigger; hit ETA and ammo scarcity must enter action quality."],
  league: ["A new policy counters its parent but forgets its grandparents.", "Freezing several generations preserves pressure on skills already learned."],
};
const problemCopy = {
  twitch: ["终局平滑分需要穿过约 200 帧。", "entropy 却在每一帧直接推动采样；梯度路径不对称。"],
  idle: ["切换率下降，原地率却升到 48%。", "保持动作最容易在“什么都不做”上满足，产生新的局部最优。"],
  wall: ["一般 persistence 无法区分坚持与无效。", "只有动作确实朝墙施力时，wall loss 才应产生梯度。"],
  dodge: ["前馈网络看见危险，却难比较九种动作。", "为每个移动提供同口径反事实，并在 actor 端显式对齐。"],
  ammo: ["一次坏射击会占用弹槽很久。", "终局输赢距离开火太远；命中 ETA 与弹药稀缺度需要进入动作质量。"],
  league: ["新策略能克制父代，却忘记祖代。", "冻结多代对手保留过去学会的能力压力。"],
};
document.querySelectorAll(".problem-board button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".problem-board button").forEach((item) => item.classList.toggle("active", item === button));
    const [title, body] = (lang === "en" ? problemCopyEn : problemCopy)[button.dataset.problem];
    document.querySelector("#problem-output b").textContent = title;
    document.querySelector("#problem-output span").textContent = body;
  });
});

const logitStrip = document.querySelector("#logit-strip");
for (let i = 0; i < 18; i += 1) {
  const bar = document.createElement("i");
  bar.dataset.action = String(i);
  logitStrip.append(bar);
}

// ------------------------------------------------------------------- i18n
// Chinese is the authored language and stays in the DOM. English lives in
// data-en / data-en-html / data-en-meaning / data-en-signal / data-en-label /
// data-tex-en, and the first switch stashes the Chinese so it can come back.
let lang = "zh";

function swapText(node, attr, prop) {
  const other = node.getAttribute(attr);
  if (other === null) return;
  const stash = `zh${prop === "innerHTML" ? "Html" : "Text"}`;
  if (node.dataset[stash] === undefined) node.dataset[stash] = node[prop];
  node[prop] = lang === "en" ? other : node.dataset[stash];
}

function applyLanguage() {
  document.documentElement.lang = lang === "en" ? "en" : "zh-CN";
  document.body.dataset.lang = lang;
  document.querySelectorAll("[data-en]").forEach((n) => swapText(n, "data-en", "textContent"));
  document.querySelectorAll("[data-en-html]").forEach((n) => swapText(n, "data-en-html", "innerHTML"));
  // Observation table: signal name lives in <th>'s first text node, so the
  // dimension <small> beside it survives the swap.
  document.querySelectorAll("[data-en-signal]").forEach((row) => {
    const th = row.querySelector("th");
    if (!th?.firstChild) return;
    if (row.dataset.zhSignal === undefined) row.dataset.zhSignal = th.firstChild.textContent;
    th.firstChild.textContent = lang === "en" ? row.dataset.enSignal : row.dataset.zhSignal;
  });
  document.querySelectorAll("[data-en-meaning]").forEach((row) => {
    if (row.dataset.zhMeaning === undefined) row.dataset.zhMeaning = row.dataset.meaning;
    row.dataset.meaning = lang === "en" ? row.dataset.enMeaning : row.dataset.zhMeaning;
  });
  document.querySelectorAll("[data-en-label]").forEach((node) => {
    if (node.dataset.zhLabel === undefined) node.dataset.zhLabel = node.dataset.label;
    node.dataset.label = lang === "en" ? node.dataset.enLabel : node.dataset.zhLabel;
  });
  // The authored TeX lives in data-tex-zh / data-tex-en, never read back out
  // of the DOM: KaTeX replaces the node's contents on first render. Compact
  // inline formulas (table cells, the legend) carry .tex-inline and render
  // in text mode instead of display mode.
  document.querySelectorAll("[data-tex-zh]").forEach((node) => {
    const tex = (lang === "en" && node.dataset.texEn) || node.dataset.texZh;
    const displayMode = !node.classList.contains("tex-inline");
    if (window.katex) katex.render(tex, node, { displayMode, throwOnError: false });
    else node.textContent = displayMode ? `$$${tex}$$` : `\(${tex}\)`;
  });
  // The toggle shows the language it switches *to*, not the current one.
  const toggle = document.querySelector("#lang-toggle");
  if (toggle) toggle.textContent = lang === "en" ? "中" : "EN";
  // The two output panels hold prose picked by a button, so re-render whatever
  // is currently selected rather than leaving the authored HTML behind.
  const loopKey = document.querySelector(".training-loop button.active")?.dataset.loop;
  if (loopKey) {
    const [title, body] = (lang === "en" ? loopCopyEn : loopCopy)[loopKey];
    document.querySelector("#loop-output b").textContent = title;
    document.querySelector("#loop-output span").textContent = body;
  }
  const problemKey = document.querySelector(".problem-board button.active")?.dataset.problem;
  if (problemKey) {
    const [title, body] = (lang === "en" ? problemCopyEn : problemCopy)[problemKey];
    document.querySelector("#problem-output b").textContent = title;
    document.querySelector("#problem-output span").textContent = body;
  }

  // Re-render anything the live loop owns, in the new language.
  if (latestFrame) { updateActionUI(latestFrame); }
  else { document.querySelectorAll(".obs-table .live-col").forEach((c) => { c.textContent = ""; }); }
  refreshPointedObservation();
  document.querySelector("#live-opponent").textContent = OPPONENT_LABEL[opponent];
}

document.querySelector("#lang-toggle")?.addEventListener("click", () => {
  lang = lang === "en" ? "zh" : "en";
  try { localStorage.setItem("paper-lang", lang); } catch {}
  applyLanguage();
});

const t = (zh, en) => (lang === "en" ? en : zh);

let latestFrame = null;

const MAP_H = 10;
const MAP_CHANNELS = 7;
const THROTTLE_ZH = ["后退", "中性", "前进"], THROTTLE_EN = ["reverse", "neutral", "forward"];
const TURN_ZH = ["左转", "不转", "右转"], TURN_EN = ["left", "straight", "right"];
const throttleName = (i) => (lang === "en" ? THROTTLE_EN : THROTTLE_ZH)[i];
const turnName = (i) => (lang === "en" ? TURN_EN : TURN_ZH)[i];
const moveName = (i) => `${throttleName(Math.floor(i / 3)) ?? "—"} · ${turnName(i % 3) ?? "—"}`;

const num = (v) => (Number.isFinite(v) ? (Math.abs(v) < .0005 ? "0" : v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "")) : "—");
const bit = (v) => (v > .5 ? t("是", "yes") : t("否", "no"));
const oneHot = (slice, names) => names[slice.indexOf(Math.max(...slice))] ?? "—";

// Each formatter turns a real slice of the live observation into rows the
// table can show. Long vectors summarise instead of dumping every number.
const observationFields = {
  map: (o) => {
    let self = null, enemy = null;
    for (let i = 0; i < 120; i += 1) {
      const base = i * MAP_CHANNELS;
      if (o[base + 5] > .5) self = i;
      if (o[base + 6] > .5) enemy = i;
    }
    const cell = (i) => (i === null ? null : `(${Math.floor(i / MAP_H)}, ${i % MAP_H})`);
    const rows = [[t("我方格", "my cell"), cell(self) ?? "—"], [t("敌方格", "enemy cell"), cell(enemy) ?? "—"]];
    if (self !== null) {
      const b = self * MAP_CHANNELS;
      rows.push([t("我方格 7 通道", "my cell, 7 channels"), `[${[0,1,2,3,4,5,6].map((k) => num(o[b + k])).join(", ")}]`]);
      rows.push(["", t("valid, 上, 右, 下, 左, self, enemy", "valid, N, E, S, W, self, enemy")]);
    }
    return { rows, note: t("840 个值 · 表内只展开两辆坦克所在格，其余 118 格省略 …", "840 values · only the two occupied cells are expanded; the other 118 are elided …") };
  },
  rays: (o) => {
    const v = Array.from(o);
    const near = v.indexOf(Math.min(...v));
    return {
      rows: [
        [t("最近墙", "nearest wall"), t(`第 ${near} 条 · ${num(v[near])}（约 ${num(v[near] * 4)} 格）`, `ray ${near} · ${num(v[near])} (~${num(v[near] * 4)} cells)`)],
        [t("前 8 条", "first 8"), `[${v.slice(0, 8).map(num).join(", ")} …]`],
        [t("均值", "mean"), num(v.reduce((a, b) => a + b, 0) / v.length)],
      ],
      note: t("16 条射线 · 四格范围内归一化", "16 rays · normalised over four cells"),
    };
  },
  self: (o) => ({
    rows: [
      [t("归一化位置", "normalised position"), `x ${num(o[0])} · y ${num(o[1])}`],
      [t("朝向 cos/sin", "heading cos/sin"), `${num(o[2])} · ${num(o[3])}`],
      [t("体轴速度", "body velocity"), t(`前 ${num(o[4])} · 左 ${num(o[5])}`, `fwd ${num(o[4])} · left ${num(o[5])}`)],
      [t("角速度", "angular rate"), num(o[6])],
      [t("剩余弹槽", "ammo slots"), `${Math.round(o[7] * 5)} / 5`],
      [t("可射 / 存活", "ready / alive"), `${bit(o[8])} / ${bit(o[9])}`],
      [t("硬碰撞 / 滑墙", "collision / wall slide"), `${bit(o[10])} / ${bit(o[11])}`],
    ],
  }),
  opponent: (o) => ({
    rows: [
      [t("相对位置", "relative position"), t(`前 ${num(o[0])} · 左 ${num(o[1])}`, `fwd ${num(o[0])} · left ${num(o[1])}`)],
      [t("相对朝向", "relative heading"), `cos ${num(o[2])} · sin ${num(o[3])}`],
      [t("相对速度", "relative velocity"), t(`前 ${num(o[4])} · 左 ${num(o[5])}`, `fwd ${num(o[4])} · left ${num(o[5])}`)],
      [t("角速度", "angular rate"), num(o[6])],
      [t("剩余弹槽", "ammo slots"), `${Math.round(o[7] * 5)} / 5`],
      [t("可射 / 存活", "ready / alive"), `${bit(o[8])} / ${bit(o[9])}`],
      [t("硬碰撞 / 滑墙", "collision / wall slide"), `${bit(o[10])} / ${bit(o[11])}`],
    ],
    note: t("全部在我方车体坐标系中", "all in my own body frame"),
  }),
  navigation: (o) => ({
    rows: [
      [t("BFS 路径长度", "BFS path length"), num(o[0])],
      [t("下一步方向", "next step"), o.slice(1, 5).some((v) => v > .5) ? oneHot(Array.from(o.slice(1, 5)), lang === "en" ? ["fwd", "left", "back", "right"] : ["前", "左", "后", "右"]) : t("无", "none")],
      [t("直线距离", "straight-line distance"), num(o[5])],
      [t("方位 cos/sin", "bearing cos/sin"), `${num(o[6])} · ${num(o[7])}`],
      [t("死胡同（我 / 敌）", "dead-end (me / them)"), `${num(o[8])} · ${num(o[9])}`],
    ],
  }),
  aim: (o) => ({
    rows: [
      [t("当前炮口结果", "current barrel"), oneHot(Array.from(o.slice(0, 3)), lang === "en" ? ["hit", "suicide", "miss"] : ["命中", "自杀", "落空"])],
      [t("命中时间", "time to hit"), num(o[3])],
      [t("最近擦过", "closest pass"), num(o[4])],
    ],
    note: t("只模拟当前炮口角度，不搜索其他角度", "simulates the current barrel angle only; no search over other angles"),
  }),
  aimOpponent: (o) => ({
    rows: [
      [t("对手炮口结果", "their barrel"), oneHot(Array.from(o.slice(0, 3)), lang === "en" ? ["hits me", "suicide", "miss"] : ["命中我", "自杀", "落空"])],
      [t("命中时间", "time to hit"), num(o[3])],
      [t("最近擦过", "closest pass"), num(o[4])],
    ],
  }),
  bullets: (o, frame) => {
    const mask = frame?.mask ?? [];
    const rows = [];
    let empty = 0;
    for (let slot = 0; slot < 10; slot += 1) {
      if (!mask[slot]) { empty += 1; continue; }
      const b = slot * 10;
      rows.push([`b${slot}`, t(`位置 ${num(o[b])},${num(o[b + 1])} · 速度 ${num(o[b + 2])},${num(o[b + 3])} · ${o[b + 4] > .5 ? "我的" : "敌的"}${o[b + 5] > .5 ? " · 已反弹" : ""}`, `pos ${num(o[b])},${num(o[b + 1])} · vel ${num(o[b + 2])},${num(o[b + 3])} · ${o[b + 4] > .5 ? "mine" : "theirs"}${o[b + 5] > .5 ? " · bounced" : ""}`)]);
      rows.push(["", t(`寿命 ${num(o[b + 6])} · 将中我 ${bit(o[b + 7])} · 将中敌 ${bit(o[b + 8])} · ETA ${num(o[b + 9])}`, `life ${num(o[b + 6])} · hits me ${bit(o[b + 7])} · hits them ${bit(o[b + 8])} · ETA ${num(o[b + 9])}`)]);
    }
    if (!rows.length) rows.push(["", t("场上没有活动子弹", "no bullets in flight")]);
    return { rows, note: t(`10 槽 × 10 维 = 100 值 · ${empty} 个空槽折叠 …`, `10 slots × 10 values = 100 · ${empty} empty slots folded …`) };
  },
  threat: (o) => ({
    rows: [
      [t("我方 incoming risk", "my incoming risk"), num(o[0])],
      [t("对手 incoming risk", "their incoming risk"), num(o[1])],
      [t("最近擦过距离", "closest pass distance"), num(o[2])],
    ],
  }),
  phase: (o) => ({
    rows: [
      [t("阶段", "phase"), oneHot(Array.from(o.slice(0, 3)), lang === "en" ? ["fighting", "settling", "frozen"] : ["对战", "结算", "冻结"])],
      [t("回合时钟", "round clock"), num(o[3])],
    ],
  }),
  lastAction: (o) => ({
    rows: [[t("t−1 油门", "t−1 throttle"), `${num(o[0])} · ${throttleName(Math.round(o[0] * 2)) ?? "—"}`],
           [t("t−1 转向", "t−1 turn"), `${num(o[1])} · ${turnName(Math.round(o[1] * 2)) ?? "—"}`],
           [t("t−1 开火", "t−1 fire"), bit(o[2])]],
  }),
  threatCount: (o) => ({ rows: [[t("会命中我的子弹", "bullets that reach me"), `${Math.round(o[0] * 10)} / 10`]] }),
  olderActions: (o) => ({
    rows: [
      ["t−2", `${throttleName(Math.round(o[0] * 2)) ?? "—"} · ${turnName(Math.round(o[1] * 2)) ?? "—"} · ${t("开火","fire")} ${bit(o[2])}`],
      ["t−3", `${throttleName(Math.round(o[3] * 2)) ?? "—"} · ${turnName(Math.round(o[4] * 2)) ?? "—"} · ${t("开火","fire")} ${bit(o[5])}`],
    ],
  }),
  changeRate: (o) => ({
    rows: [[t("本回合变更率", "change rate this round"), `${(o[0] * 100).toFixed(1)}%`], [t("平滑分参考线", "style anchor"), "13%"]],
  }),
  dodge: (o) => {
    const v = Array.from(o);
    let best = 0;
    for (let i = 1; i < v.length; i += 1) if (v[i] > v[best]) best = i;
    const spread = Math.max(...v) - Math.min(...v);
    return {
      rows: [
        // With no threat on the board every candidate scores the same, and
        // naming one of them "safest" would read as a recommendation.
        [t("最安全移动", "safest movement"), spread < 1e-6 ? t(`九个候选同分 ${num(v[0])} · 无差别`, `all nine tie at ${num(v[0])} · no difference`) : `${moveName(best)} · ${num(v[best])}`],
        [t("预计死亡", "predicted death"), t(`${v.filter((x) => x < 0).length} / 9 个候选`, `${v.filter((x) => x < 0).length} / 9 candidates`)],
        [t("九值", "nine values"), `[${v.map(num).join(", ")}]`],
      ],
      note: t("每个候选独立滚动 24 帧真实物理，只回答能否活下来", "each candidate is rolled 24 frames through the real physics; it answers survival only"),
    };
  },
  idle: (o) => ({ rows: [[t("连续原地", "consecutive idle"), `${Math.round(o[0] * 25)} / 25 ${t("帧", "frames")}`]] }),
  state: (o) => ({
    rows: [
      [t("墙射线 840","wall rays 840"), `min ${num(Math.min(...o.subarray(0, 16)))}`],
      [t("自身 856","self 856"), t(`位置 ${num(o[16])},${num(o[17])} · 弹槽 ${Math.round(o[23] * 5)}/5`, `pos ${num(o[16])},${num(o[17])} · ammo ${Math.round(o[23] * 5)}/5`)],
      [t("对手 868","opponent 868"), t(`相对 前 ${num(o[28])} 左 ${num(o[29])}`, `rel fwd ${num(o[28])} lft ${num(o[29])}`)],
      [t("导航 880","navigation 880"), `path ${num(o[40])} · 直线 ${num(o[45])}`],
      [t("瞄准 890 / 895","aim 890 / 895"), `${oneHot(Array.from(o.subarray(50, 53)), lang === "en" ? ["hit","suicide","miss"] : ["命中","自杀","落空"])} / ${oneHot(Array.from(o.subarray(55, 58)), lang === "en" ? ["hits me","suicide","miss"] : ["命中我","自杀","落空"])}`],
    ],
    note: t("60 个标量 · 逐字段请看下方表格","60 scalars · see the table below for each field"),
  }),
  history: (o) => ({
    rows: [
      [t("威胁 1000","threat 1000"), t(`我 ${num(o[0])} · 敌 ${num(o[1])}`, `me ${num(o[0])} · them ${num(o[1])}`)],
      [t("阶段 1003","phase 1003"), oneHot(Array.from(o.subarray(3, 6)), lang === "en" ? ["fighting","settling","frozen"] : ["对战","结算","冻结"])],
      ["t−1 1007", `${throttleName(Math.round(o[7] * 2)) ?? "—"} · ${turnName(Math.round(o[8] * 2)) ?? "—"}`],
      [t("变更率 1017","change rate 1017"), `${(o[17] * 100).toFixed(1)}%`],
      ["dodge 1018", `best ${num(Math.max(...o.subarray(18, 27)))}`],
      [t("原地 1027","idle 1027"), `${Math.round(o[27] * 25)}/25`],
    ],
    note: t("28 个附加通道 · 逐字段请看下方表格","28 appended channels · see the table below for each field"),
  }),
};

// One compact number per row, refreshed every frame, so the table itself
// reads as a live view of the observation rather than a static schema.
const liveSummary = {
  map: (o) => { for (let i = 0; i < 120; i += 1) if (o[i * MAP_CHANNELS + 5] > .5) return `self (${Math.floor(i / MAP_H)},${i % MAP_H})`; return "—"; },
  rays: (o) => `min ${num(Math.min(...o))}`,
  self: (o) => `ammo ${Math.round(o[7] * 5)}/5`,
  opponent: (o) => t(`前 ${num(o[0])} 左 ${num(o[1])}`, `fwd ${num(o[0])} lft ${num(o[1])}`),
  navigation: (o) => `path ${num(o[0])}`,
  aim: (o) => oneHot(Array.from(o.slice(0, 3)), lang === "en" ? ["hit", "suicide", "miss"] : ["命中", "自杀", "落空"]),
  aimOpponent: (o) => oneHot(Array.from(o.slice(0, 3)), lang === "en" ? ["hits me", "suicide", "miss"] : ["命中我", "自杀", "落空"]),
  bullets: (o, f) => t(`${(f?.mask ?? []).filter(Boolean).length} 发在飞`, `${(f?.mask ?? []).filter(Boolean).length} in flight`),
  threat: (o) => `risk ${num(o[0])}`,
  phase: (o) => `${oneHot(Array.from(o.slice(0, 3)), lang === "en" ? ["fighting", "settling", "frozen"] : ["对战", "结算", "冻结"])} ${num(o[3])}`,
  lastAction: (o) => `${throttleName(Math.round(o[0] * 2)) ?? "—"}·${turnName(Math.round(o[1] * 2)) ?? "—"}`,
  threatCount: (o) => `${Math.round(o[0] * 10)}/10`,
  olderActions: (o) => `t−2 ${throttleName(Math.round(o[0] * 2)) ?? "—"}`,
  changeRate: (o) => `${(o[0] * 100).toFixed(1)}%`,
  dodge: (o) => `best ${num(Math.max(...o))}`,
  idle: (o) => `${Math.round(o[0] * 25)}/25`,
};

const liveCells = [...document.querySelectorAll(".obs-table tr[data-obs]")].map((row) => ({
  key: row.dataset.obs,
  off: Number(row.dataset.off),
  dim: Number(row.dataset.dim),
  cell: row.querySelector(".live-col"),
}));

function updateLiveColumn() {
  const obs = latestFrame?.obs;
  if (!obs) return;
  liveCells.forEach(({ key, off, dim, cell }) => {
    if (!cell) return;
    const text = liveSummary[key]?.(obs.subarray(off, off + dim), latestFrame);
    if (text !== undefined && cell.textContent !== text) cell.textContent = text;
  });
}

function renderObservation(key, row) {
  const readout = document.querySelector("#obs-readout");
  const meta = (row?.dataset.off !== undefined ? row : null)
    ?? document.querySelector(`.obs-table tr[data-obs="${key}"]`)
    ?? document.querySelector(`.obs-vector [data-obs="${key}"]`);
  const off = Number(meta?.dataset.off);
  const dim = Number(meta?.dataset.dim);
  const title = meta?.dataset.label ?? meta?.querySelector("th")?.firstChild?.textContent?.trim() ?? key;
  const meaning = meta?.dataset.meaning ?? "";
  const obs = latestFrame?.obs;
  if (!obs || !Number.isInteger(off)) {
    readout.innerHTML = "";
    readout.append(Object.assign(document.createElement("b"), { textContent: `${title} · ${dim || "?"} 维` }),
                   Object.assign(document.createElement("span"), { textContent: t("等待实况产生第一帧观测…", "waiting for the first live observation…") }));
    return;
  }
  const slice = obs.subarray(off, off + dim);
  const { rows, note } = observationFields[key]?.(slice, latestFrame) ?? { rows: [["值", `[${Array.from(slice.subarray(0, 8)).map(num).join(", ")} …]`]] };
  readout.innerHTML = "";
  readout.append(Object.assign(document.createElement("b"), { textContent: `${title} · obs[${off} … ${off + dim - 1}] · ${dim}${t(" 维", "-d")}` }));
  if (meaning) readout.append(Object.assign(document.createElement("span"), { textContent: meaning }));
  const list = document.createElement("dl");
  rows.forEach(([label, value]) => {
    const wrap = document.createElement("div");
    wrap.append(Object.assign(document.createElement("dt"), { textContent: label }),
                Object.assign(document.createElement("dd"), { textContent: value }));
    list.append(wrap);
  });
  readout.append(list);
  if (note) readout.append(Object.assign(document.createElement("small"), { textContent: note }));
}

let pinnedObs = null;

function showObservation(key, row = null) {
  const covers = (row?.dataset.covers ?? "").split(" ").filter(Boolean);
  document.querySelectorAll(".obs-table tr").forEach((item) =>
    item.classList.toggle("focused", item === row || covers.includes(item.dataset.obs)));
  renderObservation(key, row);
  const overlay = document.querySelector("#stage-explain");
  const name = row?.dataset.label ?? row?.querySelector("th")?.firstChild?.textContent?.trim() ?? key;
  overlay.textContent = t(`${name} · 已在地图上标出`, `${name} · highlighted on the map`);
  overlay.hidden = false;
  document.body.dataset.obsGroup = key;
}

function clearObservation() {
  if (pinnedObs) return;
  document.querySelectorAll(".obs-table tr").forEach((item) => item.classList.remove("focused"));
  document.querySelector("#stage-explain").hidden = true;
  delete document.body.dataset.obsGroup;
}

document.querySelectorAll("[data-obs]").forEach((item) => {
  item.tabIndex = 0;
  const key = item.dataset.obs;
  const row = (item.matches("tr") || item.dataset.off !== undefined) ? item : null;
  item.addEventListener("mouseenter", () => { if (!pinnedObs) showObservation(key, row); });
  item.addEventListener("focus", () => { if (!pinnedObs) showObservation(key, row); });
  item.addEventListener("mouseleave", clearObservation);
  item.addEventListener("blur", clearObservation);
  // Click pins the layer so it stays readable on touch screens.
  item.addEventListener("click", () => {
    pinnedObs = pinnedObs === key ? null : key;
    if (pinnedObs) showObservation(key, row); else { clearObservation(); showObservation(key, row); }
    item.classList.toggle("pinned", pinnedObs === key);
    document.querySelectorAll("[data-obs]").forEach((other) => { if (other !== item) other.classList.remove("pinned"); });
  });
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || !pinnedObs) return;
  pinnedObs = null;
  document.querySelectorAll("[data-obs]").forEach((item) => item.classList.remove("pinned"));
  clearObservation();
});

// Keep whatever the reader is pointing at in sync with the live frame.
function refreshPointedObservation() {
  const active = pinnedObs ?? document.querySelector(".obs-table tr.focused")?.dataset.obs;
  if (active) renderObservation(active, document.querySelector(`.obs-table tr[data-obs="${active}"]`));
}

function softmax(values) {
  const max = Math.max(...values);
  const exp = values.map((v) => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
}

function updateActionUI(detail) {
  latestFrame = detail;
  const action = detail.action;
  if (!Number.isInteger(action)) return;
  const throttle = Math.floor(action / 6);
  const turn = Math.floor(action / 2) % 3;
  const fire = action % 2 === 1;
  const movement = throttle * 3 + turn;

  // Key-style highlight: exactly the controls the policy is holding this frame.
  const keys = { forward: throttle === 2, back: throttle === 0, left: turn === 0, right: turn === 2, fire };
  document.querySelectorAll("#pane-action .key").forEach((node) => node.classList.toggle("active", !!keys[node.dataset.key]));
  document.querySelectorAll("#action-grid [data-move]").forEach((item) => item.classList.toggle("active", Number(item.dataset.move) === movement));
  document.querySelector("#action-index").textContent = `a = ${action}`;
  document.querySelector("#move-readout").textContent = `${moveName(movement)}${fire ? ` · ${t("开火", "fire")}` : ""}`;

  if (detail.logits?.length === 18) {
    const min = Math.min(...detail.logits);
    const max = Math.max(...detail.logits);
    const range = Math.max(.001, max - min);
    const probs = softmax(detail.logits);
    logitStrip.querySelectorAll("i").forEach((bar, i) => {
      bar.style.setProperty("--h", `${8 + 92 * (detail.logits[i] - min) / range}%`);
      bar.classList.toggle("active", i === action);
      bar.title = `a=${i} · ${moveName(Math.floor(i / 2))}${i % 2 ? ` · ${t("开火", "fire")}` : ""} · p ${(probs[i] * 100).toFixed(1)}%`;
    });
    document.querySelector("#logit-readout").textContent = `argmax a=${action} · logit ${detail.logits[action].toFixed(2)} · p ${(probs[action] * 100).toFixed(1)}%`;
  }
  updateLiveColumn();
  refreshPointedObservation();
}
window.addEventListener("hybrid-frame", (event) => updateActionUI(event.detail));

// --------------------------------------------------------------- live match
const FPS = 25;
const STEP_MS = 1000 / FPS;
const HEADER = 18 + 12 * 10;
const OBS_DIM = 1028;
const BULLET_SLOTS = 10;
const DODGE_OFFSET = 1018;
const DODGE_DIM = 9;
const KILLFIELD_RAYS = 512;
const MAX_CATCHUP_MS = 250;
const BULLET_SPEED = 4.5; // px/frame at scale 50, matching the engine
const canvas = document.querySelector("#live-canvas");
const ctx = canvas.getContext("2d");
let wasm = null;
let policy = null;
let handle = null;
let probeMpc = null;
let probeHybrid = null;
let probeBaseline = null;
let opponent = "laika";
const OPPONENT_LABEL_ALL = { laika:"Laika", killfield:"Killfield", hybrid:{zh:"Hybrid（自博弈）", en:"Hybrid (self-play)"} };
const OPPONENT_LABEL = new Proxy({}, { get: (_, k) => { const v = OPPONENT_LABEL_ALL[k]; return typeof v === "string" ? v : v?.[lang]; } });
let paused = false;
let accumulator = 0;
let previousTime = performance.now();
let previousState = null;
let lastAction = null;
let lastLogits = null;
const latency = { hybrid: [], killfield: [] };

function assetBase() {
  if (location.hostname.endsWith("github.io")) return new URL("../viewer/", document.baseURI);
  return new URL("https://cichlider.github.io/killfield/viewer/");
}

function samples(name, value) {
  if (!Number.isFinite(value) || value < 0) return;
  latency[name].push(value);
  if (latency[name].length > 60) latency[name].shift();
  const sorted = [...latency[name]].sort((a, b) => a - b);
  const mean = latency[name].reduce((a, b) => a + b, 0) / latency[name].length;
  const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * .95))];
  document.querySelector(`#latency-${name}`).textContent = `${mean.toFixed(2)} ms`;
  document.querySelector(`#latency-${name}-p95`).textContent = `p95 ${p95.toFixed(2)} · n ${latency[name].length}`;
}

function renderBuffer(target = handle) {
  const ptr = wasm.kf_render_ptr(target);
  const len = wasm.kf_render_len(target);
  return new Float32Array(wasm.memory.buffer, ptr, len);
}

function capture(buf) {
  const walls = buf[5] | 0;
  const tanks = buf[6] | 0;
  const bullets = buf[7] | 0;
  const tankBase = HEADER + walls * 4;
  const bulletBase = tankBase + tanks * 6;
  return {
    round: buf[9],
    tanks: Array.from({ length:tanks }, (_, i) => ({ x:buf[tankBase+i*6], y:buf[tankBase+i*6+1], rotation:buf[tankBase+i*6+2] })),
    bullets: Array.from({ length:bullets }, (_, i) => ({ x:buf[bulletBase+i*2], y:buf[bulletBase+i*2+1] })),
  };
}

// Canvas sizing, ported from the viewer's Renderer. Re-assigning canvas.width
// clears the surface, and getBoundingClientRect().width is sub-pixel, so
// recomputing it every frame makes the picture flicker. Check on a slow tick
// with a one-pixel tolerance instead.
let sizeCheckTick = 0;
function syncCanvasSize(force = false) {
  if (!force && sizeCheckTick++ % 15 !== 0) return;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const want = Math.max(1, Math.round(rect.width * dpr));
  if (!force && Math.abs(want - canvas.width) <= 1) return;
  canvas.width = want;
  canvas.height = Math.max(1, Math.round(want * 500 / 712));
}

function drawTank(x, y, rotation, scale, role) {
  const th = rotation * Math.PI / 180;
  const c = Math.cos(th), s = Math.sin(th);
  const point = (lx, ly) => [x + scale * (lx*c - ly*s), y + scale * (lx*s + ly*c)];
  const polygon = (points) => {
    ctx.beginPath();
    points.forEach(([lx, ly], i) => {
      const [px, py] = point(lx, ly);
      if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    });
    ctx.closePath();
  };
  const colors = role === 0 ? { base:"#a62830", turret:"#cf3f46" } : { base:"#171816", turret:"#373934" };
  polygon([[-30.5,-40.5],[30.5,-40.5],[30.5,40.5],[-30.5,40.5]]);
  ctx.fillStyle = colors.base; ctx.fill(); ctx.strokeStyle="#080908"; ctx.lineWidth=1.4; ctx.stroke();
  ctx.fillStyle="#080908";
  polygon([[-30.5,-40.5],[-22,-40.5],[-22,40.5],[-30.5,40.5]]); ctx.fill();
  polygon([[22,-40.5],[30.5,-40.5],[30.5,40.5],[22,40.5]]); ctx.fill();
  polygon([[-8.5,0],[8.5,0],[8.5,-55],[-8.5,-55]]);
  ctx.fillStyle=colors.turret;ctx.fill();ctx.stroke();
  const [cx,cy]=point(0,0);ctx.beginPath();ctx.arc(cx,cy,23.5*scale,0,Math.PI*2);ctx.fill();ctx.stroke();
}

function interpolateAngle(a, b, t) {
  let d = (b - a) % 360;
  if (d > 180) d -= 360;
  if (d < -180) d += 360;
  return a + d * t;
}

function drawObservationOverlay(key, buf, ox, oy, scale, mazeW, mazeH, tankBase, bulletBase, bullets) {
  if (!key) return;
  const obs = latestFrame?.obs;
  ctx.save();
  const TEAL="rgba(49,95,98,.9)", RED="rgba(166,40,48,.9)", GOLD="rgba(176,137,62,.95)";
  ctx.strokeStyle=TEAL;ctx.fillStyle="rgba(49,95,98,.12)";ctx.lineWidth=1.6;
  const tank=(i)=>({x:ox+buf[tankBase+i*6],y:oy+buf[tankBase+i*6+1],r:buf[tankBase+i*6+2]});
  const me=tank(0),them=tank(1);
  const ring=(t,color)=>{ctx.strokeStyle=color;ctx.beginPath();ctx.arc(t.x,t.y,scale*.46,0,Math.PI*2);ctx.stroke();};
  // Barrel points along rotation-90deg, matching the engine's facing convention.
  const facing=(t)=>(t.r-90)*Math.PI/180;

  if(key==="map"){
    ctx.setLineDash([4,4]);ctx.strokeStyle="rgba(49,95,98,.5)";
    for(let x=0;x<=mazeW;x++){ctx.beginPath();ctx.moveTo(ox+x*scale,oy);ctx.lineTo(ox+x*scale,oy+mazeH*scale);ctx.stroke();}
    for(let y=0;y<=mazeH;y++){ctx.beginPath();ctx.moveTo(ox,oy+y*scale);ctx.lineTo(ox+mazeW*scale,oy+y*scale);ctx.stroke();}
    ctx.setLineDash([]);
    [[me,TEAL],[them,RED]].forEach(([t,c])=>{
      const cx=Math.floor((t.x-ox)/scale),cy=Math.floor((t.y-oy)/scale);
      ctx.strokeStyle=c;ctx.lineWidth=2.4;ctx.strokeRect(ox+cx*scale,oy+cy*scale,scale,scale);
    });
  } else if(key==="rays"&&obs){
    // Real ray lengths: each value is the normalised distance over four cells.
    for(let i=0;i<16;i++){
      const a=facing(me)+i*Math.PI/8;
      const len=obs[840+i]*4*scale;
      ctx.strokeStyle="rgba(49,95,98,.55)";ctx.lineWidth=1.2;
      ctx.beginPath();ctx.moveTo(me.x,me.y);ctx.lineTo(me.x+Math.cos(a)*len,me.y+Math.sin(a)*len);ctx.stroke();
      ctx.fillStyle=TEAL;ctx.beginPath();ctx.arc(me.x+Math.cos(a)*len,me.y+Math.sin(a)*len,2,0,Math.PI*2);ctx.fill();
    }
  } else if(key==="self"||key==="lastAction"||key==="olderActions"||key==="changeRate"||key==="idle"){
    ring(me,TEAL);
    if(obs){const a=facing(me);ctx.strokeStyle=TEAL;ctx.lineWidth=2;
      ctx.beginPath();ctx.moveTo(me.x,me.y);ctx.lineTo(me.x+Math.cos(a)*scale*.9,me.y+Math.sin(a)*scale*.9);ctx.stroke();}
  } else if(key==="opponent"||key==="aimOpponent"){
    ring(them,RED);
    if(key==="aimOpponent"){const a=facing(them);ctx.strokeStyle=RED;ctx.setLineDash([6,4]);ctx.lineWidth=1.8;
      ctx.beginPath();ctx.moveTo(them.x,them.y);ctx.lineTo(them.x+Math.cos(a)*scale*4,them.y+Math.sin(a)*scale*4);ctx.stroke();ctx.setLineDash([]);}
  } else if(key==="navigation"){
    ring(me,TEAL);ring(them,RED);
    ctx.strokeStyle=TEAL;ctx.setLineDash([6,4]);
    ctx.beginPath();ctx.moveTo(me.x,me.y);ctx.lineTo(them.x,them.y);ctx.stroke();ctx.setLineDash([]);
  } else if(key==="aim"){
    const a=facing(me);const hit=obs?obs[890]>.5:false,suicide=obs?obs[891]>.5:false;
    ctx.strokeStyle=suicide?GOLD:hit?RED:TEAL;ctx.lineWidth=2.2;
    ctx.beginPath();ctx.moveTo(me.x,me.y);ctx.lineTo(me.x+Math.cos(a)*scale*4,me.y+Math.sin(a)*scale*4);ctx.stroke();
  } else if(key==="bullets"||key==="threat"||key==="threatCount"){
    for(let i=0;i<bullets;i++){
      const bx=ox+buf[bulletBase+i*2],by=oy+buf[bulletBase+i*2+1];
      const danger=obs&&latestFrame?.mask?.[i]&&obs[900+i*10+7]>.5;
      ctx.strokeStyle=danger?RED:TEAL;ctx.lineWidth=danger?2.2:1.4;
      ctx.beginPath();ctx.arc(bx,by,12,0,Math.PI*2);ctx.stroke();
    }
    ring(me,TEAL);
  } else if(key==="phase"){
    ring(me,TEAL);ring(them,RED);
  } else if(key==="dodge"&&obs){
    // 3x3 candidate grid, forward on top, drawn beside the tank.
    const cell=17,gx=me.x+scale*.55,gy=me.y-cell*1.5;
    for(let row=0;row<3;row++)for(let col=0;col<3;col++){
      const idx=(2-row)*3+col;const value=obs[1018+idx];
      ctx.fillStyle=value<0?"rgba(166,40,48,.8)":`rgba(49,95,98,${.2+.6*Math.max(0,Math.min(1,value))})`;
      ctx.fillRect(gx+col*cell,gy+row*cell,cell-2,cell-2);
    }
    ctx.strokeStyle="rgba(23,24,22,.35)";ctx.lineWidth=1;ctx.strokeRect(gx,gy,cell*3-2,cell*3-2);
  }
  ctx.restore();
}

function draw(alpha = 1) {
  if (!wasm || handle === null) return;
  const buf = renderBuffer();
  syncCanvasSize();
  const width = canvas.width, height = canvas.height;
  ctx.setTransform(width/712,0,0,height/500,0,0);
  ctx.fillStyle="#fff";ctx.fillRect(0,0,712,500);
  const mazeW=buf[0],mazeH=buf[1],scale=buf[2],half=buf[3],walls=buf[5]|0,tanks=buf[6]|0,bullets=buf[7]|0;
  const worldW=mazeW*scale,worldH=mazeH*scale;
  const ox=10+Math.max(0,(692-worldW)/2),oy=10+Math.max(0,(480-worldH)/2);
  ctx.fillStyle="#efede8";ctx.fillRect(ox,oy,worldW,worldH);
  let p=HEADER;ctx.beginPath();ctx.strokeStyle="#3f4550";ctx.lineWidth=half*2;ctx.lineCap="square";
  for(let i=0;i<walls;i++){ctx.moveTo(ox+buf[p],oy+buf[p+1]);ctx.lineTo(ox+buf[p+2],oy+buf[p+3]);p+=4;}ctx.stroke();
  const tankBase=p;p+=tanks*6;const bulletBase=p;

  // A new round teleports everything, so never interpolate across one.
  const sameRound = previousState && previousState.round === buf[9];
  // Bullets have no stable identity across the FFI boundary: the engine's Vec
  // reorders slots when one is removed, so matching by index alone would
  // interpolate two unrelated bullets and teleport-jump at kill moments. A
  // real bullet cannot move farther than one ballistic step in a frame.
  const maxStep = BULLET_SPEED * (scale / 50) * 1.5;
  const maxStepSq = maxStep * maxStep;
  ctx.fillStyle="#101214";
  for(let i=0;i<bullets;i++){
    let old = sameRound ? previousState.bullets[i] : null;
    if (old) {
      const dx = buf[bulletBase+i*2] - old.x, dy = buf[bulletBase+i*2+1] - old.y;
      if (dx*dx + dy*dy > maxStepSq) old = null;
    }
    const bx=old?old.x+(buf[bulletBase+i*2]-old.x)*alpha:buf[bulletBase+i*2];
    const by=old?old.y+(buf[bulletBase+i*2+1]-old.y)*alpha:buf[bulletBase+i*2+1];
    ctx.beginPath();ctx.arc(ox+bx,oy+by,Math.max(2,2.5*scale/50),0,Math.PI*2);ctx.fill();
  }
  for(let i=0;i<tanks;i++){
    const o=tankBase+i*6;
    if(buf[o+3]<.5)continue;
    const old = sameRound ? previousState.tanks[i] : null;
    const x=old?old.x+(buf[o]-old.x)*alpha:buf[o];
    const y=old?old.y+(buf[o+1]-old.y)*alpha:buf[o+1];
    const r=old?interpolateAngle(old.rotation,buf[o+2],alpha):buf[o+2];
    drawTank(ox+x,oy+y,r,buf[o+5],i);
  }
  drawObservationOverlay(document.body.dataset.obsGroup,buf,ox,oy,scale,mazeW,mazeH,tankBase,bulletBase,bullets);
  document.querySelector("#live-score-0").textContent=String(buf[10]|0);
  document.querySelector("#live-score-1").textContent=String(buf[11]|0);
  document.querySelector("#live-round").textContent=`round ${buf[9]|0}`;
  document.querySelector("#live-bullets").textContent=`${bullets} bullet${bullets===1?"":"s"}`;
}

function driveHybrid(target, seat = 0, publish = false) {
  const ptr=wasm.kf_hybrid_observation(target,seat);
  const len=wasm.kf_hybrid_observation_len();
  const view=new Float32Array(wasm.memory.buffer,ptr,len);
  const obs=new Float32Array(view.subarray(0,OBS_DIM));
  const mask=Array.from(view.subarray(OBS_DIM,OBS_DIM+BULLET_SLOTS),v=>v>.5);
  const dodge=obs.subarray(DODGE_OFFSET,DODGE_OFFSET+DODGE_DIM);
  const started=performance.now();
  const logits=policy.logits(obs,mask,dodge);
  let action=0;for(let i=1;i<logits.length;i++)if(logits[i]>logits[action])action=i;
  if(seat===0)samples("hybrid",performance.now()-started);
  wasm.kf_set_hybrid_action(target,seat,action);
  if (publish) {
    lastAction=action;lastLogits=Array.from(logits);
    // The whole 1028-value vector goes out, so the table can slice any field.
    window.dispatchEvent(new CustomEvent("hybrid-frame",{detail:{action,logits:lastLogits,obs,mask}}));
    document.querySelector("#live-action").textContent=`action ${action}`;
  }
}

function newMatch() {
  if (!wasm) return;
  if (handle!==null) wasm.kf_free(handle);
  const seed=(Math.random()*0xffffffff)>>>0;
  // Bit i of the mask hands tank i to the scripted Laika AI. Seat 0 is always
  // Hybrid, so the observation and action panels never go dark.
  handle=wasm.kf_new(seed,opponent==="laika"?0b10:0);
  if(opponent==="killfield")wasm.kf_attach_mpc(handle,1,seed,KILLFIELD_RAYS,0);
  previousState=capture(renderBuffer());
  accumulator=0;
  document.querySelector("#live-opponent").textContent=OPPONENT_LABEL[opponent];
}

function tick() {
  driveHybrid(handle,0,true);
  if(opponent==="hybrid")driveHybrid(handle,1,false);
  wasm.kf_step(handle);
}

function probe() {
  if (!wasm || !policy) return;
  driveHybrid(probeHybrid,0,false);wasm.kf_step(probeHybrid);
  const baseStarted=performance.now();wasm.kf_step(probeBaseline);const baseMs=performance.now()-baseStarted;
  const mpcStarted=performance.now();wasm.kf_step(probeMpc);const mpcMs=performance.now()-mpcStarted;
  samples("killfield",Math.max(.01,mpcMs-baseMs));
}

function frame(now) {
  // Ported from the viewer: bound the catch-up so a long stall (a background
  // tab, a slow load) cannot fast-forward a burst of frames on return.
  const elapsed = Math.max(0, now - previousTime);
  previousTime = now;
  if (wasm) {
    if (paused) {
      accumulator = 0;
    } else {
      const total = Math.min(accumulator + elapsed, MAX_CATCHUP_MS);
      const steps = Math.floor(total / STEP_MS);
      for (let i = 0; i < steps; i += 1) { previousState = capture(renderBuffer()); tick(); }
      accumulator = total - steps * STEP_MS;
    }
    draw(paused ? 1 : Math.min(1, accumulator / STEP_MS));
  }
  requestAnimationFrame(frame);
}

document.querySelectorAll(".model-choice").forEach((button) => {
  button.addEventListener("click", () => {
    opponent=button.dataset.opponent;
    document.querySelectorAll(".model-choice").forEach((item)=>item.classList.toggle("active",item===button));
    newMatch();
  });
});
document.querySelector("#live-reroll").addEventListener("click",newMatch);
document.querySelector("#live-pause").addEventListener("click",(event)=>{
  paused=!paused;event.currentTarget.textContent=paused?t("继续","Resume"):t("暂停","Pause");
});

async function bootLive() {
  const base=assetBase();
  const [wasmResult,loadedPolicy]=await Promise.all([
    fetch(new URL("kf_engine.wasm",base)).then((response)=>{if(!response.ok)throw new Error(`WASM HTTP ${response.status}`);return response.arrayBuffer();}).then((bytes)=>WebAssembly.instantiate(bytes,{})),
    HybridPolicy.load(new URL("assets/hybrid.json",base).href,new URL("assets/hybrid.bin",base).href),
  ]);
  wasm=wasmResult.instance.exports;policy=loadedPolicy;
  if(wasm.kf_hybrid_schema_version()!==24||wasm.kf_hybrid_observation_len()!==OBS_DIM+BULLET_SLOTS)throw new Error("schema mismatch");
  newMatch();
  probeHybrid=wasm.kf_new(0x51a17,2);
  probeMpc=wasm.kf_new(0x51a18,2);
  probeBaseline=wasm.kf_new(0x51a19,3);
  wasm.kf_attach_mpc(probeMpc,0,19,KILLFIELD_RAYS,0);
  for(let i=0;i<4;i++)probe();
  document.querySelector("#live-loading").hidden=true;
  setInterval(probe,1200);
}

try {
  const saved = localStorage.getItem("paper-lang");
  if (saved === "en") lang = "en";
} catch {}
// KaTeX renders on its own onload; re-apply once it exists so the equation
// swap has something to render with.
if (lang === "en") {
  const boot = () => applyLanguage();
  if (window.katex) boot(); else window.addEventListener("load", boot, { once: true });
}

bootLive().catch((error)=>{
  const loading=document.querySelector("#live-loading");
  loading.textContent=t("实况资源载入失败 · 请刷新或打开完整客户端","Live assets failed to load · refresh, or open the full client");
  loading.title=String(error);
});
requestAnimationFrame(frame);
