/**
 * UI copy, English and Chinese.
 *
 * Ported near-verbatim from killfield/src/i18n.js. Everything here is display
 * text for index.html; none of it touches the wasm engine. Names that are
 * proper nouns (killfield, Laika) are left as-is in both languages, matching
 * how the Chinese docs already write them.
 */

export const STRINGS = {
  en: {
    htmlLang: "en",
    tagline: "A search-based agent for a maze tank duel, now running the same Rust engine the trainer links. Most of its kills are bank shots.",
    nameYou: "You",
    round: (n) => `round ${n}`,
    roundOver: (n) => `round ${n} · round over`,
    modeWatch: "Watch it play",
    modePlay: "Play against it",
    modeSelfplay: "AI vs AI",
    reroll: "New maze (R)",
    resetScore: "Reset score",
    pauseEnter: "Pause (P)",
    pauseExit: "Resume (P)",
    soundMute: "Mute sound",
    soundUnmute: "Unmute sound",
    paused: "paused",
    streakLine: (cur, best) => `win streak ${cur} · best ${best}`,
    seedLabel: "seed",
    raysLabel: "rays",
    rays512: "512 — full",
    rays256: "256 — mobile",
    oppModelLabel: "planner assumes opponent is",
    oppModelLaika: "Laika (scripted)",
    oppModelHuman: "human (unpredictable)",
    oppModelHint: "Only changes what the lookahead simulation predicts tank 1 "
      + "will do — not who is actually driving it. Pick “Laika” in watch mode, "
      + "where that prediction is exact. Pick “human” in play mode: scripting "
      + "your moves as Laika makes the planner imagine kills that never "
      + "happen, and it can stall as if it had already won.",
    keyhelpHtml: "<strong>Arrow keys</strong> or <strong>ESDF</strong> to drive &middot; "
      + "<strong>M</strong> / <strong>Space</strong> / <strong>Q</strong> to fire &middot; "
      + "<strong>R</strong> for a new maze",
    note: "Bullets ricochet and stay lethal for ten seconds — including to whoever "
      + "fired them. A round is not decided the instant someone dies: the world "
      + "keeps running for three more seconds, so a shot already in the air can "
      + "still take the apparent winner with it.",
    fullscreenEnter: "Fullscreen",
    fullscreenExit: "Exit fullscreen",
    orientationTitle: "Rotate your phone",
    orientationBody: "Turn off portrait lock, then rotate to landscape.",
    touchControls: {
      joystick: "Joystick",
      dpad: "Forward / turn",
      joystickAria: "16-direction movement joystick: 270-degree forward sector, 90-degree reverse sector",
      dpadAria: "Forward, reverse, turn left and turn right controls",
      fire: "FIRE",
      hide: "Hide touch controls",
      show: "Show touch controls",
      hideShort: "HIDE",
      showShort: "SHOW",
    },
    tuning: {
      eyebrow: "AI lab",
      title: "Live parameters",
      description: "Changes apply to the next MPC plan and are saved in this browser.",
      reset: "Reset defaults",
      status: "live · saved locally",
      groups: {
        navigation: "Navigation & hunt",
        fire: "Fire discipline",
        safety: "Terminal & safety",
      },
      labels: {
        fieldAscentWeight: "Killfield ascent",
        fieldPeakWeight: "Peak shooting position",
        guidanceProgressWeight: "Guidance progress",
        huntChainGainWeight: "Hunt-chain gain",
        huntTimeScaleSeconds: "Hunt ramp time (s)",
        huntTimeMaxMultiplier: "Hunt max multiplier",
        alignmentWeight: "Aim alignment",
        mobilityWeight: "Net mobility",
        goodFireBonus: "Predicted-hit bonus",
        shotFlightTimeWeight: "Flight-time penalty",
        ammoReserveWeight: "Ammo reserve pressure",
        ammoFlightPressure: "Low-ammo time pressure",
        failedFirePenalty: "Miss penalty",
        suicideFirePenalty: "Suicide-shot penalty",
        activeKillTimeWeight: "Kill-speed reward",
        riskWeight: "Incoming-fire risk",
      },
    },
    telemetryLabels: {
      decision: "decision",
      planP95: "plan p95",
      fieldBuilds: "field builds",
      huntChain: "hunt chain",
      ownBulletGuard: "own-bullet guard",
      stuckEvents: "stuck events",
    },
    telemetryValue: {
      planP95: (ms) => `${ms} ms / 40 ms budget`,
      fieldBuilds: (n, ms) => `${n} @ ${ms} ms`,
      huntChain: (n, total) => `${n} (${total} total)`,
    },
    langToggleLabel: "中文",
    langToggleAria: "Switch to Chinese",
  },

  zh: {
    htmlLang: "zh-Hans",
    tagline: "浏览器里的迷宫坦克对战，配一个搜索型 AI 对手——现在跑的是训练用的同一份 Rust 引擎。它的击杀大多靠反弹。",
    nameYou: "你",
    round: (n) => `第 ${n} 回合`,
    roundOver: (n) => `第 ${n} 回合 · 已结束`,
    modeWatch: "看它打",
    modePlay: "和它打",
    modeSelfplay: "AI 对 AI",
    reroll: "换一张迷宫 (R)",
    resetScore: "清零比分",
    pauseEnter: "暂停 (P)",
    pauseExit: "继续 (P)",
    soundMute: "关闭音效",
    soundUnmute: "打开音效",
    paused: "已暂停",
    streakLine: (cur, best) => `连胜 ${cur} · 最佳 ${best}`,
    seedLabel: "种子",
    raysLabel: "射线",
    rays512: "512 — 完整",
    rays256: "256 — 移动端",
    oppModelLabel: "规划器假设对手是",
    oppModelLaika: "Laika（脚本）",
    oppModelHuman: "人类（不可预测）",
    oppModelHint: "这个选项只改变规划器前瞻推演时对 1 号坦克行为的预测，"
      + "不改变实际操作者。看它打 Laika 时选「Laika」，这个预测是准确的；"
      + "自己上场时要选「人类」——如果还按 Laika 的脚本去想象对手，"
      + "规划器会想象出一堆从未发生的击杀，然后像已经赢了一样停下来不动。",
    keyhelpHtml: "<strong>方向键</strong>或 <strong>ESDF</strong> 控制移动 &middot; "
      + "<strong>M</strong> / <strong>空格</strong> / <strong>Q</strong> 开火 &middot; "
      + "<strong>R</strong> 换一张迷宫",
    note: "子弹会反弹，十秒内都有杀伤力——包括对开枪的人自己。"
      + "回合不会在有人死亡的瞬间结束：世界还会继续跑三秒，"
      + "飞行中的子弹仍可能把看起来已经获胜的一方也带走。",
    fullscreenEnter: "全屏",
    fullscreenExit: "退出全屏",
    orientationTitle: "请将手机横过来",
    orientationBody: "先关闭系统竖屏锁定，再把手机旋转到横屏。",
    touchControls: {
      joystick: "手柄",
      dpad: "前后左右",
      joystickAria: "十六方向移动轮盘：270 度前进区，90 度后退区",
      dpadAria: "前进、后退、左转、右转控制",
      fire: "开火",
      hide: "隐藏触控控制器",
      show: "显示触控控制器",
      hideShort: "隐藏",
      showShort: "显示",
    },
    tuning: {
      eyebrow: "AI 实验室",
      title: "实时参数",
      description: "修改会从下一次 MPC 规划开始生效，并保存在当前浏览器。",
      reset: "恢复默认值",
      status: "实时生效 · 本地保存",
      groups: {
        navigation: "导航与追猎",
        fire: "开火纪律",
        safety: "终局与安全",
      },
      labels: {
        fieldAscentWeight: "Killfield 上坡",
        fieldPeakWeight: "射击位峰值",
        guidanceProgressWeight: "引导进展",
        huntChainGainWeight: "追猎链收益",
        huntTimeScaleSeconds: "追猎爬升时间（秒）",
        huntTimeMaxMultiplier: "追猎最高倍率",
        alignmentWeight: "炮口对准",
        mobilityWeight: "有效位移",
        goodFireBonus: "预计命中奖励",
        shotFlightTimeWeight: "飞行时间惩罚",
        ammoReserveWeight: "弹夹储备压力",
        ammoFlightPressure: "低弹时间压力",
        failedFirePenalty: "打空惩罚",
        suicideFirePenalty: "自杀弹惩罚",
        activeKillTimeWeight: "击杀速度奖励",
        riskWeight: "来袭火力风险",
      },
    },
    telemetryLabels: {
      decision: "决策",
      planP95: "规划耗时 p95",
      fieldBuilds: "场构建次数",
      huntChain: "猎杀链",
      ownBulletGuard: "自弹规避",
      stuckEvents: "卡墙次数",
    },
    telemetryValue: {
      planP95: (ms) => `${ms} ms / 预算 40 ms`,
      fieldBuilds: (n, ms) => `${n} 次 @ ${ms} ms`,
      huntChain: (n, total) => `${n}（共 ${total}）`,
    },
    langToggleLabel: "EN",
    langToggleAria: "切换到英文",
  },
};

const STORAGE_KEY = "killfield-lang";

export function loadLang() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "en" || saved === "zh") return saved;
  } catch {
    // localStorage can throw in locked-down contexts; default below.
  }
  return "en";
}

export function saveLang(lang) {
  try {
    localStorage.setItem(STORAGE_KEY, lang);
  } catch {
    // Non-fatal: language just won't persist across reloads.
  }
}
