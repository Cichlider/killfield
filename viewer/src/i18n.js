/**
 * UI copy, English and Chinese.
 *
 * Ported near-verbatim from killfield/src/i18n.js. Everything here is display
 * text for index.html; none of it touches the wasm engine. Names that are
 * proper nouns (killfield, Laika, Hybrid) are left as-is in both languages,
 * matching how the Chinese docs already write them.
 */

export const STRINGS = {
  en: {
    htmlLang: "en",
    nameYou: "You",
    round: (n) => `round ${n}`,
    roundOver: (n) => `round ${n} · round over`,
    modeWatch: "Watch",
    modePlay: "Play",
    reroll: "New maze (R)",
    resetScore: "Reset score",
    instantTurnOn: "Instant turn: on",
    instantTurnOff: "Instant turn: off",
    instantTurnAria: "Toggle instant joystick heading for the human player",
    pauseEnter: "Pause (P)",
    pauseExit: "Resume (P)",
    soundMute: "Mute sound",
    soundUnmute: "Unmute sound",
    paused: "paused",
    streakLine: (cur, best) => `win streak ${cur} · best ${best}`,
    watchLeftLabel: "Left",
    watchRightLabel: "Right",
    opponentLabel: "Opponent",
    forwardAlignmentLabel: "Wheel forward region",
    forwardAlignmentValue: (forward, reverse) => reverse === 0
      ? forward + "° / 360° · no reverse"
      : forward + "° / 360° · reverse " + reverse + "°",
    reactionDelayLabel: "Opponent delay",
    reactionDelayOptions: ["0 frames", "1 frame", "2 frames", "3 frames"],
    openingDelayLabel: "Opening pause",
    openingDelayValue: (seconds) => `${seconds.toFixed(1)} s`,
    openingDelayCountdown: (seconds) => `Opponent starts in ${seconds.toFixed(1)}s`,
    fullscreenEnter: "Fullscreen",
    fullscreenExit: "Exit fullscreen",
    orientationTitle: "Rotate your phone",
    orientationBody: "Turn off portrait lock, then rotate to landscape.",
    touchControls: {
      joystick: "Joystick",
      dpad: "Forward / turn",
      joystickAria: "128-direction movement joystick with configurable forward and reverse sectors",
      dpadAria: "Forward, reverse, turn left and turn right controls",
      fire: "FIRE",
      hide: "Hide touch controls",
      show: "Show touch controls",
      hideShort: "Hide",
      showShort: "Show",
    },
    langToggleLabel: "中文",
    langToggleAria: "Switch to Chinese",
  },

  zh: {
    htmlLang: "zh-Hans",
    nameYou: "你",
    round: (n) => `第 ${n} 回合`,
    roundOver: (n) => `第 ${n} 回合 · 已结束`,
    modeWatch: "观看",
    modePlay: "对战",
    reroll: "换一张迷宫 (R)",
    resetScore: "清零比分",
    instantTurnOn: "瞬间转向：开",
    instantTurnOff: "瞬间转向：关",
    instantTurnAria: "切换人类玩家轮盘瞬间转向",
    pauseEnter: "暂停 (P)",
    pauseExit: "继续 (P)",
    soundMute: "关闭音效",
    soundUnmute: "打开音效",
    paused: "已暂停",
    streakLine: (cur, best) => `连胜 ${cur} · 最佳 ${best}`,
    watchLeftLabel: "左方",
    watchRightLabel: "右方",
    opponentLabel: "对手",
    forwardAlignmentLabel: "轮盘区域规划",
    forwardAlignmentValue: (forward, reverse) => reverse === 0
      ? forward + "° / 360° · 不后退"
      : forward + "° / 360° · 后退区 " + reverse + "°",
    reactionDelayLabel: "对手延迟",
    reactionDelayOptions: ["0 帧", "1 帧", "2 帧", "3 帧"],
    openingDelayLabel: "开局停顿",
    openingDelayValue: (seconds) => `${seconds.toFixed(1)} 秒`,
    openingDelayCountdown: (seconds) => `对手将在 ${seconds.toFixed(1)} 秒后行动`,
    fullscreenEnter: "全屏",
    fullscreenExit: "退出全屏",
    orientationTitle: "请将手机横过来",
    orientationBody: "先关闭系统竖屏锁定，再把手机旋转到横屏。",
    touchControls: {
      joystick: "手柄",
      dpad: "前后左右",
      joystickAria: "一百二十八方向移动轮盘：可调前向与后退对齐范围",
      dpadAria: "前进、后退、左转、右转控制",
      fire: "开火",
      hide: "隐藏触控控制器",
      show: "显示触控控制器",
      hideShort: "隐藏",
      showShort: "显示",
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
