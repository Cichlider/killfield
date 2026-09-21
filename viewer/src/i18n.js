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
    rankedStart: "Ranked run",
    rankedStop: "End run",
    rankedIdle: "A ranked run records up to 200 rounds for replay verification.",
    rankedRecording: (stats, maxRounds) => `Recording · ${stats.wins} wins / ${stats.rounds}`
      + ` rounds · ${stats.losses} losses · ${stats.doubleKills} double KOs`
      + (stats.rounds >= maxRounds ? " · round limit reached" : ""),
    rankedFinished: (stats) => `${stats.wins} wins in ${stats.rounds} rounds. Name the run and submit.`,
    rankedTooShort: (need) => `Win at least ${need} round to submit a ranked run.`,
    rankedUpload: "Submit",
    rankedDownload: "Download replay",
    rankedSubmittingButton: "Submitting…",
    rankedSubmitting: "Submitting the replay for verification…",
    rankedSubmitted: "Submitted. Verification is running; the leaderboard will update automatically.",
    rankedNotConfigured: "Score submission is not configured yet.",
    rankedChallengeUnavailable: "Human verification is still loading. Try Submit again.",
    rankedChallengeFailed: "Human verification failed. Try Submit again.",
    rankedNetworkFailed: "Cannot reach the submission service. Your replay is still here—switch networks and press Submit again; do not refresh this page.",
    rankedLargeNetworkFailed: "This replay needs the submission service because it is too large for one GitHub Issue. It is still here—retry Submit or download the JSON before refreshing.",
    rankedGithubFallback: "Submit with GitHub",
    rankedGithubCopied: "Replay copied. Paste it into Record on the GitHub page and submit the issue; a maintainer will approve verification.",
    rankedGithubCopyFailed: "The browser could not copy the replay. Allow clipboard access, then press Submit with GitHub again.",
    rankedDownloaded: "Replay downloaded. Keep the JSON file; you can send it to the maintainer if submission fails.",
    rankedBoard: "Leaderboard",
    rankedUnit: "wins",
    rankedNameLabel: "Player name",
    rankedGithubLabel: "GitHub account (optional)",
    rankedNamePlaceholder: "Your name",
    rankedGithubPlaceholder: "GitHub (optional)",
    controlsHelp: {
      trigger: "Controls",
      title: "Desktop controls",
      forward: "Forward",
      backup: "Reverse",
      left: "Turn left",
      right: "Turn right",
      fire: "Fire",
      reroll: "new maze",
      pause: "pause",
    },
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
    rankedStart: "排位记录",
    rankedStop: "结束记录",
    rankedIdle: "排位会逐帧记录最多 200 局，榜单靠重放核验成绩。",
    rankedRecording: (stats, maxRounds) => `录制中 · ${stats.wins} 胜 / ${stats.rounds} 局`
      + ` · ${stats.losses} 负 · ${stats.doubleKills} 双亡`
      + (stats.rounds >= maxRounds ? " · 已达局数上限" : ""),
    rankedFinished: (stats) => `共 ${stats.rounds} 局，${stats.wins} 胜、${stats.losses} 负、`
      + `${stats.doubleKills} 双亡。填个名字即可提交。`,
    rankedTooShort: (need) => `至少赢 ${need} 局才能提交排位成绩。`,
    rankedUpload: "提交",
    rankedDownload: "下载录像",
    rankedSubmittingButton: "提交中…",
    rankedSubmitting: "正在提交录像并启动验分…",
    rankedSubmitted: "已提交。系统正在验分，排行榜会自动更新。",
    rankedNotConfigured: "成绩提交服务尚未配置。",
    rankedChallengeUnavailable: "人机验证仍在加载，请再次点击提交。",
    rankedChallengeFailed: "人机验证失败，请再次点击提交。",
    rankedNetworkFailed: "无法连接提交服务。录像还在本页，请切换网络后再次点“提交”，不要刷新页面。",
    rankedLargeNetworkFailed: "这份录像太大，无法放进单个 GitHub Issue，必须通过提交服务分块上传。录像还在本页；请重试“提交”，或在刷新前下载 JSON。",
    rankedGithubFallback: "改用 GitHub 提交",
    rankedGithubCopied: "录像已复制。请在打开的 GitHub 页面粘贴到 Record 并提交 Issue；管理员审核后会启动验分。",
    rankedGithubCopyFailed: "浏览器未允许复制录像。请允许剪贴板访问，再点一次“改用 GitHub 提交”。",
    rankedDownloaded: "录像已下载。请保留这个 JSON 文件；如果提交失败，可以直接发给维护者。",
    rankedBoard: "排行榜",
    rankedUnit: "胜场",
    rankedNameLabel: "玩家名字",
    rankedGithubLabel: "GitHub 账号（可选）",
    rankedNamePlaceholder: "你的名字",
    rankedGithubPlaceholder: "GitHub 账号（可选）",
    controlsHelp: {
      trigger: "操作",
      title: "电脑端按键",
      forward: "前进",
      backup: "后退",
      left: "左转",
      right: "右转",
      fire: "开火",
      reroll: "更换迷宫",
      pause: "暂停",
    },
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
