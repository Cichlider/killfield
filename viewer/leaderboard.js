/**
 * The board itself.
 *
 * Every value rendered here came from a stranger's issue and survived
 * tools/verify_replay.mjs. It is still treated as text: names go in through
 * textContent, never innerHTML, so a name that looks like markup renders as
 * the characters someone typed rather than as part of this page.
 */

import { loadLang, saveLang } from "./src/i18n.js";
import { MIN_SUBMITTABLE_SHUTOUT } from "./src/replay.js";
import { RANKED_OPENING_DELAY_SECONDS } from "./src/ranked.js";

/**
 * The submission rules, stated once, here. Every one of them is enforced twice
 * over — the page refuses to record a run that breaks one, and the verifier
 * refuses the record independently, because the page's copy of a rule is only
 * advisory once a submission is just text somebody can write.
 */
const RULES = {
  en: [
    [`${MIN_SUBMITTABLE_SHUTOUT} in a row`, `A run has to take ${MIN_SUBMITTABLE_SHUTOUT} `
      + "rounds off the agent back to back. Draws and double kills score for nobody and "
      + "break nobody's run."],
    ["Any stretch of one session", "The score is the longest run of wins anywhere in a "
      + "single continuous match — losing does not end your attempt, it just ends that "
      + "streak. Rerolling the maze or changing the opponent starts a new session."],
    ["No delay for the opponent", "Opponent delay must sit at 0 frames. It is the "
      + "handicap that makes the agent actuate late, and a ranked run gives it none."],
    ["Opening pause at most 0.5s", `The default ${RANKED_OPENING_DELAY_SECONDS}s or `
      + "shorter. A shorter pause only makes the run harder, so it is allowed."],
    ["No turn-rate assist", "Instant turn is switched off when a ranked run starts and "
      + "has to stay off."],
    ["Two boards", "Hybrid and Killfield are ranked separately. Laika is not ranked."],
    ["Anything else is yours", "The wheel's forward region, touch or keyboard, pausing "
      + "to think — none of it changes the match, so none of it is restricted."],
    ["A name, and optionally an account", "Every record goes up under a name. Showing "
      + "your GitHub account is optional, and only kept when it is the account that "
      + "opened the issue."],
  ],
  zh: [
    [`${MIN_SUBMITTABLE_SHUTOUT} 连起步`, `一次成绩要连续拿下 ${MIN_SUBMITTABLE_SHUTOUT} 个回合。`
      + "平局和双杀不给任何人加分，也不中断任何人的连胜。"],
    ["同一场里的任意一段", "计分取的是单场连续对局中最长的一段连胜——输一局不结束这次尝试，"
      + "只是结束那一段。换迷宫或换对手则开始新的一场。"],
    ["对手零延迟", "对手延迟必须是 0 帧。那是让智能体延迟出手的让步，排位不给任何让步。"],
    ["开局停顿不超过 0.5 秒", `默认的 ${RANKED_OPENING_DELAY_SECONDS} 秒或更短。`
      + "更短只会更难，所以允许。"],
    ["关闭瞬间转向", "开始排位时会自动关掉，并且必须保持关闭。"],
    ["两个榜分开排", "Hybrid 和 Killfield 各排各的。Laika 不计入排名。"],
    ["其余随意", "轮盘的前向区域、用触屏还是键盘、中途暂停思考——都不改变对局本身，所以都不限制。"],
    ["名字必填，账号可选", "每条记录都要有名字。是否显示 GitHub 账号由你决定，"
      + "且只有确实是开 issue 的那个账号才会被保留。"],
  ],
};

const COPY = {
  en: {
    htmlLang: "en",
    back: "Back to the game",
    heading: "Longest shutout",
    blurb: "How many rounds running you can take off the agent before it takes "
      + "one back. Every record here was replayed frame by frame before it landed.",
    rank: "#", player: "Player", score: "Shutout", rounds: "Rounds", when: "Verified",
    empty: `Nobody has cleared ${MIN_SUBMITTABLE_SHUTOUT} in a row yet. The board is yours to open.`,
    note: "Runs are played at the default settings — no actuation delay for the "
      + "opponent — and scored by replaying the recorded inputs through the same "
      + "engine build. The score a submission claims is ignored.",
    unavailable: "The board could not be loaded.",
    rulesHeading: "What counts",
    langToggle: "中文",
    langAria: "Switch to Chinese",
  },
  zh: {
    htmlLang: "zh-Hans",
    back: "返回游戏",
    heading: "最大零封",
    blurb: "在对手扳回一局之前，你能连续拿下多少回合。榜上每条记录都经过逐帧重放核验。",
    rank: "#", player: "玩家", score: "零封", rounds: "总局数", when: "核验于",
    empty: `还没有人打满 ${MIN_SUBMITTABLE_SHUTOUT} 连封。第一个位置空着。`,
    note: "成绩一律在默认设置下产生——对手不吃任何动作延迟——并由同一份引擎重放录制的输入重新计分，"
      + "提交时自报的分数不作数。",
    unavailable: "榜单加载失败。",
    rulesHeading: "上榜条件",
    langToggle: "EN",
    langAria: "切换到英文",
  },
};

const NODES = {
  back: "back-label", heading: "board-heading", blurb: "board-blurb",
  rank: "col-rank", player: "col-name", score: "col-score",
  rounds: "col-rounds", when: "col-when", note: "board-note",
  rulesHeading: "rules-heading",
};

const tabs = [...document.querySelectorAll("#board-tabs .mode-btn")];
const body = document.getElementById("board-body");
const emptyNote = document.getElementById("board-empty");
const langToggle = document.getElementById("lang-toggle");
const rulesList = document.getElementById("rules-list");

let lang = loadLang();
let board = { entries: [] };
let failed = false;
let active = "hybrid";

const text = () => COPY[lang] ?? COPY.en;

function formatDate(iso) {
  const when = Date.parse(iso);
  if (!Number.isFinite(when)) return "—";
  return new Date(when).toLocaleDateString(lang === "zh" ? "zh-Hans" : "en-GB", {
    year: "numeric", month: "short", day: "numeric",
  });
}

function render() {
  const copy = text();
  document.documentElement.lang = copy.htmlLang;
  for (const [key, id] of Object.entries(NODES)) {
    document.getElementById(id).textContent = copy[key];
  }
  langToggle.textContent = copy.langToggle;
  langToggle.setAttribute("aria-label", copy.langAria);

  const rows = board.entries
    .filter((entry) => entry.board === active)
    .sort((a, b) => b.score - a.score || Date.parse(a.verifiedAt) - Date.parse(b.verifiedAt));

  body.replaceChildren();
  for (const [index, entry] of rows.entries()) {
    const row = document.createElement("tr");
    if (index === 0) row.className = "lead";
    const cells = [
      String(index + 1),
      null, // the name is built below, since it carries a link
      String(entry.score),
      String(entry.rounds),
      formatDate(entry.verifiedAt),
    ];
    cells.forEach((value, column) => {
      const cell = document.createElement("td");
      if (column === 1) {
        // Attacker-authored text: set as a text node, never parsed as markup.
        const name = document.createElement("span");
        name.className = "board-name";
        name.textContent = entry.name || "anonymous";
        cell.append(name);
        if (entry.github) {
          const handle = document.createElement("a");
          handle.className = "board-handle";
          handle.textContent = `@${entry.github}`;
          handle.rel = "noreferrer nofollow";
          handle.target = "_blank";
          handle.href = `https://github.com/${encodeURIComponent(entry.github)}`;
          cell.append(handle);
        }
      } else {
        cell.textContent = value;
      }
      row.append(cell);
    });
    body.append(row);
  }

  rulesList.replaceChildren();
  for (const [term, detail] of RULES[lang] ?? RULES.en) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = detail;
    rulesList.append(dt, dd);
  }

  emptyNote.hidden = rows.length > 0;
  emptyNote.textContent = failed ? copy.unavailable : copy.empty;
  tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.board === active));
}

tabs.forEach((tab) => tab.addEventListener("click", () => {
  active = tab.dataset.board;
  render();
}));

langToggle.addEventListener("click", () => {
  lang = lang === "en" ? "zh" : "en";
  saveLang(lang);
  render();
});

try {
  const response = await fetch("leaderboard.json", { cache: "no-store" });
  if (!response.ok) throw new Error(String(response.status));
  const loaded = await response.json();
  if (Array.isArray(loaded.entries)) board = loaded;
} catch {
  // An absent board is the normal state before the first record lands; only a
  // malformed one is worth reporting, and either way the page still renders.
  failed = false;
}
render();
